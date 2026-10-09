"""Orchestrates the engines into one auditable analysis run.

The function :func:`run_analysis` only wires the engines together; mathematics
lives in the engine modules. Every step records timing, warnings and the
parameters used, so the run can be reproduced from the manifest.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from config import APP_VERSION, AppConfig
from core.backtesting.overfitting import StrategyRegistry
from core.backtesting.walk_forward import (
    BacktestResult,
    BacktestSettings,
    lookahead_invariance_check,
    walk_forward,
)
from core.covariance.estimators import CovarianceResult, estimate_cov
from core.covariance.estimators import evaluate_oos as cov_oos
from core.data_loader import load_price_dataset, load_synthetic_dataset
from core.estimators.expected_returns import estimate as estimate_mu
from core.estimators.expected_returns import evaluate_oos as mu_oos
from core.extreme_risk.evt import evt_analysis
from core.extreme_risk.reverse_stress import (
    empirical_breach_frequency,
    gaussian_reverse_stress,
    scenario_multipliers,
)
from core.extreme_risk.stress import (
    default_scenarios,
    historical_worst_windows,
    liquidity_analysis,
    named_crisis_replay,
    parametric_stress_var,
    run_scenarios,
)
from core.extreme_risk.tail_dependence import empirical_lower_tail_dependence, fit_t_copula
from core.extreme_risk.var_es import RiskEstimate, fit_student_t, var_es_table
from core.factors.models import factor_exposures, pca_factors
from core.market_data.base import DataUnavailableError
from core.market_data.cache import SeriesCache
from core.market_data.quality import QualitySettings
from core.market_data.yfinance_provider import YFinanceProvider
from core.metrics import (
    concentration,
    diversification_ratio,
    log_returns,
    performance_summary,
    risk_contributions,
    simple_returns,
)
from core.model_validation import independent_validator as iv
from core.model_validation.model_selection import (
    COV_COMPLEXITY,
    MU_COMPLEXITY,
    select_simplest,
    select_strategy,
    select_vol_model,
)
from core.model_validation.var_backtest import backtest_var_models
from core.optimization import optimizers as opt
from core.optimization.constraints import build_constraints, check_feasibility
from core.optimization.robust import resampled_weights, robust_mean_variance, sensitivity_analysis
from core.profiler import assess_profile
from core.recommendation.asset_analysis import asset_statistics, explain_roles
from core.regimes.models import (
    markov_summary,
    select_n_states,
    state_conditional_moments,
    volatility_rule_regimes,
)
from core.simulation.engine import CashflowPlan, SimulationResult, SimulationSettings, simulate_portfolio
from core.simulation.models import (
    BlockBootstrapModel,
    BootstrapModel,
    CCCGarchModel,
    GBMModel,
    RegimeSwitchingModel,
    StudentTModel,
)
from core.volatility.models import compare_volatility_models
from models.investor import InvestorProfileInput, ProfileAssessment
from models.market_data import PriceDataset
from models.portfolio import OptimizationResult, PortfolioConstraints
from utils.logging_config import get_logger
from utils.validation import InsufficientDataError

log = get_logger("engine")


@dataclass
class AnalysisResult:
    run_id: str
    started_at_utc: str
    config: AppConfig
    profile_input: InvestorProfileInput
    profile: ProfileAssessment
    dataset: PriceDataset
    returns: pd.DataFrame
    log_returns: pd.DataFrame
    asset_stats: pd.DataFrame
    mu: pd.Series
    cov: CovarianceResult
    constraints: PortfolioConstraints
    feasibility: dict[str, object]
    candidates: dict[str, OptimizationResult]
    frontier: pd.DataFrame
    selection: dict[str, object]
    selected_name: str
    selected: OptimizationResult
    risk_contrib: pd.DataFrame
    var_table: pd.DataFrame
    sections: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    validation: iv.ValidationReport | None = None
    finished_at_utc: str = ""
    mc_main: SimulationResult | None = None
    mc_profile: SimulationResult | None = None
    backtest: BacktestResult | None = None


class _Timer:
    def __init__(self, store: dict[str, float], key: str) -> None:
        self.store, self.key = store, key

    def __enter__(self) -> None:
        self.t0 = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self.store[self.key] = round(time.perf_counter() - self.t0, 3)


def load_data(cfg: AppConfig) -> tuple[PriceDataset, object | None]:
    qs = QualitySettings(cfg.min_history, cfg.max_missing_fraction, cfg.frozen_price_run,
                         cfg.suspicious_return_abs)
    if cfg.offline:
        return load_synthetic_dataset(seed=cfg.seed, quality=qs)
    prov = YFinanceProvider(cache=SeriesCache(cfg.cache_dir, cfg.cache_ttl_hours),
                            timeout_s=cfg.request_timeout_s, max_retries=cfg.max_retries,
                            min_interval_s=cfg.min_seconds_between_requests)
    syms = list(cfg.tickers)
    if cfg.benchmark and cfg.benchmark not in syms:
        syms.append(cfg.benchmark)
    ds = load_price_dataset(syms, prov, period=cfg.period, interval=cfg.interval,
                            base_currency=cfg.base_currency, quality=qs)
    return ds, None


def _benchmark_symbol(cfg: AppConfig, ds: PriceDataset) -> str | None:
    if cfg.offline:
        return "SINT_ACOES_BR"
    return cfg.benchmark if cfg.benchmark in ds.symbols else None


def _strategies(pc: PortfolioConstraints, cfg: AppConfig, rf: float, mu_method: str, cov_method: str,
                benchmark: str | None, symbols: list[str]) -> dict[str, object]:
    P = cfg.trading_days
    s = opt.SolverSettings(n_starts=3, seed=cfg.seed)

    def wrap(fn):  # noqa: ANN001, ANN202
        def strat(train: pd.DataFrame, prev: np.ndarray | None) -> np.ndarray:
            res = fn(train)
            if not res.success:
                raise ValueError(f"otimização falhou: {res.status}")
            return res.weights.to_numpy()
        return strat

    ew = equal_weight_feasible(pc)

    def cov(train: pd.DataFrame) -> pd.DataFrame:
        return estimate_cov(train, cov_method, P).cov

    strategies: dict[str, object] = {
        "equal_weight": lambda train, prev: ew,
        "min_variance": wrap(lambda t: opt.min_variance(cov(t), pc, settings=s)),
        "risk_parity": wrap(lambda t: opt.risk_parity(cov(t), pc, settings=s)),
        "max_diversification": wrap(lambda t: opt.max_diversification(cov(t), pc, settings=s)),
        "max_sharpe": wrap(lambda t: opt.max_sharpe(estimate_mu(t, mu_method, P), cov(t), pc, rf, s)),
        "min_cvar": wrap(lambda t: opt.min_cvar(t.to_numpy(), pc, 0.95)),
        "robust_mean_variance": wrap(lambda t: robust_mean_variance(
            estimate_mu(t, mu_method, P), cov(t), pc, len(t), rf, settings=s)),
        "hrp": wrap(lambda t: opt.hrp(cov(t), pc)),
    }
    if benchmark is not None:
        wb = np.array([1.0 if x == benchmark else 0.0 for x in symbols])
        strategies["benchmark"] = lambda train, prev: wb
    return strategies


def equal_weight_feasible(pc: PortfolioConstraints) -> np.ndarray:
    """Closest feasible portfolio to 1/N (least squares) — the constrained naive baseline."""
    n = pc.n
    target = np.full(n, 1.0 / n)
    res = opt.solve("equal_weight", lambda w: float(np.sum((w - target) ** 2)), lambda w: 2 * (w - target),
                    pc, None, np.eye(n), 0.0, opt.SolverSettings(n_starts=2))
    if not res.success:
        raise ValueError("no feasible equal-weight projection")
    return res.weights.to_numpy()


def run_analysis(cfg: AppConfig, profile_input: InvestorProfileInput) -> AnalysisResult:
    cfg.ensure_dirs()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    started = datetime.now(timezone.utc).isoformat()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    P = cfg.trading_days
    rf = float(cfg.risk_free_rate)  # type: ignore[arg-type]
    if cfg.rf_is_fallback:
        warnings.append(f"Taxa livre de risco {rf:.2%} é FALLBACK DE DEMONSTRAÇÃO para {cfg.base_currency}, "
                        "não uma cotação; informe --risk-free-rate para análises reais.")
    sections: dict[str, object] = {}

    with _Timer(timings, "profile"):
        prof = assess_profile(profile_input)
    with _Timer(timings, "data"):
        ds, synth_truth = load_data(cfg)
    if ds.is_synthetic:
        warnings.append("Execução em modo OFFLINE com DADOS SINTÉTICOS determinísticos: nenhum número "
                        "deste relatório descreve ativos reais.")
    rets = simple_returns(ds.prices)
    lrets = log_returns(ds.prices)
    bench = _benchmark_symbol(cfg, ds)
    if bench is None:
        warnings.append("Benchmark indisponível: beta e comparação com benchmark não calculados.")

    # ------------------------------------------------------------- research
    with _Timer(timings, "asset_stats"):
        astats = asset_statistics(rets, rf, P, rets[bench] if bench else None)
    with _Timer(timings, "estimator_selection"):
        mu_eval = mu_oos(rets, lookback=252, horizon=63, step=63, periods=P)
        cov_eval = cov_oos(rets, lookback=252, horizon=63, step=63, periods=P)
        sel_mu = select_simplest(mu_eval, "mse", "mse_se", MU_COMPLEXITY) if not mu_eval.empty else {}
        sel_cov = select_simplest(cov_eval, "gmv_realized_vol", "gmv_realized_vol_se", COV_COMPLEXITY) \
            if not cov_eval.empty else {}
        sections["estimator_selection"] = {"mu_oos": mu_eval, "cov_oos": cov_eval,
                                           "mu_selection": sel_mu, "cov_selection": sel_cov,
                                           "configured_mu": cfg.expected_return_method.value,
                                           "configured_cov": cfg.covariance_method.value}
    mu_method = cfg.expected_return_method.value
    cov_method = cfg.covariance_method.value
    mu = estimate_mu(rets, mu_method, P)
    covr = estimate_cov(rets, cov_method, P)
    warnings += [f"Covariância: {w}" for w in covr.warnings]
    if sel_cov and sel_cov.get("selected") != cov_method:
        warnings.append(f"Avaliação fora da amostra sugere covariância '{sel_cov['selected']}' "
                        f"(configurada: '{cov_method}'). Mantida a configuração; ver seção de estimadores.")
    se_mu = np.sqrt(np.diag(covr.cov.to_numpy()) * P / len(rets))
    sections["mu_uncertainty"] = pd.DataFrame({"mu": mu, "se_annual": se_mu,
                                               "t_stat": mu.to_numpy() / se_mu}, index=mu.index)
    if np.all(np.abs(mu.to_numpy() / se_mu) < 2):
        warnings.append("Nenhum retorno esperado é estatisticamente distinto de zero (|t|<2): "
                        "carteiras dependentes de mu (máx. Sharpe) são frágeis.")

    # ------------------------------------------------------------- constraints
    pc = build_constraints(ds.symbols, ds.meta, prof.research_limits, max_weight=cfg.max_weight,
                           min_weight=cfg.min_weight, excluded=profile_input.excluded_assets,
                           excluded_classes=profile_input.excluded_asset_classes)
    feas = check_feasibility(pc)
    feas_d = {"feasible": feas.feasible, "conflicts": feas.conflicts, "suggestions": feas.suggestions,
              "violations_of_nearest": feas.violations}
    if not feas.feasible:
        warnings.append("Restrições do perfil INVIÁVEIS para este universo: " + "; ".join(feas.conflicts))

    # ------------------------------------------------------------- portfolios
    cands: dict[str, OptimizationResult] = {}
    cov = covr.cov
    with _Timer(timings, "optimization"):
        cands["equal_weight_naive"] = opt.equal_weight(pc, mu, cov, rf)
        if feas.feasible:
            ewf = equal_weight_feasible(pc)
            w = pd.Series(ewf, index=pc.symbols)
            vol = float(np.sqrt(ewf @ cov.to_numpy() @ ewf))
            cands["equal_weight"] = OptimizationResult(
                "equal_weight", w, True, "projeção viável de 1/N", float(ewf @ mu.to_numpy()), vol,
                (float(ewf @ mu.to_numpy()) - rf) / vol, True)
        cands["min_variance"] = opt.min_variance(cov, pc, mu, rf)
        cands["max_sharpe"] = opt.max_sharpe(mu, cov, pc, rf)
        cands["risk_parity"] = opt.risk_parity(cov, pc, mu, rf)
        cands["max_diversification"] = opt.max_diversification(cov, pc, mu, rf)
        cands["hrp"] = opt.hrp(cov, pc, mu, rf)
        cands["min_cvar"] = opt.min_cvar(rets.to_numpy(), pc, 0.95, mu, cov, rf)
        cands["robust_mean_variance"] = robust_mean_variance(mu, cov, pc, len(rets), rf, periods=P)
        band = prof.research_limits.target_volatility_band
        cands["target_volatility"] = opt.target_volatility(mu, cov, pc, band[1], rf)
        frontier = opt.efficient_frontier(mu, cov, pc, rf, n_points=25) if feas.feasible else pd.DataFrame()

    # ------------------------------------------------------------- backtest + selection
    registry = StrategyRegistry(cfg.logs_dir / f"strategy_registry_{run_id}.jsonl")
    bt = None
    selection: dict[str, object] = {}
    if feas.feasible and cfg.run_heavy_analyses:
        with _Timer(timings, "backtest"):
            bset = BacktestSettings(cfg.backtest_lookback, cfg.rebalance_every,
                                    cfg.transaction_cost_bps + cfg.spread_bps, cfg.slippage_bps,
                                    cfg.validation_frac, rf, P)
            strats = _strategies(pc, cfg, rf, mu_method, cov_method, bench, ds.symbols)
            bt = walk_forward(rets, strats, bset, point_in_time=False, synthetic=ds.is_synthetic)
            for name in bt.strategies:
                registry.record(name, {"mu": mu_method, "cov": cov_method, "lookback": bset.lookback,
                                       "rebalance": bset.rebalance_every, "cost_bps": bset.cost_bps},
                                bt.metrics_validation.loc[name].to_dict(), ds.content_hash(), "validation")
            elig = [k for k in bt.strategies if k in cands and cands[k].success and cands[k].feasible]
            selection = select_strategy(bt, "equal_weight", elig)
            look = lookahead_invariance_check(rets, strats["min_variance"], bset, cut=len(rets) // 2)
            # cost sensitivity of the selected strategy
            cost_rows = []
            sel_fn = strats[selection["selected"]]
            for c in (0.0, bset.cost_bps, 3 * bset.cost_bps):
                b2 = BacktestSettings(**{**bset.__dict__, "cost_bps": c})
                from core.backtesting.walk_forward import run_strategy
                r2 = run_strategy("x", sel_fn, rets, b2)
                cost_rows.append({"cost_bps": c + b2.slippage_bps,
                                  **{k: v for k, v in performance_summary(r2.net_returns.to_numpy(), rf, P).items()
                                     if k in ("cagr", "sharpe", "max_drawdown")}})
            sections["backtest_cost_sensitivity"] = pd.DataFrame(cost_rows)
            sections["lookahead_check"] = look
    selected_name = str(selection.get("selected", "min_variance"))
    if selected_name not in cands or not cands[selected_name].success:
        warnings.append(f"Estratégia selecionada '{selected_name}' indisponível; usada 'min_variance'.")
        selected_name = "min_variance"
    if selected_name == "benchmark":
        selected_name = "equal_weight"
    selected = cands[selected_name]
    if not selection:
        selection = {"selected": selected_name, "rule": "fallback: mínima variância (backtest não executado)"}
    sw = selected.weights

    # ------------------------------------------------------------- risk
    with _Timer(timings, "risk"):
        rc = risk_contributions(sw.to_numpy(), cov.to_numpy(), list(sw.index))
        port_r = pd.Series(rets.to_numpy() @ sw.to_numpy(), index=rets.index, name="portfolio")
        vt = var_es_table(port_r.to_numpy(), cfg.confidence_levels, "1 dia útil")
        var_df = pd.DataFrame([e.as_dict() for e in vt])
        conc = concentration(sw.to_numpy(), {s: ds.meta[s].asset_class for s in sw.index}, list(sw.index))
        conc["diversification_ratio"] = diversification_ratio(sw.to_numpy(), cov.to_numpy())
        sections["concentration"] = conc
        sections["portfolio_hist_performance"] = performance_summary(port_r.to_numpy(), rf, P)
        # distribution comparison
        x = port_r.to_numpy()
        tfit = fit_student_t(x)
        mu_n, sd_n = float(x.mean()), float(x.std(ddof=1))
        ll_n = float(np.sum(stats.norm.logpdf(x, mu_n, sd_n)))
        half = len(x) // 2
        t1, t2 = fit_student_t(x[:half]), fit_student_t(x[half:])
        sections["distribution"] = {
            "normal": {"loglik": ll_n, "aic": 4 - 2 * ll_n, "bic": 2 * np.log(len(x)) - 2 * ll_n},
            "student_t": {**tfit, "aic": 6 - 2 * tfit["loglik"], "bic": 3 * np.log(len(x)) - 2 * tfit["loglik"]},
            "skew": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x)),
            "jarque_bera_p": float(stats.jarque_bera(x).pvalue),
            "stability_df_halves": [t1["df"], t2["df"]],
            "stability_vol_halves": [float(x[:half].std(ddof=1) * np.sqrt(P)), float(x[half:].std(ddof=1) * np.sqrt(P))],
        }

    # ------------------------------------------------------------- extreme risk
    with _Timer(timings, "black_swan"):
        # threshold must lie below the target level: u = q_{0.90} for a <= 0.95, q_{0.95} otherwise
        evt_res = {a: evt_analysis(-port_r.to_numpy(), a, 0.90 if a <= 0.95 else 0.95,
                                   cfg.evt_min_exceedances, seed=cfg.seed,
                                   sensitivity_quantiles=(0.85, 0.90, 0.925) if a <= 0.95 else (0.90, 0.925, 0.95, 0.975))
                   for a in cfg.confidence_levels}
        sections["evt"] = evt_res
        scen = default_scenarios()
        base_w = cands.get("equal_weight", cands["equal_weight_naive"]).weights
        stress_tbl = run_scenarios(sw, ds.meta, profile_input.capital, scen, baseline=base_w)
        sections["stress"] = stress_tbl
        sections["stress_shocks"] = pd.DataFrame({r: stress_tbl.loc[r, "shocks"] for r in stress_tbl.index}).T
        sections["historical_stress"] = historical_worst_windows(port_r)
        sections["named_crises"] = named_crisis_replay(ds.prices, sw, ds.is_synthetic)
        cov_d = cov.to_numpy() / P
        sections["parametric_stress"] = parametric_stress_var(sw.to_numpy(), cov_d, 0.99)
        adv = None
        if ds.volumes is not None:
            adv = (ds.volumes * ds.prices).iloc[-63:].mean()
        sections["liquidity"] = liquidity_analysis(sw, profile_input.capital, adv,
                                                   spread_bps={s: cfg.spread_bps for s in sw.index})
        limit = prof.research_limits.max_drawdown_reference
        h = 21
        rs = gaussian_reverse_stress(sw, rets.mean() * h, rets.cov() * h, limit, t_df=tfit["df"])
        sections["reverse_stress"] = {
            "gaussian": rs, "scenario_multipliers": scenario_multipliers(sw, ds.meta, limit, scen),
            "empirical": empirical_breach_frequency(port_r, limit, h), "horizon_days": h}
        try:
            tc = fit_t_copula(rets)
            sections["tail_dependence"] = {"t_copula": tc, "empirical_lower_5pct": empirical_lower_tail_dependence(rets)}
        except InsufficientDataError as e:
            sections["tail_dependence"] = {"error": str(e)}

    # ------------------------------------------------------------- research extras
    if cfg.run_heavy_analyses:
        with _Timer(timings, "volatility_models"):
            try:
                vc = compare_volatility_models(port_r.to_numpy(), cfg.ewma_lambda)
                vc["selection"] = select_vol_model(vc)
                sections["volatility_models"] = vc
            except InsufficientDataError as e:
                sections["volatility_models"] = {"error": str(e)}
        with _Timer(timings, "var_backtest"):
            try:
                sections["var_backtest"] = pd.concat([backtest_var_models(port_r, a, refit_every=10)
                                                      for a in cfg.confidence_levels])
            except InsufficientDataError as e:
                sections["var_backtest"] = str(e)
        with _Timer(timings, "regimes"):
            signal = rets[bench] if bench else rets.mean(axis=1)
            tbl, fits = select_n_states(signal.to_numpy(), seed=cfg.seed)
            reg: dict[str, object] = {"selection_table": tbl, "signal": bench or "média igual dos ativos"}
            if fits:
                kbest = int(tbl["bic"].astype(float).idxmin())
                reg["k_best_bic"] = kbest
                if 2 in fits:
                    f2 = fits[2]
                    reg["_hmm2_fit"] = f2
                    reg["hmm2"] = {"vol_ann": np.sqrt(f2.variances * P).tolist(), "transition": f2.transition.tolist(),
                                   "expected_duration": f2.expected_durations}
                    filt = f2.filtered.argmax(axis=1)
                    smooth = f2.smoothed.argmax(axis=1)
                    reg["filtered_vs_smoothed_agreement"] = float(np.mean(filt == smooth))
                    if synth_truth is not None:
                        truth = synth_truth.regimes.reindex(rets.index).to_numpy()  # type: ignore[attr-defined]
                        reg["accuracy_vs_truth_filtered"] = float(np.mean(filt == truth))
                        reg["accuracy_vs_truth_smoothed"] = float(np.mean(smooth == truth))
                        reg["truth_stress_share"] = float(np.mean(truth))
                rule = volatility_rule_regimes(signal)
                reg["vol_rule_markov"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                          for k, v in markov_summary(rule).items()}
                if synth_truth is not None:
                    tr = synth_truth.regimes.reindex(rule.dropna().index).to_numpy()  # type: ignore[attr-defined]
                    reg["vol_rule_accuracy_vs_truth"] = float(np.mean(rule.dropna().to_numpy() == tr))
            sections["regimes"] = reg
        with _Timer(timings, "factors"):
            try:
                pca = pca_factors(rets, 3)
                fac = {"pca_explained": pca["explained"], "pca_loadings": pca["loadings"]}
                if bench:
                    fe = factor_exposures(rets, rets[[bench]].rename(columns={bench: "MKT"}), P)
                    fac["market_exposures"] = fe
                    fac["portfolio_beta"] = float(sum(sw.get(s, 0) * fe.loc[s, "beta_MKT"] for s in fe.index)
                                                  + sw.get(bench, 0.0))
                    fac["factor_note"] = (f"fator de mercado = retornos de {bench} (proxy empírico do próprio "
                                          "universo; não é fator acadêmico)")
                sections["factors"] = fac
            except InsufficientDataError as e:
                sections["factors"] = {"error": str(e)}
        with _Timer(timings, "robustness"):
            def ms(m_, c_, p_):  # noqa: ANN001, ANN202
                return opt.max_sharpe(m_, c_, p_, rf, opt.SolverSettings(n_starts=3, seed=cfg.seed))
            rob: dict[str, object] = {}
            if feas.feasible:
                rob["max_sharpe_sensitivity"] = sensitivity_analysis(rets, ms, pc, mu_method, cov_method, rf, P,
                                                                     n_mu_draws=30, seed=cfg.seed)
                if cfg.n_resamples > 0:
                    rr = resampled_weights(rets, ms, pc, mu_method, cov_method, cfg.n_resamples, seed=cfg.seed)
                    rob["max_sharpe_resampling"] = rr
                    w_avg = rr.average_portfolio.to_numpy()
                    if np.all(np.isfinite(w_avg)):
                        vol = float(np.sqrt(w_avg @ cov.to_numpy() @ w_avg))
                        er = float(w_avg @ mu.to_numpy())
                        cands["resampled_max_sharpe"] = OptimizationResult(
                            "resampled_max_sharpe", rr.average_portfolio, True, "média de reamostragens",
                            er, vol, (er - rf) / vol, not opt.verify_weights(w_avg, pc),
                            violations=opt.verify_weights(w_avg, pc))
                if selected_name not in ("max_sharpe", "equal_weight"):
                    fn_map = {"min_variance": lambda m_, c_, p_: opt.min_variance(c_, p_, m_, rf),
                              "risk_parity": lambda m_, c_, p_: opt.risk_parity(c_, p_, m_, rf),
                              "max_diversification": lambda m_, c_, p_: opt.max_diversification(c_, p_, m_, rf),
                              "robust_mean_variance": lambda m_, c_, p_: robust_mean_variance(m_, c_, p_, len(rets), rf),
                              "hrp": lambda m_, c_, p_: opt.hrp(c_, p_, m_, rf),
                              "min_cvar": None}
                    fn = fn_map.get(selected_name)
                    if fn is not None:
                        rob["selected_sensitivity"] = sensitivity_analysis(rets, fn, pc, mu_method, cov_method,
                                                                           rf, P, n_mu_draws=30, seed=cfg.seed)
            sections["robustness"] = rob

    # ------------------------------------------------------------- Monte Carlo
    with _Timer(timings, "monte_carlo"):
        gbm = GBMModel.from_log_returns(lrets, P)
        sw_np = sw.to_numpy()
        models: dict[str, object] = {"gbm": gbm}
        try:
            models["student_t"] = StudentTModel(gbm.mu, gbm.cov, max(tfit["df"], 2.5))
        except ValueError:
            pass
        models["bootstrap"] = BootstrapModel(rets.to_numpy())
        models["block_bootstrap"] = BlockBootstrapModel(rets.to_numpy(), 21)
        if cfg.run_heavy_analyses:
            try:
                models["garch"] = CCCGarchModel.fit(rets)
            except InsufficientDataError:
                pass
            reg = sections.get("regimes", {})
            if isinstance(reg, dict) and "hmm2" in reg:
                f2 = reg["_hmm2_fit"]
                means, covs = state_conditional_moments(rets, f2.smoothed)
                last = f2.filtered[-1]
                models["regime"] = RegimeSwitchingModel(means, covs, f2.transition, last / last.sum())
        cf = CashflowPlan(profile_input.monthly_contribution, profile_input.monthly_withdrawal, 21,
                          cfg.transaction_cost_bps)
        main_model = models.get(cfg.simulation_model.value, gbm)
        if cfg.simulation_model.value not in models:
            warnings.append(f"Modelo de simulação '{cfg.simulation_model.value}' indisponível; usado GBM.")
        ss = SimulationSettings(n_paths=cfg.n_simulations, n_steps=cfg.horizon_days, seed=cfg.seed,
                                initial_capital=profile_input.capital, rebalance_every=21,
                                rebalance_cost_bps=cfg.transaction_cost_bps + cfg.spread_bps,
                                inflation_annual=cfg.inflation_annual, periods_per_year=P,
                                batch_size=cfg.simulation_batch_size, goal=cfg.goal,
                                confidence_levels=tuple(cfg.confidence_levels))
        mc_main = simulate_portfolio(main_model, sw_np, ss, cf)  # type: ignore[arg-type]
        comp = []
        for name, m in models.items():
            r_ = simulate_portfolio(m, sw_np, SimulationSettings(**{**ss.__dict__, "n_paths": min(5000, cfg.n_simulations),
                                                                    "seed": cfg.seed + 1}), cf)  # type: ignore[arg-type]
            sm = r_.summary
            comp.append({"model": name, "median_terminal": sm["terminal_wealth_percentiles"]["p50"],
                         "p05_terminal": sm["terminal_wealth_percentiles"]["p5"],
                         "p_negative_return": sm["p_negative_return"],
                         "var95_horizon": sm["var_es_horizon"]["levels"]["0.95"]["var"],
                         "es95_horizon": sm["var_es_horizon"]["levels"]["0.95"]["es"],
                         "es99_1step": sm["var_es_1step"]["levels"]["0.99"]["es"],
                         "mdd_median": sm["max_drawdown_unit_median"]})
        from core.simulation.models import stress_covariance
        stressed = GBMModel(gbm.mu, stress_covariance(gbm.cov, 0.5, 1.5), gbm.dt, name="gbm_stressed_corr")
        r_ = simulate_portfolio(stressed, sw_np, SimulationSettings(**{**ss.__dict__, "n_paths": min(5000, cfg.n_simulations),
                                                                       "seed": cfg.seed + 1}), cf)
        sm = r_.summary
        comp.append({"model": "gbm_corr+50%_vol x1.5 (cenário)", "median_terminal": sm["terminal_wealth_percentiles"]["p50"],
                     "p05_terminal": sm["terminal_wealth_percentiles"]["p5"], "p_negative_return": sm["p_negative_return"],
                     "var95_horizon": sm["var_es_horizon"]["levels"]["0.95"]["var"],
                     "es95_horizon": sm["var_es_horizon"]["levels"]["0.95"]["es"],
                     "es99_1step": sm["var_es_1step"]["levels"]["0.99"]["es"], "mdd_median": sm["max_drawdown_unit_median"]})
        sections["mc_model_comparison"] = pd.DataFrame(comp).set_index("model")
        mc_profile = None
        if cfg.profile_horizon_simulation:
            yrs = profile_input.horizon_years
            steps = int(round(yrs * P))
            infl = cfg.inflation_annual
            goal = cfg.goal
            if goal is None:
                months = int(steps // 21)
                goal = profile_input.capital * (1 + infl) ** yrs + sum(
                    profile_input.monthly_contribution * (1 + infl) ** ((steps - 21 * (m + 1)) / P)
                    for m in range(months))
            sp = SimulationSettings(**{**ss.__dict__, "n_steps": steps, "goal": goal,
                                       "record_every": 21})
            mc_profile = simulate_portfolio(main_model, sw_np, sp, cf)  # type: ignore[arg-type]
            mc_profile.summary["goal_definition"] = (
                "patrimônio final nominal >= capital inicial e aportes corrigidos pela inflação HIPOTÉTICA "
                f"de {infl:.1%} a.a." if cfg.goal is None else "objetivo informado pelo usuário")
        sections["mc_gbm_params"] = gbm.describe()

    # ------------------------------------------------------------- per-asset roles
    roles = explain_roles(sw, rc, port_r, rets, ds.meta, pd.Series(pc.upper, index=pc.symbols),
                          sections.get("stress_shocks"), {s: f.value for s, f in ds.quality.per_symbol_flag.items()},
                          (ds.volumes * ds.prices).iloc[-63:].mean() if ds.volumes is not None else None)
    sections["asset_roles"] = roles

    result = AnalysisResult(
        run_id=run_id, started_at_utc=started, config=cfg, profile_input=profile_input, profile=prof,
        dataset=ds, returns=rets, log_returns=lrets, asset_stats=astats, mu=mu, cov=covr, constraints=pc,
        feasibility=feas_d, candidates=cands, frontier=frontier, selection=selection,
        selected_name=selected_name, selected=selected, risk_contrib=rc, var_table=var_df,
        sections=sections, warnings=warnings, timings=timings, mc_main=mc_main, mc_profile=mc_profile,
        backtest=bt)

    # ------------------------------------------------------------- independent validation
    with _Timer(timings, "validation"):
        result.validation = validate_result(result)
    result.finished_at_utc = datetime.now(timezone.utc).isoformat()
    return result


def validate_result(res: AnalysisResult) -> iv.ValidationReport:
    cfg = res.config
    ds = res.dataset
    checks: list[iv.Check] = []
    checks += iv.check_data(ds.prices, ds.is_synthetic, ds.quality.overall.value,
                            {s: p.currency for s, p in ds.provenance.items() if s in ds.prices.columns},
                            cfg.base_currency, cfg.min_history)
    checks.append(iv.check_covariance(res.cov.cov))
    a, lo, hi, names = res.constraints.group_matrix()
    groups = [(names[k], [res.constraints.symbols[i] for i in range(res.constraints.n) if a[k, i] > 0],
               float(lo[k]), float(hi[k])) for k in range(len(names))]
    for name, c in res.candidates.items():
        if not c.success or name in ("hrp", "equal_weight_naive"):
            continue
        checks += iv.check_portfolio(name, c.weights, res.cov.cov, res.constraints.lower, res.constraints.upper,
                                     groups, c.volatility if np.isfinite(c.volatility) else None,
                                     res.risk_contrib["rc"] if name == res.selected_name else None)
    for name in ("hrp", "equal_weight_naive"):
        c = res.candidates.get(name)
        if c is not None and c.violations:
            checks.append(iv.Check(f"{name}: limites", iv.Status.WARNING, False,
                                   "carteira de referência viola restrições (não elegível para seleção): "
                                   + "; ".join(c.violations[:3])))
    checks.append(iv.check_var_es(res.var_table))
    if not res.frontier.empty:
        ms = res.candidates["max_sharpe"]
        mv = res.candidates["min_variance"]
        checks.append(iv.check_frontier(res.frontier, ms.sharpe if ms.success else None,
                                        mv.volatility if mv.success else None))
    gbm = GBMModel(np.array([0.08]), np.array([[0.04]]))

    def run_fn(seed: int, n_paths: int) -> np.ndarray:
        return simulate_portfolio(gbm, np.array([1.0]), SimulationSettings(
            n_paths=n_paths, n_steps=63, seed=seed, initial_capital=1.0, rebalance_every=0)).terminal_wealth

    checks.append(iv.check_mc_reproducibility(run_fn))
    tw = run_fn(7, 20000)
    checks.append(iv.check_mc_theory(float(tw.mean()), float(tw.std(ddof=1) / np.sqrt(len(tw))),
                                     float(np.exp(0.08 * 63 / 252)), "GBM E[S_T]=S_0 e^{mu T}"))
    checks.append(iv.check_concentration(res.selected.weights, res.profile.research_limits.max_weight_per_asset))
    rob = res.sections.get("robustness", {})
    if isinstance(rob, dict) and "max_sharpe_sensitivity" in rob:
        checks.append(iv.check_sensitivity(rob["max_sharpe_sensitivity"]["mu_perturbation_summary"]))
    if res.backtest is not None:
        checks += iv.check_backtest(res.sections.get("lookahead_check", {}), res.backtest.bias_warnings)
        checks.append(iv.check_complexity_gain(res.selection))
    checks.append(iv.check_risk_free(cfg.rf_is_fallback, float(cfg.risk_free_rate)))  # type: ignore[arg-type]
    evt = res.sections.get("evt", {})
    if isinstance(evt, dict):
        for a, e in evt.items():
            st = iv.Status.WARNING if (e.refused or e.warnings) else iv.Status.PASS
            msg = e.refusal_reason if e.refused else (
                f"EVT VaR {a:.0%} = {e.var:.4%} (empírico {e.empirical_var:.4%}); " + "; ".join(e.warnings[:2]))
            checks.append(iv.Check(f"EVT/POT {a:.0%}", st, False, msg))
    if not res.feasibility["feasible"]:
        checks.append(iv.Check("viabilidade das restrições", iv.Status.FAIL, True,
                               "restrições do perfil inviáveis: " + "; ".join(res.feasibility["conflicts"])))
    return iv.ValidationReport(checks)


__all__ = ["AnalysisResult", "run_analysis", "validate_result", "DataUnavailableError", "RiskEstimate"]
