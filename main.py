"""Quant Portfolio Intelligence Engine — command-line entry point (orchestration only).

Examples
--------
    python main.py --offline
    python main.py --offline --profile arrojado --simulations 5000 --fast
    python main.py --profile moderado --period 5y --risk-free-rate 0.105

Exit codes: 0 = completed and validated (PASS/WARNING); 2 = data unavailable or invalid
input; 3 = completed but NOT validated (critical FAIL in the independent validator).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from config import APP_NAME, APP_VERSION, AppConfig
from core.market_data.base import DataUnavailableError
from core.profiler import demo_profile
from models.investor import InvestorProfileInput, RiskProfile
from utils.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=f"{APP_NAME} v{APP_VERSION} — pesquisa e simulação "
                                "(não executa ordens, não é recomendação de investimento).")
    p.add_argument("--offline", action="store_true", default=None,
                   help="dados SINTÉTICOS determinísticos, sem internet (também via QPI_OFFLINE=true)")
    p.add_argument("--profile", choices=[x.value for x in RiskProfile], default=None,
                   help="perfil do investidor hipotético de demonstração (padrão: moderado)")
    p.add_argument("--profile-file", type=Path, help="JSON com InvestorProfileInput (substitui --profile)")
    p.add_argument("--tickers", nargs="+", help="símbolos (modo online)")
    p.add_argument("--benchmark", help="símbolo do benchmark (modo online)")
    p.add_argument("--period", help="histórico do provedor, ex. 5y, 3y, 10y (padrão 5y)")
    p.add_argument("--interval", choices=["1d"],
                   help="frequência dos dados; somente diária (1d) nesta versão — todas as anualizações usam 252")
    p.add_argument("--base-currency", help="moeda-base ISO-4217 (padrão BRL)")
    p.add_argument("--risk-free-rate", type=float, help="taxa livre de risco anual decimal (ex. 0.105)")
    p.add_argument("--simulations", type=int, help="número de trajetórias Monte Carlo (padrão 20000)")
    p.add_argument("--horizon-days", type=int, help="horizonte da simulação principal em dias úteis (padrão 252)")
    p.add_argument("--seed", type=int, help="semente (padrão 42)")
    p.add_argument("--sim-model", choices=["gbm", "student_t", "bootstrap", "block_bootstrap", "regime", "garch"])
    p.add_argument("--cov-method", choices=["auto", "sample", "ledoit_wolf", "lw_constant_corr", "oas", "ewma", "factor_pca"],
                   help="estimador de covariância (padrão auto = escolhido fora da amostra)")
    p.add_argument("--mu-method", choices=["historical", "ewma", "james_stein"])
    p.add_argument("--var-method", choices=["historical", "normal", "student_t", "cornish_fisher", "simulated"])
    p.add_argument("--max-weight", type=float, help="peso máximo por ativo (0-1), além do limite do perfil")
    p.add_argument("--cost-bps", type=float, help="corretagem/custo de transação em bps")
    p.add_argument("--spread-bps", type=float, help="meio-spread em bps")
    p.add_argument("--inflation", type=float, help="inflação anual HIPOTÉTICA para valores reais (padrão 0.045)")
    p.add_argument("--goal", type=float, help="objetivo nominal de patrimônio final (opcional)")
    p.add_argument("--capital", type=float, help="capital hipotético (substitui o do perfil de demonstração)")
    p.add_argument("--monthly-contribution", type=float, help="aporte mensal hipotético")
    p.add_argument("--horizon-years", type=float, help="horizonte do investidor em anos")
    p.add_argument("--resamples", type=int, help="reamostragens para robustez (padrão 50; 0 desativa)")
    p.add_argument("--fast", action="store_true",
                   help="pula análises pesadas (backtest, modelos de volatilidade, regimes, robustez)")
    p.add_argument("--no-charts", action="store_true", help="não gera PNGs")
    p.add_argument("--output-dir", type=Path, help="diretório de saída (padrão outputs/)")
    p.add_argument("--ask", action="append", default=[],
                   help="pergunta ao analista de IA (baseado em ferramentas); pode repetir")
    p.add_argument("--external-file", type=Path, action="append", default=[],
                   help="documento externo (texto) tratado como dado NÃO confiável pelo analista")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def config_from_args(a: argparse.Namespace) -> AppConfig:
    overrides: dict[str, object] = {}   # only explicit CLI flags override QPI_* environment settings
    if a.offline is not None:
        overrides["offline"] = a.offline
    if a.profile is not None:
        overrides["profile"] = a.profile
    mapping = {"tickers": "tickers", "benchmark": "benchmark", "period": "period", "interval": "interval",
               "base_currency": "base_currency", "risk_free_rate": "risk_free_rate",
               "simulations": "n_simulations", "horizon_days": "horizon_days", "seed": "seed",
               "sim_model": "simulation_model", "cov_method": "covariance_method",
               "mu_method": "expected_return_method", "var_method": "var_method", "max_weight": "max_weight",
               "cost_bps": "transaction_cost_bps", "spread_bps": "spread_bps", "inflation": "inflation_annual",
               "goal": "goal", "resamples": "n_resamples", "output_dir": "output_dir"}
    for arg, key in mapping.items():
        v = getattr(a, arg)
        if v is not None:
            overrides[key] = v
    if a.fast:
        overrides["run_heavy_analyses"] = False
    if a.no_charts:
        overrides["make_charts"] = False
    return AppConfig(**overrides)


def profile_from_args(a: argparse.Namespace, cfg: AppConfig) -> InvestorProfileInput:
    if a.profile_file:
        data = json.loads(a.profile_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("--profile-file deve conter um objeto JSON (InvestorProfileInput)")
        p = InvestorProfileInput(**data)
        if p.base_currency != cfg.base_currency:
            raise ValueError(f"moeda do perfil ({p.base_currency}) difere da moeda-base ({cfg.base_currency})")
        upd: dict[str, object] = {}
    else:
        p = demo_profile(RiskProfile(cfg.profile))
        upd = {"base_currency": cfg.base_currency}
    if a.capital is not None:
        upd["capital"] = a.capital
    if a.monthly_contribution is not None:
        upd["monthly_contribution"] = a.monthly_contribution
    if a.horizon_years is not None:
        upd["horizon_years"] = a.horizon_years
    return InvestorProfileInput(**{**p.model_dump(), **upd})


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
        cfg.ensure_dirs()
        prof = profile_from_args(args, cfg)
        external_docs = [(f.read_text(encoding="utf-8"), f"arquivo local {f.name}") for f in args.external_file]
    except (PydanticValidationError, ValueError, TypeError, OSError, UnicodeDecodeError) as e:
        print(f"ERRO de configuração/entrada: {e}", file=sys.stderr)
        return 2
    logger = configure_logging(cfg.logs_dir / "run.jsonl", getattr(logging, args.log_level))
    logger.info("run start", extra={"offline": cfg.offline, "profile": prof.investor_id})

    from core.portfolio_engine import run_analysis  # heavy imports after argument validation
    from core.reporter import write_outputs

    try:
        res = run_analysis(cfg, prof)
    except DataUnavailableError as e:
        print(f"ERRO: dados indisponíveis — {e}\nNenhum dado sintético foi usado em substituição. "
              "Para demonstração sem internet use --offline.", file=sys.stderr)
        logger.error("data unavailable", extra={"error": str(e)})
        return 2
    outputs = write_outputs(res)
    if args.ask:
        from core.ai_analyst.analyst import ResearchAnalyst
        from core.ai_analyst.tools import AnalysisContext

        analyst = ResearchAnalyst(AnalysisContext.from_result(res))
        ext = external_docs
        parts = [f"# Analista de IA — execução {res.run_id}\n",
                 "Respostas geradas por regras determinísticas a partir de ferramentas validadas; "
                 "nenhum modelo de linguagem externo foi chamado.\n"]
        for q in args.ask:
            ans = analyst.ask(q, ext)
            parts.append(ans.render())
            print("\n" + ans.render())
        ap = cfg.reports_dir / f"{res.run_id}_analyst.md"
        ap.write_text("\n\n".join(parts), encoding="utf-8")
        outputs["analyst"] = str(ap)
    v = res.validation
    print(f"{APP_NAME} v{APP_VERSION} — execução {res.run_id}")
    print(f"Dados: {'SINTÉTICOS (offline)' if res.dataset.is_synthetic else 'reais (yfinance)'}; "
          f"{len(res.dataset.prices)} datas, {len(res.dataset.symbols)} ativos")
    print(f"Perfil final: {res.profile.final_profile.value}; carteira selecionada: {res.selected_name}")
    print(f"Retorno esperado [E] {res.selected.expected_return:.2%}; volatilidade [E] {res.selected.volatility:.2%}")
    if res.mc_main is not None:
        sm = res.mc_main.summary
        print(f"Monte Carlo [S] {sm['n_paths']} trajetórias: mediana do patrimônio final "
              f"R$ {sm['terminal_wealth_percentiles']['p50']:,.0f}; P(retorno<0) {sm['p_negative_return']:.1%}")
    print(f"Validação independente: {v.overall.value if v else 'N/D'}")
    for w in res.warnings:
        print(f"  ! {w}")
    print(f"Relatório: {outputs['report']}")
    print(f"Manifesto: {outputs['manifest']}")
    if "analyst" in outputs:
        print(f"Analista: {outputs['analyst']}")
    logger.info("run end", extra={"run_id": res.run_id, "validation": v.overall.value if v else None,
                                  "timings": res.timings})
    return 0 if (v is None or v.validated) else 3


if __name__ == "__main__":
    sys.exit(main())
