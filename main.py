"""Quant Portfolio Intelligence Engine — command-line entry point (orchestration only).

Examples
--------
    python main.py --offline
    python main.py --offline --profile arrojado --simulations 5000 --fast
    python main.py --profile moderado --period 5y --risk-free-rate 0.105
    python main.py --offline --proposito --objetivo 900000 --aporte 2000 --horizon-years 10 \\
        --motivo "Quero investir em saúde e energia limpa"
    python main.py --acao PETR4 --horizon-years 3 --capital 10000

Exit codes: 0 = completed and validated (PASS/WARNING); 2 = data unavailable or invalid
input; 3 = completed but NOT validated (critical FAIL in the independent validator).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError as PydanticValidationError

from config import APP_NAME, APP_VERSION, AppConfig
from core.market_data.base import DataUnavailableError
from core.profiler import demo_profile
from models.investor import InvestorProfileInput, RiskProfile
from utils.logging_config import configure_logging
from utils.validation import parse_amount

if TYPE_CHECKING:
    from core.purpose.engine import PurposeRequest


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
    g = p.add_argument_group("modo propósito (carteira alinhada ao motivo do investidor)")
    g.add_argument("--proposito", action="store_true",
                   help="roda o motor de seleção alinhado ao propósito (etapas 1-9) em vez da análise padrão")
    g.add_argument("--objetivo", type=float, help="valor-alvo nominal na moeda do objetivo (obrigatório no modo propósito)")
    g.add_argument("--aporte", type=float, help="aporte periódico (padrão 0)")
    g.add_argument("--periodicidade", choices=["mensal", "trimestral", "semestral", "anual", "nenhum"],
                   help="periodicidade do aporte (padrão mensal)")
    g.add_argument("--motivo", help="motivo do investidor em texto livre (vazio = só etapa quantitativa)")
    g.add_argument("--motivo-file", type=Path, help="arquivo de texto com o motivo do investidor")
    g.add_argument("--peso-alinhamento", type=float, help="peso do alinhamento no slider, 0-1 (padrão 0.30)")
    g.add_argument("--incluir-sob-revisao", action="store_true", help="inclui ativos com governança sob revisão")
    g.add_argument("--governanca-file", type=Path,
                   help="JSON com restrições de governança por ticker ({\"flags\": {TICKER: {status, reason, source}}})")
    g.add_argument("--rf", action="append", default=[], metavar="MOEDA=TAXA",
                   help="taxa livre de risco por moeda, ex. --rf BRL=0.105 --rf USD=0.043 (repita por moeda)")
    g.add_argument("--classes", nargs="+", help="macroclasses do catálogo a considerar (padrão: todas)")
    g.add_argument("--objetivo-tipo", choices=["crescimento", "renda", "preservacao"],
                   help="objetivo principal (não informado = hipótese de crescimento, registrada no relatório)")
    g.add_argument("--queda-maxima", type=float, help="maior queda aceitável, fração (ex. 0.15)")
    g.add_argument("--liquidez-12m", type=float, help="fração do dinheiro que pode precisar em 12 meses (0-1)")
    g.add_argument("--mercados", nargs="+", choices=["BR", "US"], help="mercados aceitos (padrão: Brasil e EUA)")
    g.add_argument("--permitir-cripto", action="store_true", help="permite criptoativos (ETFs listados)")
    g.add_argument("--excluir", nargs="+", default=[], metavar="TICKER", help="ativos que você não quer")
    g.add_argument("--excluir-setores", nargs="+", default=[], metavar="SETOR", help="setores que você não quer")
    g.add_argument("--carteira-atual", action="append", default=[], metavar="TICKER=VALOR",
                   help="o que você já tem (repetível), valor na moeda do objetivo")
    g.add_argument("--fracionario-eua", action="store_true", help="a corretora compra frações de ações dos EUA")
    g.add_argument("--compra-minima", type=float, help="menor compra por ativo que vale a pena (padrão 100)")
    a = p.add_argument_group("vale a pena? (análise de uma ação)")
    a.add_argument("--acao", metavar="CODIGO_OU_NOME",
                   help="analisa uma ação (ex.: PETR4, VALE3, AAPL, 'Petrobras'): veredito por regras + projeção; "
                        "usa --horizon-years (padrão 3), --capital como valor investido (padrão 10000), --profile")
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


def purpose_request_from_args(a: argparse.Namespace, cfg: AppConfig) -> PurposeRequest:
    from core.purpose.engine import PurposeRequest

    if a.objetivo is None:
        raise ValueError("--objetivo (valor-alvo) é obrigatório no modo --proposito")
    rf: dict[str, float] = {}
    for item in a.rf:
        ccy, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"--rf espera MOEDA=TAXA (ex. BRL=0.105), recebido {item!r}")
        rf[ccy.strip().upper()] = float(val)
    if a.risk_free_rate is not None:
        rf.setdefault(cfg.base_currency, float(a.risk_free_rate))
    motivo = a.motivo or ""
    if a.motivo_file:
        motivo = a.motivo_file.read_text(encoding="utf-8")
    flags: dict[str, object] = {}
    if a.governanca_file:
        raw = json.loads(a.governanca_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("--governanca-file deve conter um objeto JSON")
        flags = raw.get("flags", raw)
    prof = demo_profile(RiskProfile(cfg.profile))
    data: dict[str, object] = {
        "capital_inicial": a.capital if a.capital is not None else prof.capital,
        "aporte_periodico": a.aporte if a.aporte is not None else 0.0,
        "valor_alvo": a.objetivo, "moeda_objetivo": cfg.base_currency,
        "horizonte_anos": a.horizon_years if a.horizon_years is not None else prof.horizon_years,
        "perfil_risco": cfg.profile, "taxa_livre_risco": rf, "incluir_sob_revisao": a.incluir_sob_revisao,
        "restricoes_governanca": flags,
        "motivo_investidor": motivo, "seed": cfg.seed, "offline": cfg.offline, "classes": a.classes,
    }
    if a.periodicidade is not None:
        data["periodicidade"] = a.periodicidade
    elif not data["aporte_periodico"]:
        data["periodicidade"] = "nenhum"
    if a.peso_alinhamento is not None:
        data["peso_alinhamento"] = a.peso_alinhamento
    if a.simulations is not None:
        data["n_simulacoes"] = a.simulations
    held: dict[str, float] = {}
    for item in a.carteira_atual:
        tk, sep, val = item.partition("=")
        if not sep or not tk.strip():
            raise ValueError(f"--carteira-atual espera TICKER=VALOR (ex. PETR4=1.500,00), recebido {item!r}")
        held[tk.strip()] = held.get(tk.strip(), 0.0) + parse_amount(val)
    data.update({"objetivo_tipo": a.objetivo_tipo, "tolerancia_perda": a.queda_maxima,
                 "necessidade_liquidez": a.liquidez_12m, "mercados": a.mercados, "permitir_cripto": a.permitir_cripto,
                 "ativos_excluidos": a.excluir, "setores_excluidos": a.excluir_setores, "carteira_atual": held,
                 "fracionario_eua": a.fracionario_eua})
    if a.compra_minima is not None:
        data["valor_minimo_posicao"] = a.compra_minima
    return PurposeRequest(**data)


def run_purpose_cli(args: argparse.Namespace) -> int:
    try:
        cfg = config_from_args(args)
        req = purpose_request_from_args(args, cfg)
    except PydanticValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, x.get('loc', ()))) or 'entrada'}: "
                        f"{str(x.get('msg', '')).removeprefix('Value error, ')}" for x in e.errors())
        print(f"ERRO de configuração/entrada: {msg}", file=sys.stderr)
        return 2
    except (ValueError, TypeError, OSError, UnicodeDecodeError) as e:
        print(f"ERRO de configuração/entrada: {e}", file=sys.stderr)
        return 2
    import pandas as pd

    from core.purpose.engine import run_purpose
    from core.purpose.report import write_purpose_outputs

    out_dir = cfg.output_dir / "proposito"
    try:
        res = run_purpose(req, output_dir=out_dir)
    except DataUnavailableError as e:
        print(f"ERRO: dados indisponíveis — {e}\nNenhum dado sintético foi usado em substituição. "
              "Para demonstração sem internet use --offline.", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 2
    files = write_purpose_outputs(res, out_dir)
    e = res.expected
    print(f"{APP_NAME} v{APP_VERSION} — modo propósito, execução {res.analysis.run_id}")
    if res.is_synthetic:
        print("DEMONSTRAÇÃO COM DADOS SINTÉTICOS — NÃO É UMA CARTEIRA FINAL")
    rate = res.required.rate
    exp = e["cagr_esperado_mediano_simulado"]
    print(f"CAGR necessário {'indisponível' if rate is None else f'{rate:.2%}'} | CAGR esperado (mediana simulada) "
          f"{'indisponível' if exp is None else f'{exp:.2%}'}")
    print(f"P(atingir o objetivo) segundo a simulação: {res.goal['p_objetivo']:.1%} "
          f"(erro padrão {res.goal['erro_padrao']:.2%})")
    for t, row in res.table[res.table["peso"] > 0].iterrows():
        sc = row["score_alinhamento"]
        print(f"  {t:<14} {row['classe']:<14} {row['peso']:6.1%}  alinhamento "
              f"{'n/c' if pd.isna(sc) else f'{sc:.2f}'}  {row['moeda']}")
    for a in res.alerts:
        if a.severity != "INFO":
            print(f"  ! [{a.severity}] {a.message}")
    print(f"Relatório: {files['report']}")
    print(f"JSON: {files['json']}")
    critical = [a for a in res.alerts if a.severity == "FAIL" and a.category in ("validacao", "linguagem")]
    return 3 if critical else 0


def run_stock_cli(a: argparse.Namespace) -> int:
    from core.stock_check.analyzer import analyze_stock, render_markdown, to_dict

    offline = bool(a.offline) if a.offline is not None else False
    try:
        res = analyze_stock(a.acao, a.horizon_years or 3.0, a.capital or 10_000.0, a.profile or "moderado",
                            offline=offline, rf=a.risk_free_rate, n_paths=a.simulations or 5_000,
                            seed=a.seed if a.seed is not None else 42)
    except DataUnavailableError as e:
        print(f"ERRO: dados indisponíveis — {e}\nNada foi estimado no lugar.", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 2
    out = (a.output_dir or Path("outputs")) / "acao"
    out.mkdir(parents=True, exist_ok=True)
    stem = res.resolved.ticker.replace(".", "_").replace("^", "")
    (out / f"{stem}.md").write_text(render_markdown(res), encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(to_dict(res), ensure_ascii=False, indent=2), encoding="utf-8")
    v, pj = res.verdict, res.projection
    print(f"{res.resolved.name} ({res.resolved.ticker}) — veredito do modelo: {v['rotulo']}")
    print(v["manchete"])
    print(f"Perfil {res.profile_fit['perfil']}: {res.profile_fit['leitura']}.")
    for c in pj["cenarios"]:
        print(f"  {c['cenario']:<18} {res.currency} {c['valor_final']:>12,.0f}  ({c['retorno_anual']:+.1%} a.a.)")
    for al in res.alerts:
        print(f"  ! {al}")
    print(f"Relatório: {out / (stem + '.md')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.acao:
        return run_stock_cli(args)
    if args.proposito:
        return run_purpose_cli(args)
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
