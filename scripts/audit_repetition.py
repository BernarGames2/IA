"""Audit: does the portfolio react to profile, horizon, capital, goal and market data — and only to them?

Runs the real-data path of the purpose engine with test doubles (exchange listings, a structured market
in which asset families differ in drift and volatility, Banco Central series), changing ONE input at a
time from a base case, and prints whether the chosen strategy, the candidates and the portfolio changed.
No network is used. Usage:

    python scripts/audit_repetition.py [output.json]
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.purpose.engine import PurposeRequest, run_purpose
from tests.market_doubles import Prov, http, structured_market

BASE = {"capital_inicial": 100_000, "aporte_periodico": 1_000, "valor_alvo": 500_000, "horizonte_anos": 10,
        "perfil_risco": "moderado", "offline": False, "universo": "completo", "max_ativos": 30, "n_simulacoes": 500}
VARIATIONS: list[tuple[str, dict[str, object], int]] = [
    ("base (moderado, 10 anos, R$ 100 mil, alvo R$ 500 mil)", {}, 0),
    ("perfil conservador", {"perfil_risco": "conservador"}, 0),
    ("perfil arrojado", {"perfil_risco": "arrojado"}, 0),
    ("horizonte 30 anos", {"horizonte_anos": 30}, 0),
    ("horizonte 6 anos", {"horizonte_anos": 6}, 0),
    ("horizonte 2 anos", {"horizonte_anos": 2}, 0),
    ("capital R$ 1.000", {"capital_inicial": 1_000}, 0),
    ("capital R$ 5 milhões", {"capital_inicial": 5_000_000}, 0),
    ("alvo R$ 5 milhões", {"valor_alvo": 5_000_000}, 0),
    ("objetivo renda", {"objetivo_tipo": "renda"}, 0),
    ("só Brasil", {"mercados": ["BR"]}, 0),
    ("outro conjunto de preços de mercado", {}, 1),
    ("mais um conjunto de preços", {}, 2),
]


def run(label: str, changes: dict[str, object], market_seed: int) -> dict[str, object]:
    tmp = Path(tempfile.mkdtemp())
    r = run_purpose(PurposeRequest(**{**BASE, **changes}), output_dir=tmp,
                    provider=Prov(structured_market(seed_offset=market_seed)), cache_dir=tmp, rf_fetch=http, fast=True)
    w = r.combined.final[r.combined.final > 1e-4].sort_values(ascending=False)
    return {"variacao": label, "estrategia": r.analysis.selected_name,
            "candidatos": list(r.universe["candidatos"]), "pesos": {k: round(float(v), 4) for k, v in w.items()},
            "impressao_digital": r.reproducibility["impressao_digital_carteira"], "n_posicoes": len(w),
            "volatilidade": round(float(r.combined.vol_final), 4), "perfil_final": r.policy["perfil_final"]}


def main(out: str | None = None) -> int:
    warnings.filterwarnings("ignore")
    rows = [run(*v) for v in VARIATIONS]
    base = rows[0]
    print("| Variação | Estratégia | Candidatos iguais à base? | Carteira igual à base? | Posições | Volatilidade |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        same_c = "sim" if r["candidatos"] == base["candidatos"] else "não"
        same_w = "sim" if r["impressao_digital"] == base["impressao_digital"] else "não"
        print(f"| {r['variacao']} | {r['estrategia']} | {same_c} | {same_w} | {r['n_posicoes']} | "
              f"{r['volatilidade']:.1%} |")
    if out:
        Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
