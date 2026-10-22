"""Auditable registry of forecasts: what the model said, when, and how it turned out — no retroactive edits.

1. ``seal`` (when the forecast is computed, on the server): stamps ``gerado_em`` (UTC) and, when the server has a key
   (environment variable ``QPI_AUDIT_KEY``), an HMAC-SHA256 ``assinatura`` over the canonical JSON of the record, so
   its numbers and dates cannot be changed or invented without the signature failing.
2. ``register`` (when the investor clicks "Registrar"): checks that signature, stamps ``registrado_em`` (UTC) and
   chains the entry to the previous one: ``hash = sha256(canonical({registro, registrado_em, hash_anterior}))``; with
   the key, ``selo = HMAC(hash)``. Editing, removing or reordering an entry breaks the chain from that point on
   (``verify``). Removing the LAST entries cannot be detected by a chain alone, and without the key someone who
   rewrites the whole registry (recomputing every hash) is not detected either — both are said in the result.
3. Retroactive: an entry registered more than ``RETRO_BDAYS`` business days after its reference date (stale data, or a
   forecast about the past) is flagged and left out of every metric, since part of the outcome was already known.
4. ``evaluate``: an entry stays ``pendente`` until the day after its horizon ends and a close near the end date
   exists. The outcome comes from the CURRENT adjusted series (reference and end dates read from the same series, so
   dividends paid in between count as return, as in the model fitted on adjusted closes); the stored reference price
   is shown, never used for the return. While pending, the invalidation conditions are checked on request (fell below
   the 5% worst scenario; realised volatility above 1.5x the model's) — nothing runs in the background.

Scores over the evaluated entries: Brier score and log score of each probability (gain over always saying 50%),
reliability table, coverage of the 10–90% interval, PIT histogram, the median forecast vs the realised return, and the
same events after round-trip costs; broken down by asset class, horizon and volatility regime.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.forecasting.calibration import _brier, _logscore, reliability
from core.forecasting.data import BENCH, P, _clean, rf_growth, rf_index_from_annual_pct
from core.market_data.cash import fetch_cdi_cash
from core.market_data.yahoo_chart import FetchFn
from core.portfolio_analytics.data import CLASS_PT

GENESIS = "0" * 64
RETRO_BDAYS = 3
KEY_ENV = "QPI_AUDIT_KEY"
VERSION = 1
MAX_ENTRIES = 500
EVENTS = {"p_alta": "subir (retorno > 0)", "p_superar_rf": "superar a renda fixa", "p_queda_20": "cair mais de 20%",
          "p_superar_referencia": "superar o índice de referência"}
REQUIRED = ("ativo", "preco_referencia", "data_referencia", "data_fim", "distribuicao", "probabilidades", "premissas",
            "gerado_em")
_HEX = re.compile(r"[0-9a-f]{64}")


def audit_key() -> bytes | None:
    k = os.environ.get(KEY_ENV, "")
    return k.encode() if k else None


def _canon(x: object) -> object:
    """Numbers as JSON round-trips them through a browser: an integral float becomes an int (JS prints 100.0 as 100)."""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_canon(v) for v in x]
    if isinstance(x, float) and x.is_integer() and abs(x) < 2 ** 53:
        return int(x)
    return x


def canonical(obj: object) -> bytes:
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _mac(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _now(now: datetime | None) -> datetime:
    return (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)


def seal(record: dict[str, object], key: bytes | None, now: datetime | None = None) -> dict[str, object]:
    rec = {k: v for k, v in record.items() if k != "assinatura"}
    rec["versao_registro"] = VERSION
    rec["gerado_em"] = _now(now).isoformat()
    rec["assinatura"] = _mac(key, canonical(rec)) if key else None
    return rec


def _signature_ok(rec: dict[str, object], key: bytes) -> bool:
    sig = rec.get("assinatura")
    body = {k: v for k, v in rec.items() if k != "assinatura"}
    try:
        return isinstance(sig, str) and hmac.compare_digest(sig, _mac(key, canonical(body)))
    except (TypeError, ValueError):
        return False


def _parse_dt(s: object) -> datetime:
    t = datetime.fromisoformat(str(s))
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


def _check_record(rec: object) -> dict[str, object]:
    if not isinstance(rec, dict):
        raise ValueError("registro inválido: esperado um objeto com a previsão")  # noqa: TRY004 - input error -> 400
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise ValueError(f"registro incompleto: faltam {', '.join(missing)}")
    if not all(isinstance(rec[k], dict) for k in ("ativo", "distribuicao", "probabilidades", "premissas")):
        raise ValueError("registro inválido: ativo, distribuição, probabilidades e premissas devem ser objetos")
    try:
        ref, end = date.fromisoformat(str(rec["data_referencia"])), date.fromisoformat(str(rec["data_fim"]))
        _parse_dt(rec["gerado_em"])
        m, s = float(rec["distribuicao"]["m_log"]), float(rec["distribuicao"]["s_log"])  # type: ignore[index]
        tk = str(rec["ativo"]["ticker"])  # type: ignore[index]
        probs = [float(v) for v in rec["probabilidades"].values()]  # type: ignore[union-attr]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"registro inválido: {e}") from e
    if end <= ref or not (np.isfinite(m) and np.isfinite(s) and s > 0) or not tk or len(tk) > 20:
        raise ValueError("registro inválido: datas, distribuição ou código do ativo fora do esperado")
    if not all(0.0 <= p <= 1.0 for p in probs):
        raise ValueError("registro inválido: probabilidades devem estar entre 0 e 1")
    return rec


def _bdays(a: date, b: date) -> int:
    return int(np.busday_count(a, b)) if b > a else 0


def is_retroactive(rec: dict[str, object], registered_at: str) -> bool:
    ref = date.fromisoformat(str(rec["data_referencia"]))
    reg = _parse_dt(registered_at).astimezone(UTC).date()
    return _bdays(ref, reg) > RETRO_BDAYS


def _entry_hash(e: dict[str, object]) -> str:
    return hashlib.sha256(canonical({"registro": e.get("registro"), "registrado_em": e.get("registrado_em"),
                                     "hash_anterior": e.get("hash_anterior")})).hexdigest()


def register(record: dict[str, object], prev_hash: str, key: bytes | None, now: datetime | None = None
             ) -> dict[str, object]:
    rec = _check_record(record)
    if key and not _signature_ok(rec, key):
        raise ValueError("registro alterado ou não gerado por este servidor: a assinatura não confere")
    if prev_hash != GENESIS and not _HEX.fullmatch(str(prev_hash)):
        raise ValueError("hash da entrada anterior inválido")
    t = _now(now)
    if _parse_dt(rec["gerado_em"]) > t + timedelta(minutes=5):
        raise ValueError("registro com data de geração no futuro")
    entry: dict[str, object] = {"registro": rec, "registrado_em": t.isoformat(), "hash_anterior": prev_hash}
    entry["hash"] = _entry_hash(entry)
    entry["selo"] = _mac(key, str(entry["hash"]).encode()) if key else None
    return entry


def verify(entries: list[dict[str, object]], key: bytes | None) -> list[dict[str, object]]:
    """Integrity of each entry, in order. The chain continues from each entry's STORED hash, so one tampered entry
    is pointed at without condemning every entry after it."""
    prev, out = GENESIS, []
    for i, e in enumerate(entries):
        problems = []
        if not isinstance(e, dict):
            out.append({"indice": i, "ok": False, "problemas": ["entrada em formato inválido"]})
            continue
        try:
            h = _entry_hash(e)
        except (TypeError, ValueError):
            h = None
        if h is None or h != e.get("hash"):
            problems.append("conteúdo alterado depois do registro (o hash não confere)")
        if e.get("hash_anterior") != prev:
            problems.append("cadeia quebrada: uma entrada anterior foi removida, alterada ou trocada de ordem")
        if key:
            selo = e.get("selo")
            if not isinstance(selo, str) or not hmac.compare_digest(selo, _mac(key, str(e.get("hash")).encode())):
                problems.append("selo do servidor ausente ou inválido")
            rec = e.get("registro")
            if not isinstance(rec, dict) or not _signature_ok(rec, key):
                problems.append("assinatura da previsão ausente ou inválida")
        try:
            _check_record(e.get("registro"))
            _parse_dt(e.get("registrado_em"))
        except ValueError as err:
            problems.append(str(err))
        prev = str(e.get("hash"))
        out.append({"indice": i, "ok": not problems, "problemas": problems})
    return out


# ----------------------------------------------------------------------------- evaluation
def _on_or_before(s: pd.Series, t: pd.Timestamp, tol_days: int = 7) -> tuple[pd.Timestamp, float] | None:
    x = s.loc[:t]
    if x.empty or (t - x.index[-1]).days > tol_days:
        return None
    return x.index[-1], float(x.iloc[-1])


def _market(entries: list[dict[str, object]], provider: object, fetch: FetchFn | None, as_of: date
            ) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """Prices of every asset and benchmark involved, and the risk-free index per currency, up to ``as_of``."""
    tick, ccys = set(), set()
    for e in entries:
        a = e["registro"]["ativo"]  # type: ignore[index]
        tick.add(str(a["ticker"]))
        ccys.add(str(a.get("moeda", "")))
    syms = sorted(tick | {BENCH[c][0] for c in ccys if c in BENCH} | ({"^IRX"} if "USD" in ccys else set()))
    got = provider.fetch(syms, "10y", "1d")  # type: ignore[attr-defined]
    px = {s: _clean(got[s].close, as_of) for s in syms if s in got and got[s].close is not None}
    px = {s: v for s, v in px.items() if not v.empty}
    rf: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    if "BRL" in ccys:
        try:
            rf["BRL"] = _clean(fetch_cdi_cash(fetch, start=as_of - timedelta(days=3650), end=as_of).close, as_of)
        except Exception as e:  # noqa: BLE001 - reported; the 'superar a renda fixa' event is then not scored
            notes["BRL"] = f"CDI indisponível ({e})"
    if "USD" in ccys:
        if "^IRX" in px:
            rf["USD"] = rf_index_from_annual_pct(px["^IRX"])
        else:
            notes["USD"] = "T-bill (^IRX) indisponível"
    return px, rf, notes


def _invalidation(rec: dict[str, object], s: pd.Series, t_ref: pd.Timestamp) -> list[str]:
    q05 = rec["distribuicao"].get("quantis", {}).get("p05")  # type: ignore[union-attr]
    vol = rec["premissas"].get("volatilidade_log")  # type: ignore[union-attr]
    after = s.loc[t_ref:]
    hits = []
    if len(after) > 1:
        path = after / after.iloc[0] - 1.0
        if q05 is not None and float(path.iloc[1:].min()) < float(q05):
            hits.append(f"caiu abaixo do cenário dos 5% piores ({float(q05):+.1%}) antes do fim do horizonte")
        lr = np.log(after).diff().dropna()
        if vol and len(lr) >= 20 and float(lr.std(ddof=1) * np.sqrt(P)) > 1.5 * float(vol):
            hits.append(f"volatilidade realizada de {float(lr.std(ddof=1) * np.sqrt(P)):.0%} ao ano, acima de 1,5× a do "
                        f"modelo ({float(vol):.0%})")
    return hits


def _horizon_bucket(days: int) -> str:
    return "até 6 meses" if days <= 186 else ("de 6 a 12 meses" if days <= 366 else "mais de 12 meses")


def _event_scores(df: pd.DataFrame, pcol: str, ocol: str) -> dict[str, object] | None:
    sub = df[df[pcol].notna() & df[ocol].notna()]
    if sub.empty:
        return None
    p, o = sub[pcol].to_numpy(float), sub[ocol].astype(float).to_numpy()
    bs = _brier(p, o)
    return {"n": len(sub), "brier": bs, "habilidade_vs_50": 1.0 - bs / 0.25, "log_score": _logscore(p, o),
            "previsto_medio": float(p.mean()), "frequencia_observada": float(o.mean()), "confiabilidade": reliability(p, o)}


def _group(df: pd.DataFrame, col: str) -> dict[str, dict[str, object]]:
    out = {}
    for k, g in df.groupby(col):
        item: dict[str, object] = {"n": len(g), "cobertura_10_90": float(g["dentro_10_90"].mean())}
        for ev in ("alta", "superou_rf"):
            sc = _event_scores(g, "p_" + ("alta" if ev == "alta" else "superar_rf"), ev)
            if sc:
                item[f"brier_{ev}"] = sc["brier"]
        out[str(k)] = item
    return out


def evaluate(entries: list[dict[str, object]], provider: object, fetch: FetchFn | None, as_of: date | None = None,
             cost_bps: float = 0.0, key: bytes | None = None) -> dict[str, object]:
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"no máximo {MAX_ENTRIES} entradas por avaliação")
    if not 0 <= cost_bps <= 500:
        raise ValueError("custo por operação deve estar entre 0 e 500 pontos-base (0% a 5%)")
    as_of = as_of or datetime.now(UTC).date()
    checks = verify(entries, key)
    usable = [e for e, c in zip(entries, checks, strict=True) if c["ok"]]
    px, rf, rf_notes = _market(usable, provider, fetch, as_of) if usable else ({}, {}, {})
    c = cost_bps / 1e4
    rows, table = [], []
    for e, chk in zip(entries, checks, strict=True):
        rec = e.get("registro") if isinstance(e, dict) else None
        a = rec.get("ativo") if isinstance(rec, dict) else None
        a = a if isinstance(a, dict) else {}
        line: dict[str, object] = {"indice": chk["indice"], "ticker": a.get("ticker"), "nome": a.get("nome"),
                                   "data_referencia": rec.get("data_referencia") if isinstance(rec, dict) else None,
                                   "data_fim": rec.get("data_fim") if isinstance(rec, dict) else None,
                                   "registrado_em": e.get("registrado_em") if isinstance(e, dict) else None,
                                   "probabilidades": rec.get("probabilidades") if isinstance(rec, dict) else None}
        table.append(line)
        if not chk["ok"]:
            line.update(estado="inválida", motivo="; ".join(chk["problemas"]))  # type: ignore[arg-type]
            continue
        assert isinstance(rec, dict)
        if is_retroactive(rec, str(e["registrado_em"])):
            line.update(estado="retroativa", motivo=f"registrada mais de {RETRO_BDAYS} dias úteis depois da data de "
                        "referência: parte do resultado já era conhecida; fora das métricas")
            continue
        tk, ccy = str(a["ticker"]), str(a.get("moeda", ""))
        s = px.get(tk)
        t_ref, t_end = pd.Timestamp(str(rec["data_referencia"])), pd.Timestamp(str(rec["data_fim"]))
        start = _on_or_before(s, t_ref) if s is not None else None
        if start is None:
            line.update(estado="sem dados", motivo=f"sem cotação de {tk} na data de referência")
            continue
        if as_of <= t_end.date():
            last = s.index[-1]
            line.update(estado="pendente", motivo=f"o horizonte termina em {t_end:%d/%m/%Y}",
                        retorno_ate_agora=float(s.iloc[-1] / start[1] - 1.0), cotacao_ate=f"{last:%Y-%m-%d}",
                        invalidacao=_invalidation(rec, s, start[0]))
            continue
        end = _on_or_before(s, t_end)
        if end is None:
            line.update(estado="sem dados", motivo=f"sem cotação de {tk} perto de {t_end:%d/%m/%Y}")
            continue
        R = end[1] / start[1] - 1.0
        d = rec["distribuicao"]
        m, sd = float(d["m_log"]), float(d["s_log"])  # type: ignore[index]
        q = d.get("quantis", {})  # type: ignore[union-attr]
        probs = rec["probabilidades"]
        rf_real = rf_growth(rf.get(ccy), t_ref, t_end)
        bt = BENCH.get(ccy, (None,))[0]
        rb = None
        if bt and bt in px and bt != tk:
            b0, b1 = _on_or_before(px[bt], t_ref), _on_or_before(px[bt], t_end)
            rb = b1[1] / b0[1] - 1.0 if b0 and b1 else None
        r_net = (1.0 + R) * (1.0 - c) ** 2 - 1.0
        cls = str(a.get("classe", "nao_classificado"))
        row = {"ticker": tk, "classe": CLASS_PT.get(cls, "não classificado" if cls == "nao_classificado" else cls),
               "regime_vol": {"alta": "volatilidade alta", "baixa": "volatilidade baixa"}.get(str(rec.get("regime_vol")), "não medido"),
               "horizonte": _horizon_bucket(int(rec.get("horizonte_dias") or (t_end - t_ref).days)),
               "realizado": R, "realizado_liquido": r_net, "rf_realizado": rf_real, "referencia_realizado": rb,
               "mediana_prevista": q.get("p50"), "p_alta": probs.get("p_alta"), "alta": R > 0,  # type: ignore[union-attr]
               "p_superar_rf": probs.get("p_superar_rf") if rf_real is not None else None,  # type: ignore[union-attr]
               "superou_rf": (R > rf_real) if rf_real is not None else None,
               "superou_rf_liquido": (r_net > rf_real) if rf_real is not None else None,
               "p_queda_20": probs.get("p_queda_20"), "queda_20": R < -0.20,  # type: ignore[union-attr]
               "p_superar_referencia": probs.get("p_superar_referencia") if rb is not None else None,  # type: ignore[union-attr]
               "superou_referencia": (R > rb) if rb is not None else None,
               "pit": float(norm.cdf((np.log1p(R) - m) / sd)) if sd > 0 else None,
               "dentro_10_90": (q.get("p10") is not None and q.get("p90") is not None  # type: ignore[union-attr]
                                and float(q["p10"]) <= R <= float(q["p90"]))}  # type: ignore[index]
        rows.append(row)
        line.update(estado="avaliada", realizado=R, realizado_liquido=r_net, rf_realizado=rf_real,
                    referencia_realizado=rb, pit=row["pit"], dentro_10_90=row["dentro_10_90"],
                    eventos={"alta": row["alta"], "superou_rf": row["superou_rf"], "queda_20": row["queda_20"],
                             "superou_referencia": row["superou_referencia"]})
    states = pd.Series([t.get("estado") for t in table], dtype="object").value_counts().to_dict() if table else {}
    out: dict[str, object] = {"data_avaliacao": f"{as_of:%Y-%m-%d}", "entradas": table, "contagem": states,
                              "assinado_pelo_servidor": key is not None, "custo_bps": cost_bps,
                              "integridade": checks, "fontes_indisponiveis": rf_notes}
    notes = ["entradas retroativas, inválidas ou pendentes ficam fora das métricas",
             ("apagar as últimas entradas não é detectável só pela cadeia de hashes; registrar só as previsões de que "
              "se gostou também distorce a avaliação")]
    if key is None:
        notes.append("sem a chave do servidor: os hashes detectam edições acidentais, mas não quem reescreve o "
                     "registro inteiro")
    if rows:
        df = pd.DataFrame(rows)
        metrics: dict[str, object] = {"n": len(df)}
        for pcol, ocol in (("p_alta", "alta"), ("p_superar_rf", "superou_rf"), ("p_queda_20", "queda_20"),
                           ("p_superar_referencia", "superou_referencia")):
            metrics[ocol] = _event_scores(df, pcol, ocol)
        metrics["superou_rf_apos_custos"] = _event_scores(df, "p_superar_rf", "superou_rf_liquido")
        metrics["cobertura_10_90"] = float(df["dentro_10_90"].mean())
        pits = df["pit"].dropna().to_numpy(float)
        metrics["pit_histograma"] = np.histogram(pits, bins=10, range=(0.0, 1.0))[0].tolist() if len(pits) else []
        metrics["mediana_prevista_media"] = float(df["mediana_prevista"].dropna().mean()) if df["mediana_prevista"].notna().any() else None
        metrics["realizado_medio"] = float(df["realizado"].mean())
        metrics["realizado_liquido_medio"] = float(df["realizado_liquido"].mean())
        metrics["por_classe"] = _group(df, "classe")
        metrics["por_horizonte"] = _group(df, "horizonte")
        metrics["por_regime"] = _group(df, "regime_vol")
        out["metricas"] = metrics
        if len(df) < 30:
            notes.append(f"só {len(df)} previsões avaliadas: com menos de 30, as métricas oscilam muito de uma para outra")
    out["notas"] = notes
    out["eventos"] = EVENTS
    return out
