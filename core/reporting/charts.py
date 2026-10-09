"""Static PNG charts (matplotlib, Agg backend).

Every chart has a title, axis labels with units, a legend when there are two or
more series, a source line, and a visible "DADOS SINTÉTICOS" banner when the
data are synthetic. Colours follow a fixed categorical order (validated
colour-blind-safe reference palette); magnitude uses one blue ramp and polarity
a blue<->red diverging ramp with a neutral grey midpoint.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a7f", "#b8302f"])
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _finish(fig: plt.Figure, path: Path, source: str, synthetic: bool) -> Path:
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.01, 0.005, f"Fonte: {source}", fontsize=7, color=INK2, ha="left", va="bottom")
    if synthetic:
        fig.text(0.99, 0.995, "DADOS SINTÉTICOS — não representam ativos reais", fontsize=8,
                 color="#b8302f", ha="right", va="top", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


_pct = PercentFormatter(1.0, decimals=0)
_money = FuncFormatter(lambda x, _: f"{x / 1000:,.0f} mil".replace(",", "."))


def efficient_frontier_chart(frontier: pd.DataFrame, points: dict[str, tuple[float, float]],
                             assets: pd.DataFrame, path: Path, source: str, synthetic: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    _style(ax)
    if not frontier.empty:
        ax.plot(frontier["volatility"], frontier["expected_return"], color=SERIES[0], lw=2,
                label="Fronteira eficiente (somente pontos viáveis)")
    ax.scatter(assets["volatility"], assets["expected_return"], s=28, color="#8a8984",
               label="Ativos individuais", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for s, row in assets.iterrows():
        ax.annotate(str(s), (row["volatility"], row["expected_return"]), fontsize=6.5, color=INK2,
                    xytext=(3, 3), textcoords="offset points")
    markers = ["o", "D", "s", "^", "v", "P", "X"]
    for i, (name, (vol, ret)) in enumerate(points.items()):
        ax.scatter([vol], [ret], s=70, color=SERIES[(i + 1) % len(SERIES)], marker=markers[i % 7],
                   label=name, zorder=4, edgecolor=SURFACE, linewidth=2)
    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_xlabel("Volatilidade anualizada estimada (%)", color=INK)
    ax.set_ylabel("Retorno esperado anualizado estimado (%)", color=INK)
    ax.set_title("Fronteira eficiente (estimativas, não previsões)", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="best")
    return _finish(fig, path, source, synthetic)


def mc_fan_chart(pct: pd.DataFrame, rep_paths: np.ndarray, time_index: np.ndarray,
                 path: Path, source: str, synthetic: bool, periods_per_year: int = 252) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    _style(ax)
    x = time_index / periods_per_year
    for p in rep_paths[:: max(1, len(rep_paths) // 12)]:
        ax.plot(x, p, color="#b9b8b2", lw=0.6, alpha=0.8)
    ax.fill_between(x, pct["p5"], pct["p95"], color=BLUE_RAMP[1], alpha=0.6, label="Percentis 5–95")
    ax.fill_between(x, pct["p25"], pct["p75"], color=BLUE_RAMP[3], alpha=0.6, label="Percentis 25–75")
    ax.plot(x, pct["p50"], color=BLUE_RAMP[6], lw=2, label="Mediana")
    ax.plot([], [], color="#b9b8b2", lw=0.8, label="Trajetórias representativas")
    ax.yaxis.set_major_formatter(_money)
    ax.set_xlabel("Horizonte (anos)", color=INK)
    ax.set_ylabel("Patrimônio nominal (R$)", color=INK)
    ax.set_title("Monte Carlo: trajetórias e percentis do patrimônio", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return _finish(fig, path, source, synthetic)


def mc_terminal_hist(terminal: np.ndarray, initial: float, invested: float, path: Path,
                     source: str, synthetic: bool) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    _style(ax)
    lo, hi = np.percentile(terminal, [0.5, 99.5])
    ax.hist(np.clip(terminal, lo, hi), bins=80, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ymax = ax.get_ylim()[1]
    for val, lab, ls in ((np.percentile(terminal, 5), "P5", ":"), (np.median(terminal), "Mediana", "-"),
                         (np.percentile(terminal, 95), "P95", ":"), (initial, "Capital inicial", "--"),
                         (invested, "Total líquido aportado", "-.")):
        ax.axvline(val, color=INK2, lw=1, ls=ls)
        ax.text(val, ymax * 0.97, f" {lab}", fontsize=7, color=INK, rotation=90, va="top")
    ax.xaxis.set_major_formatter(_money)
    ax.set_xlabel("Patrimônio final nominal (R$) — caudas 0,5% recortadas na visualização", color=INK)
    ax.set_ylabel("Número de trajetórias", color=INK)
    ax.set_title("Distribuição do patrimônio final (Monte Carlo)", color=INK, fontsize=11, loc="left")
    return _finish(fig, path, source, synthetic)


def correlation_heatmap(corr: pd.DataFrame, path: Path, source: str, synthetic: bool) -> Path:
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * n, 0.8 + 0.55 * n))
    im = ax.imshow(corr.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=60, ha="right", fontsize=7, color=INK2)
    ax.set_yticks(range(n), corr.index, fontsize=7, color=INK2)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="#ffffff" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("Correlação de Pearson (retornos diários)", fontsize=7, color=INK2)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("Matriz de correlação", color=INK, fontsize=11, loc="left")
    return _finish(fig, path, source, synthetic)


def weights_chart(weights: pd.DataFrame, by_class: pd.DataFrame, path: Path, source: str,
                  synthetic: bool) -> Path:
    """Grouped horizontal bars: weights per asset and per class for candidate portfolios."""
    cols = list(weights.columns)[:8]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 + 0.42 * max(len(weights), len(by_class)) * len(cols) / 2.2),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    for ax, df, title in ((axes[0], weights, "Peso por ativo"), (axes[1], by_class, "Peso por classe")):
        _style(ax)
        y = np.arange(len(df))
        h = 0.8 / len(cols)
        for k, c in enumerate(cols):
            ax.barh(y + k * h, df[c].to_numpy(), height=h * 0.9, color=SERIES[k], label=c)
        ax.set_yticks(y + 0.4 - h / 2, df.index, fontsize=7)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(_pct)
        ax.set_xlabel("Peso na carteira (%)", color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    return _finish(fig, path, source, synthetic)


def stress_chart(table: pd.DataFrame, path: Path, source: str, synthetic: bool) -> Path:
    """Horizontal bars of scenario portfolio returns (losses red, gains blue)."""
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.38 * len(table)))
    _style(ax)
    vals = table["portfolio_return"].to_numpy()
    colors = np.where(vals < 0, "#b8302f", "#256abf")
    y = np.arange(len(table))
    ax.barh(y, vals, color=colors, height=0.7)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.1%} ", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    ax.set_yticks(y, table.index, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.xaxis.set_major_formatter(_pct)
    ax.set_xlabel("Retorno da carteira no cenário (%) — cenários são hipóteses, não previsões", color=INK)
    ax.set_title("Cenários de estresse (carteira selecionada)", color=INK, fontsize=11, loc="left")
    return _finish(fig, path, source, synthetic)


def backtest_chart(equity: pd.DataFrame, drawdowns: pd.DataFrame, path: Path, source: str,
                   synthetic: bool) -> Path:
    cols = list(equity.columns)[:8]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for ax in (a1, a2):
        _style(ax)
    for k, c in enumerate(cols):
        a1.plot(equity.index, equity[c], color=SERIES[k], lw=1.4, label=c)
        a2.plot(drawdowns.index, drawdowns[c], color=SERIES[k], lw=1.0)
    a1.set_ylabel("Valor da cota (início = 1)", color=INK)
    a1.set_title("Backtest walk-forward (líquido de custos)", color=INK, fontsize=11, loc="left")
    a1.legend(fontsize=7, frameon=False, ncol=2)
    a2.yaxis.set_major_formatter(_pct)
    a2.set_ylabel("Drawdown (%)", color=INK)
    a2.set_xlabel("Data", color=INK)
    return _finish(fig, path, source, synthetic)
