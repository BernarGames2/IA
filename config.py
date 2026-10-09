"""Centralised, validated configuration for the Quant Portfolio Intelligence Engine.

All tunable parameters live here. Values can come from (in increasing priority):
defaults below -> environment variables prefixed with ``QPI_`` (or a ``.env`` file)
-> explicit CLI arguments handled in ``main.py``.

Design decisions (documented in docs/assumptions_and_limitations.md):
* The risk-free rate is NEVER silently assumed to be zero. If the user does not
  provide one, ``risk_free_rate_source`` is set to ``"demo_fallback"`` and the
  value is flagged in every report as a demonstration fallback.
* Short selling and leverage are disabled by default.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_NAME = "Quant Portfolio Intelligence Engine"
APP_VERSION = "0.6.0"

# Demonstration fallbacks per currency. These are *not* market quotes: they are
# round, plausible orders of magnitude used only so that the offline demo can run.
# Every report marks them as DEMO FALLBACK when used.
DEMO_RISK_FREE_FALLBACK: dict[str, float] = {"BRL": 0.10, "USD": 0.04, "EUR": 0.025}


class ReturnMethod(str, Enum):
    SIMPLE = "simple"
    LOG = "log"


class CovarianceMethod(str, Enum):
    SAMPLE = "sample"
    AUTO = "auto"                      # chosen by out-of-sample evaluation (1-SE rule)
    LEDOIT_WOLF = "ledoit_wolf"
    LW_CONSTANT_CORR = "lw_constant_corr"
    OAS = "oas"
    EWMA = "ewma"
    FACTOR_PCA = "factor_pca"


class ExpectedReturnMethod(str, Enum):
    HISTORICAL = "historical"
    EWMA = "ewma"
    JAMES_STEIN = "james_stein"


class SimulationModel(str, Enum):
    GBM = "gbm"
    STUDENT_T = "student_t"
    BOOTSTRAP = "bootstrap"
    BLOCK_BOOTSTRAP = "block_bootstrap"
    REGIME = "regime"
    GARCH = "garch"


class VaRMethod(str, Enum):
    HISTORICAL = "historical"
    NORMAL = "normal"
    STUDENT_T = "student_t"
    CORNISH_FISHER = "cornish_fisher"
    SIMULATED = "simulated"


class AppConfig(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="QPI_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- data -----------------------------------------------------------------
    tickers: list[str] = Field(
        default_factory=lambda: [
            "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BOVA11.SA", "IVVB11.SA", "KNRI11.SA",
        ]
    )
    benchmark: str | None = "BOVA11.SA"
    period: str = "5y"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    base_currency: str = "BRL"
    offline: bool = False
    cache_dir: Path = Path(".cache/market_data")
    cache_ttl_hours: float = Field(24.0, gt=0)
    request_timeout_s: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0, le=10)
    min_seconds_between_requests: float = Field(1.0, ge=0)
    min_history: int = Field(252, ge=30)
    max_missing_fraction: float = Field(0.10, ge=0, le=1)
    frozen_price_run: int = Field(5, ge=2)
    suspicious_return_abs: float = Field(0.40, gt=0)

    # --- conventions --------------------------------------------------------------
    trading_days: int = Field(252, ge=1)
    return_method: ReturnMethod = ReturnMethod.SIMPLE
    risk_free_rate: float | None = Field(None, ge=-0.05, le=1.0)
    risk_free_rate_source: str = "user"

    # --- estimation ---------------------------------------------------------------
    expected_return_method: ExpectedReturnMethod = ExpectedReturnMethod.JAMES_STEIN
    covariance_method: CovarianceMethod = CovarianceMethod.AUTO
    ewma_lambda: float = Field(0.94, gt=0, lt=1)

    # --- portfolio constraints -------------------------------------------------
    allow_short: bool = False
    max_leverage: float = Field(1.0, ge=1.0)
    min_weight: float = Field(0.0, ge=-1.0, le=1.0)
    max_weight: float | None = Field(None, gt=0, le=1.0)
    transaction_cost_bps: float = Field(10.0, ge=0)
    spread_bps: float = Field(5.0, ge=0)

    # --- simulation -------------------------------------------------------------
    n_simulations: int = Field(20_000, ge=100, le=2_000_000)
    horizon_days: int = Field(252, ge=1)
    simulation_model: SimulationModel = SimulationModel.GBM
    simulation_batch_size: int = Field(5_000, ge=100)
    seed: int = 42
    confidence_levels: list[float] = Field(default_factory=lambda: [0.95, 0.99])
    var_method: VaRMethod = VaRMethod.HISTORICAL

    inflation_annual: float = Field(0.045, ge=-0.05, le=1.0,
                                    description="HYPOTHESIS used to deflate simulated wealth")
    goal: float | None = Field(None, gt=0, description="explicit nominal wealth goal (optional)")
    profile_horizon_simulation: bool = True
    n_resamples: int = Field(50, ge=0, le=2000)
    evt_min_exceedances: int = Field(50, ge=10)

    # --- backtest ---------------------------------------------------------------
    backtest_lookback: int = Field(252, ge=60)
    rebalance_every: int = Field(21, ge=1)
    slippage_bps: float = Field(5.0, ge=0)
    validation_frac: float = Field(0.6, gt=0.1, lt=0.95)

    # --- output -----------------------------------------------------------------
    output_dir: Path = Path("outputs")
    make_charts: bool = True
    profile: str = "moderado"
    run_heavy_analyses: bool = True

    @field_validator("tickers")
    @classmethod
    def _tickers_not_empty(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in v if t and t.strip()]
        if len(cleaned) < 2:
            raise ValueError("at least two tickers are required for portfolio analysis")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("duplicate tickers in universe")
        return cleaned

    @field_validator("confidence_levels")
    @classmethod
    def _conf_levels(cls, v: list[float]) -> list[float]:
        for c in v:
            if not 0.5 < c < 1.0:
                raise ValueError(f"confidence level {c} must be in (0.5, 1)")
        return sorted(set(v))

    @field_validator("base_currency")
    @classmethod
    def _ccy(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3:
            raise ValueError("base_currency must be an ISO-4217 code such as BRL or USD")
        return v

    @model_validator(mode="after")
    def _resolve_risk_free(self) -> "AppConfig":
        if self.risk_free_rate is None:
            fallback = DEMO_RISK_FREE_FALLBACK.get(self.base_currency)
            if fallback is None:
                raise ValueError(
                    f"no risk-free rate supplied and no demo fallback for {self.base_currency}; "
                    "pass --risk-free-rate explicitly"
                )
            self.risk_free_rate = fallback
            self.risk_free_rate_source = "demo_fallback"
        if not self.allow_short and self.min_weight < 0:
            raise ValueError("min_weight < 0 requires allow_short=True")
        return self

    @property
    def rf_is_fallback(self) -> bool:
        return self.risk_free_rate_source == "demo_fallback"

    @property
    def charts_dir(self) -> Path:
        return self.output_dir / "charts"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def logs_dir(self) -> Path:
        return self.output_dir / "logs"

    def ensure_dirs(self) -> None:
        for d in (self.charts_dir, self.reports_dir, self.tables_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
