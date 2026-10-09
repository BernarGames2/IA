"""Black Swan Lab facade: extreme-risk engine independent from the basic Monte Carlo."""

from core.extreme_risk.evt import EVTRefusal, evt_analysis, fit_pot, gpd_var_es
from core.extreme_risk.reverse_stress import (
    empirical_breach_frequency,
    gaussian_reverse_stress,
    scenario_multipliers,
)
from core.extreme_risk.stress import (
    Scenario,
    apply_scenario,
    default_scenarios,
    historical_worst_windows,
    liquidity_analysis,
    named_crisis_replay,
    parametric_stress_var,
    run_scenarios,
)
from core.extreme_risk.tail_dependence import empirical_lower_tail_dependence, fit_t_copula
from core.extreme_risk.var_es import (
    cornish_fisher_var_es,
    empirical_var_es,
    fit_student_t,
    historical_var_es,
    normal_var_es,
    simulated_var_es,
    sqrt_time_scaled,
    student_t_var_es,
    var_es_table,
)

__all__ = [n for n in dir() if not n.startswith("_")]
