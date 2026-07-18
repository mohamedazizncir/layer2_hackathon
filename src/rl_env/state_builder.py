import numpy as np


STATE_FIELDS = [
    "stockout_rate_chronic",
    "stockout_rate_essential",
    "stockout_rate_comfort",
    "season_sin",
    "season_cos",
    "import_disruption_active",
    "epidemic_active",
]


def state_dict_to_array(state) -> np.ndarray:
    """Flatten simulator context in the fixed RL observation order."""
    return np.asarray([state[field] for field in STATE_FIELDS], dtype=np.float32)
