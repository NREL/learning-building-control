import os

# Directories
_this_dir = os.path.abspath(os.path.dirname(__file__))
DEFAULT_RESULTS_DIR = os.path.join(_this_dir, "results")

# Configuration dicts for scenarios.  Use SCENARIO_TEST for short runs.
SCENARIO_DEFAULT = {
    "start_time": "00:05:00",
    "end_time": "23:55:00",
    "zone_temp_init_mean": 26.0
}

SCENARIO_TEST = {
    "start_time": "11:00:00",
    "end_time": "13:00:00",
    "zone_temp_init_mean": 23.0
}

# Params that are re-used
DR_PROGRAM = "TOU"
LOOKAHEAD = 4

# All configs get these settings
COMMON = {
    "batch_size": 31,
    "dr_program": DR_PROGRAM,
    "scenario_config": SCENARIO_DEFAULT,
    "results_dir": DEFAULT_RESULTS_DIR
}


## DPC POLICY
DPC = {
    "policy_type": "DPC",
    "policy_config": {
        "model_config": {
            "hidden_dim": 128,
            "num_time_windows": 24
        },
        "lr": 1e-2,
        "num_epochs": 1000,
    }
}
DPC.update(COMMON.copy())


## MPCOneShot POLICY
MPCOneShot = {
    "policy_type": "MPCOneShot",
    "policy_config": {
        "tee": False
    }
}
MPCOneShot.update(COMMON.copy())


## MPC POLICY
MPC = {
    "policy_type": "MPC",
    "policy_config": {
        "num_lookahead_steps": LOOKAHEAD,
        "tee": False
    }
}
MPC.update(COMMON.copy())


## REINFORCEMENT LEARNING POLICY
RLC = {
    "policy_type": "RLC",
    "policy_config": {
        "node_ip_address": None
    }
}
RLC.update(COMMON.copy())


## CPL CONFIG
CPL = {
    "policy_type": "CPL",
    "policy_config": {
        "lookahead": LOOKAHEAD,
        "lr": 10,
        "num_epochs": 50,
        "use_value_function": 0,
        "num_time_windows": 24,
    }
}
CPL.update(COMMON.copy())


## RBC POLICY
RBC_SETPOINTS = {
    "TOU": [(0, 27), (83, 24), (131, 21), (143, 24), (215, 24), (215, 27)],
    "PC": [(0, 27), (83, 24), (143, 21), (155, 24), (203, 24), (215, 27)],
    "RTP": [(0, 27), (83, 24), (131, 21), (143, 24), (215, 27)]
}
RBC = {
    "policy_type": "RBC",
    "policy_config": {
        "setpoints": RBC_SETPOINTS[DR_PROGRAM],
        "p_flow": 1.0,
        "p_temp": 1.0
    }
}
RBC.update(COMMON.copy())


CONFIGS = {
    "MPCOneShot": MPCOneShot,
    "MPC": MPC,
    "CPL": CPL,
    "DPC": DPC,
    "RLC": RLC,
    "RBC": RBC
}


def get_config(name, **args):
    
    config = CONFIGS[name].copy()

    # Args are assumed to be top-level config values for the 
    # policy runner.
    common = {k: v for k, v in args.items()}
    config.update(common)

    return config


