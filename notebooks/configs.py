from lbc.experiments.runner import SCENARIO_DEFAULT

results_dir = "./_scratch"

def make_configs(dr, batch_size, lookahead=None):

    configs = {
        "DPC": {
            "name": f"DPC-{dr}-test",
            "policy_type": "DPC",
            "batch_size": batch_size,
            "dr_program": dr,
            "scenario_config": SCENARIO_DEFAULT,
            "policy_config": {
                "model_config": {
                    "hidden_dim": 128,
                    "num_time_windows": 24
                },
                "lr": 1e-2,
                "num_epochs": 1000,
            },
            "training": True,
            "dry_run": 0,
            "results_dir": results_dir
        },
        "MPCOneShot": {
            "name": f"MPCOneShot-{dr}-test",
            "policy_type": "MPCOneShot",
            "dr_program": dr,
            "batch_size": batch_size,
            "scenario_config": SCENARIO_DEFAULT,
            "policy_config": {"tee": False},
            "training": False,
            "dry_run": 0,
            "results_dir": results_dir
        },
        "MPC": {
            "name": f"MPC-{dr}-test",
            "policy_type": "MPC",
            "batch_size": batch_size,
            "dr_program": dr,
            "scenario_config": SCENARIO_DEFAULT,
            "policy_config": {"num_lookahead_steps": lookahead, "tee": False},
            "training": False,
            "dry_run": 0,
            "results_dir": results_dir
        },
        "RLC": {
            "name": f"RLC-{dr}-test",
            "policy_type": "RLC",
            "dr_program": dr,
            "batch_size": batch_size,
            "scenario_config": SCENARIO_DEFAULT,
            "policy_config": {
                "node_ip_address": None
            },
            "training": False,
            "dry_run": 0,
            "results_dir": results_dir
        },
        "CPL": {
            "name": f"CPL-{dr}-test",
            "policy_type": "CPL",
            "batch_size": batch_size,
            "dr_program": dr,
            "scenario_config": SCENARIO_DEFAULT,
            "policy_config": {
                "lookahead": lookahead,
                "lr": 10,
                "num_epochs": 50,
                "use_value_function": 0,
                "num_time_windows": 24,
            },
            "training": True,
            "dry_run": 0,
            "results_dir": results_dir
        }
    }

    return configs
