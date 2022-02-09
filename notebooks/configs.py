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
                    "num_intervals": 48
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
            "policy_config": {},
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
            "policy_config": {"num_lookahead_steps": lookahead},
            "training": False,
            "dry_run": 0,
            "results_dir": results_dir
        }
    }

    return configs
