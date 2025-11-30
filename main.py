from utils.config import config
from utils.pipeline_utils import *


def main():
    datasets = ["FordA", "FordB", "FaultDetectionA"]
    for dataset in datasets:
        config["dataset"] = dataset
        cfg = inject_input_configs(config)
        globals_cfg = cfg.get("globals", {})

        runs = cfg.get("runs", None)
        runs = expand_runs(runs)

        for r in runs:
            run_one(r, globals_cfg, dataset)


if __name__ == "__main__":
    main()
