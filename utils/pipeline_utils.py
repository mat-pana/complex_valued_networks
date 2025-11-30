from experiments.registry import build
from experiments.single_experiment import single_experiment


def clone_with(d, **overrides):
    x = dict(d)
    x.update(overrides)
    return x


def expand_activation_list(run_cfg):
    params = run_cfg.get("params", {})
    act = params.get("activation", None)

    if not isinstance(act, list):
        return [run_cfg]

    tag_base = run_cfg.get("tag", run_cfg["name"])
    expanded = []
    for a in act:
        new_params = dict(params)
        new_params["activation"] = a
        new_tag = f"{tag_base}_act-{a}"
        expanded.append(clone_with(run_cfg, params=new_params, tag=new_tag))
    return expanded


def expand_runs(runs):
    out = []
    for r in runs:
        out.extend(expand_activation_list(r))
    return out


def inject_input_configs(config):
    device = config["globals"]["device"]
    dataset = config["dataset"]
    real_size = 512 if dataset == "FaultDetectionA" else 500
    complex_size = int(real_size / 2) + 1
    learning_rate = 0.0001 if dataset == "FaultDetectionA" else 0.001
    n_epochs = 500 if dataset == "FaultDetectionA" else 1000


    for idx, run in enumerate(config["runs"]):
        run_name = run["name"]
        if run_name == "RealFeedForward":
            config["runs"][idx]["params"]["input_size"] = real_size
        elif run_name == "ComplexFeedForward":
            config["runs"][idx]["params"]["input_size"] = complex_size
        elif run_name == "DualBranchFusion":
            config["runs"][idx]["params"]["cv_branch"]["params"][
                "input_size"
            ] = complex_size

        config["runs"][idx]["params"]["device"] = device

        if run_name == "DualBranchFusion":
            for branch in ("rnn_branch", "cv_branch"):
                config["runs"][idx]["params"][branch]["params"]["device"] = device

        config["globals"]["learning_rate"] = learning_rate
        config["globals"]["n_epochs"] = n_epochs

    return config


def run_one(run_cfg, globals_cfg, dataset):
    model_cfg = dict(run_cfg)
    learning_rate = globals_cfg.get("learning_rate", "results")
    n_epochs = globals_cfg.get("n_epochs", "results")
    result_dir = globals_cfg.get("result_dir", "results")
    device = globals_cfg.get("device", "cuda:0")
    seed = globals_cfg.get("seed", None)

    model = build(model_cfg)
    tag = run_cfg.get("tag", run_cfg["name"])

    print(f"\n=== Running {tag} on {device} ===")
    single_experiment(
        model=model,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        experiment_name=tag,
        dataset=dataset,
        log_path=result_dir,
        seed=seed,
        device=device,
    )
