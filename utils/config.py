config = {
    "experiment_name": "architecture_comparison",
    "dataset": "FaultDetectionA",
    "globals": {
        "result_dir": "results/",
        "device": "cuda:0",
        "seed": 42,
    },
    # Architectures parameters.
    "runs": [
        {
            "name": "RealFeedForward",
            "tag": "rff_20x10",
            "params": {
                "hidden_layers": [20, 10],
                "dropout_val": 0.2,
                "output_size": 1,
            },
        },
        {
            "name": "ComplexFeedForward",
            "tag": "complex_ff_20x10",
            "params": {
                "hidden_layers": [20, 10],
                "dropout_val": 0.2,
                "activation": [
                    "modrelu",
                    "cardioid",
                    "splitgelu",
                    "zrelu",
                    "crelu",
                    "cleakyrelu",
                    "celu",
                    "softsign",
                ],
            },
        },
        # {
        #     "name": "RealRNN",
        #     "tag": "rnn_30x10",
        #     "params": {
        #         "input_size": 1,
        #         "hidden_layers": [30],
        #         "head": [10],
        #         "output_size": 1,
        #         "cell": "gru",
        #         "dropout_val": 0.2,
        #         "bidirectional": True,
        #         "pooling": "last",
        #     },
        # },
        # {
        #     "name": "DualBranchFusion",
        #     "tag": "fusion",
        #     "params": {
        #         "rnn_branch": {
        #             "name": "RealRNN",
        #             "params": {
        #                 "input_size": 1,
        #                 "hidden_layers": [15, 10],
        #                 "output_size": 1,
        #                 "cell": "gru",
        #                 "dropout_val": 0.2,
        #                 "bidirectional": True,
        #                 "no_top": True,
        #                 "pooling": "last",
        #             },
        #         },
        #         "cv_branch": {
        #             "name": "ComplexFeedForward",
        #             "params": {
        #                 "hidden_layers": [10],
        #                 "activation": "celu",
        #                 "dropout_val": 0.2,
        #                 "no_top": True,
        #                 "init": "xavier",
        #             },
        #         },
        #         "head_hidden": [10],
        #         "head_dropout": 0.1,
        #         "head_activation": "relu",
        #         "branch_norm": False,
        #     },
        # },
    ],
}
