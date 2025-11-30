To execute experiments, datasets should be downloaded from
https://timeseriesclassification.com and placed in the directories like follows:

.
└── datasets/
    ├── FaultDetectionA/
    │   ├── FaultDetectionA_TEST.ts
    │   └── FaultDetectionA_TRAIN.ts
    ├── FordA/
    │   ├── FordA_TEST.ts
    │   └── FordA_TRAIN.ts
    └── FordB/
        ├── FordB_TEST.ts
        └── FordB_TEST.ts

Only .ts files are needed for experiments.

Current links to the dataset:
FaultDetectionA: https://timeseriesclassification.com/description.php?Dataset=FaultDetectionA
FordA: https://timeseriesclassification.com/description.php?Dataset=FordA
FordB: https://timeseriesclassification.com/description.php?Dataset=FordB

Basic configuration for all the architectures is contained in utils/config.py.

All experiments can be run at once from main.py.
