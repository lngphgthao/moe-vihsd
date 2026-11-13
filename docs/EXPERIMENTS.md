# Experiment Registry

## Status Definitions

- PLANNED — experiment has been defined but not executed
- RUNNING — currently executing
- COMPLETED — successfully executed and results recorded
- FAILED — execution failed
- INVALID — result should not be used
- ABANDONED — intentionally discontinued

## Experiments

| ID   | Dataset | Backbone     | Architecture | Router | Experts | LR   | Seed | Status  |
| ---- | ------- | ------------ | ------------ | ------ | ------- | ---- | ---- | ------- |
| E001 | ViANLI  | PhoBERT-base | Dense        | -      | -       | 2e-5 | 42   | PLANNED |
| E002 | ViANLI  | XLM-R-base   | Dense        | -      | -       | 2e-5 | 42   | PLANNED |
| E003 | ViANLI  | PhoBERT-base | MoE          | Top-1  | 4       | 2e-5 | 42   | PLANNED |
| E004 | ViANLI  | XLM-R-base   | MoE          | Top-1  | 4       | 2e-5 | 42   | PLANNED |
| E005 | ViHSD   | PhoBERT-base | Dense        | -      | -       | 2e-5 | 42   | PLANNED |
| E006 | ViHSD   | XLM-R-base   | Dense        | -      | -       | 2e-5 | 42   | PLANNED |
