# Invariants for this repository

Read this before editing training or pruning code.

- Do not select checkpoints by test accuracy. The reported number is the last epoch. `n_test_evals` must stay 1 on full-split training runs.
- Do not change 160/40 epoch budgets, `kappa_max` for Scratch-B, or the frozen file `conf/frozen_hparams.json` after the git tag `hparams-v1`. A bugfix that changes results restarts every affected run under a new hash.
- Structured pruning removes channels. A mask on weights is not a pruned model. ResNet prunes only the internal channels of each block (conv1/bn1), never the residual stream.
- Network Slimming L1 is added to the gamma gradient in fp32. It is not weight decay.
- Unstructured masks are applied to the gradient before the step and to the weights after the step.
- Augmentation draws a crop and a flip for every image.
- `init_seed = seed + 1000 * mode_id`. Scratch and the base of the same seed must not start from the same weights.
- Tests in `tests/` stay. A run with skipped tests is not a passing G0.
- Train on the RTX 5070 Ti only (`CUDA_VISIBLE_DEVICES=1`, capability (12, 0)). Do not use the RTX 2070.
- Do not enable `torch.compile`.
