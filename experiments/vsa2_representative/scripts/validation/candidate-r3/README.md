# R3 candidate validation worker

Portable execution code for frozen source `882156dbd14908520d1d4066e37c9218602ca8da`.
The controller supplies access, resource policy, compiler locations and task paths through environment variables;
none of that configuration is included here. The source and pinned ThunderKittens manifests identify the
immutable runtime closure. Copy these execution files under `_control/` when assembling that closure.

The worker performs a fresh build and CPU, benchmark-helper, generic and TK correctness checks, including
forward configurations 8 and 9. At length 16,384 it runs matched A/A (CFG1 versus CFG1), vector (CFG1 versus
CFG8), and manual-softmax (CFG6 versus CFG9) comparisons. A length-32,768 comparison runs only if the best
measured full-operator F+B gain exceeds both the A/A displacement and 1%; this is an experiment-routing rule,
not a significance test. The checker validates measured timings, output checks, the unchanged input/selection
identity and source/build identity. Kernel-only timings and backward estimates remain separately labelled.

Raw execution identities and logs remain with the controller; share only sanitized relative build/test/performance
feedback. This directory preserves the R3 worker separately from the earlier `../candidate/` worker and the
original baseline worker in the parent directory. Publishing these scripts does not report a completed GPU run.
