# Candidate validation worker

Portable execution code for source 3c291f27e2a147ee515516c3c8503aee9a99294b. The controller supplies all access/resource/installation settings separately; none is stored here. The source and pinned TK manifests identify the immutable runtime closure.

The worker performs a fresh build and expanded CPU/generic/TK correctness checks, checks the manual-softmax control, then runs A/A and CFG6-versus-CFG1 comparisons. A longer sequence runs only after a measured full F+B gain exceeds the A/A displacement and 1%. Diagnostics remain kernel-only. Device telemetry is bound to the actual allocated device by its private runtime identity.

Only sanitized build/test/performance feedback should be shared with coding assistants. Original baseline execution scripts remain in the parent directory for reproduction.
