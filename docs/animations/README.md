# VSA2 load balancing animation

Open [the standalone HTML](vsa2-load-balance-standalone.html) in a browser, or use
[the Canvas component](vsa2-load-balance.canvas.tsx) in Codex Canvas / a React host.
The separate [HTML fragment](vsa2-load-balance.html) is intended for the conversation visualization host.

The example computes attention for Q (8×D), Kᵀ (D×8) and V (8×D), with D=4 numerically.
It keeps 16 selected query–key pairs across uniform, hot-Q, hot-KV and combined cases.
Switch forward/backward and step through the worker timelines to see splitting and merging.
The worker count, chunk size, router rule and time units are illustrative; they are not H100 benchmark results.
See [the actual H100 policy, implementation and measurements](../vsa2_splitkv.md).

Below the toy timeline, **H100 measurements** show actual FWD, BWD and F+B latency in milliseconds,
including both speedups and regressions. Four Auto C64 examples are visible immediately; expand the
comparison to see all 12 historical cases with Manual C64 and Auto C64, including the uniform control,
long sequences and the explicitly labelled H2/K4 diagnostic. The checked-in
[`h100-results.json`](vsa2-load-balance/h100-results.json) records the matched medians, percentage changes,
actual split counts and original result-file hashes from existing H100 job 802632; no new GPU run was used.
All are frozen synthetic selections, with planner/merge costs included and router excluded, median of 10
after three warmups. The phases were timed independently; F+B is not their arithmetic sum.

Forward combines partial outputs using their LSE weights. Backward uses the saved global softmax.
The backward sum illustrates the mathematics; the H100 TK implementation uses TMA reduce-add into
existing FP32 accumulators rather than a separate dK/dV partial-merge kernel.

The shared sources are in `vsa2-load-balance/`. From the repository root, with Node.js:

```sh
node docs/animations/vsa2-load-balance/build.js
node docs/animations/vsa2-load-balance/build.js --check
node docs/animations/vsa2-load-balance/check.js
```

The build regenerates the fragment and Canvas component. The standalone HTML is a checked-in export
with the visualization host styles and runtime embedded, so it can be opened directly.

The checks cover the selected-pair count, attention/gradient equivalence, scheduling, both directions,
router modes, seeking, playback, reduced motion and cleanup. The animation requires no GPU or model files.
