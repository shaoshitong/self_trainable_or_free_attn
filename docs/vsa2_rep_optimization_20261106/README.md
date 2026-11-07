# VSA2 representative on H100 — optimisation record (2026-11-06)

Chinese-language records for the campaign that produced the current default path of
`experiments/vsa2_representative`. The English summary lives in
`docs/vsa2_representative_handoff.md`; these files are the raw material behind it.

| file | what it is |
|---|---|
| `REPORT.md` | reader-facing summary: six accepted changes, the cost breakdown, the refuted directions |
| `RESULTS.md` | the full record, 31 sections in chronological order, including the measurement rules, every A/A floor, and the hand-off section (§21) |
| `NEXT_STEPS.md` | the plan that was worked through item by item, including the items added during the campaign (P6/P7/P8) |
| `OPTION_CATALOG.md` | the 126 candidate optimisations gathered from outside sources, with the verdict for each one that was tried |

## Result at this commit

| | before | after | change |
|---|---|---|---|
| full-operator forward | 1.3704 ms | **~1.128 ms** | **-17.7%** |
| full F+B | 4.1789 ms | **~3.873 ms** | **-7.3%** |

Six accepted changes, each with a config that restores what it replaced so the A/B can be re-run on this exact
source: **20** (tier-B gather), **37** (tier-A dK/dV store), **40** (coarse metadata cache), **43** (representative
tables from the router pools), **45** (sequential top-k chains), **47** (three-kernel router column stage).
The default path (`REP_TK_FWD_CFG` unset) is the fastest configuration measured.

Fourteen measured refutations and four directions that were assessed and declined with their reasoning; all of
them are in `RESULTS.md` §2 and §31. Nothing here is a hardware-counter measurement: `ncu` is not permitted on
this machine (`ERR_NVGPUCTRPERM`), so every efficiency statement is a wall-clock A/B against a same-run A/A arm.

## Two things to know before quoting a number

- Absolute times drift ~2% between runs; only same-run A/B ratios are claims.
- The operator's own backward is not run-to-run deterministic (`dq`/`dk`/`dv` differ in ~300-1200 of 33.5M
  elements), so gradient comparisons need a tolerance, never byte equality. `RESULTS.md` §29 has the numbers.
