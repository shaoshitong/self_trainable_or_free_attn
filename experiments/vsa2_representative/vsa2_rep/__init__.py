"""VSA2 attention with a three-tier representative KV budget (native CUDA / ThunderKittens-style).

VSA2 keeps the top-k KV blocks of its router and drops the rest. Here the dropped blocks are kept at
reduced resolution instead: the top `frac_full` of the (query block, kv block) pairs keep all 64
tokens, the next `frac_mid` keep `mid_reps` representatives, and every remaining pair keeps the block
mean. A representative of c tokens enters the one online softmax with `+ log2(c)` on its logit, so it
carries the mass of the tokens it stands for in both the numerator and the denominator.

Routing is VSA2's own PSP router and exact global top-k, run at two budgets. Representatives are plain
means of the contiguous router groups -- no clustering, no reordering, no reuse of an earlier lse.
"""
from .ops import (PHASE_A, PHASE_ALL, PHASE_B, PHASE_C, RepTiers, TierSelection, rep_attention,
                  rep_fine_attention, route_tiers)

__all__ = ["PHASE_A", "PHASE_ALL", "PHASE_B", "PHASE_C", "RepTiers", "TierSelection", "rep_attention",
           "rep_fine_attention", "route_tiers"]
