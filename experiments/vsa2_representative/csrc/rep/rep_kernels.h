// VSA2 representative-block attention: host-visible parameter structs and launchers.
//
// This extends VSA2's native fine block-sparse attention with a three-tier KV budget. VSA2 keeps the
// top-k KV blocks of its native router and drops the rest; here the dropped blocks are kept at reduced
// resolution instead:
//
//   tier A ("full")  the most important frac_full of the (query block, kv block) pairs: all 64 tokens,
//                    exactly VSA2's native fine attention.
//   tier B ("mid")   the next frac_mid of the pairs: RM representatives per kv block (default RM = 4),
//                    each the plain mean of one contiguous 64/RM-token group.
//   tier C ("tail")  every remaining pair: one representative per kv block, the plain block mean.
//
// A representative that stands for c tokens enters the softmax as a single column whose log2-domain
// logit carries + log2(c):
//
//   sum_{i in S} exp(scale * q.k_i) v_i  ~=  c * exp(scale * q.kbar) * vbar
//
// so the same factor c appears in the numerator and the denominator of the one online softmax that
// runs over all three tiers. The result stays a convex combination of values, is exact when the tokens
// of a group are identical, and reduces to dense attention at RM = 64. An empty group carries
// log2(c) = -inf and is excluded. There is no clustering: groups are the contiguous R-token router
// groups of VSA2's own block layout (R = 64/RM, i.e. the native R = 16 groups at RM = 4), so token
// order and the layout contract are untouched.
//
// Tier assignment comes from VSA2's own router and its exact global top-k, run at two budgets:
// budget_A = frac_full * NB * NB selects tier A (and yields its CSR lists), budget_AB selects A | B.
// The per-tier membership of every pair is then read from the two selection bitmaps, so tiers B and C
// need no CSR list and no NB*NB index array: tier C is the complement of A | B.
//
// Tensor conventions follow vsa2_kernels.h (contiguous, L = NB * 64, block-padded, valid tokens first):
//   token tensors   q, k, v, o, dO : [BH, L, D]
//   mid reps        k_mid, v_mid   : [BH, NB * RM, D]   row = blk * RM + g
//                   lc_mid         : [BH, NB * RM] fp32  log2(valid tokens of the group), -inf if empty
//   tail reps       k_tail, v_tail : [BH, NB, D]        row = blk
//                   lc_tail        : [BH, NB] fp32       log2(blk_cnt[blk])
//   selection bits  bits_a/bits_ab : [BH, NB, W] uint32, W = ceil(NB/32); bit j of row i set when
//                                    pair (query block i, kv block j) is in the tier set.
//   lse                            : [BH, L] fp32 natural log-sum-exp over every column of every tier.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "vsa2_kernels.h"

namespace vsa2rep {

using vsa2::DType;
constexpr int kBlock = vsa2::kBlock;  // 64

// Phase mask (ablation / validation): bit 0 tier A, bit 1 tier B, bit 2 tier C.
enum { kPhaseA = 1, kPhaseB = 2, kPhaseC = 4, kPhaseAll = 7 };

inline int words_of(int nb) { return (nb + 31) >> 5; }

// ---------------------------------------------------------------- representative construction
// Plain group means over valid tokens; no clustering, no reordering. RM in {1, 2, 4, 8, 16}.
struct RepBuildParams {
  const void* k;
  const void* v;
  void* k_mid;
  void* v_mid;
  float* lc_mid;
  void* k_tail;
  void* v_tail;
  float* lc_tail;
  const int* blk_cnt;
  int BH, L, NB, RM;
};
void launch_rep_build(const RepBuildParams& p, int D, DType dt, cudaStream_t s);

// Backward of the construction: dK/dV of a representative is spread over the tokens it averaged,
// dx_i += drep / c. dk/dv = cast(acc_k + mid/c_group + tail/c_block), zero on padded slots.
// acc_k / acc_v are the tier-A fp32 token gradients (nullable, treated as 0).
struct RepUnpoolParams {
  const float* acc_k;
  const float* acc_v;
  const float* dk_mid;
  const float* dv_mid;
  const float* dk_tail;
  const float* dv_tail;
  void* dk;
  void* dv;
  const int* blk_cnt;
  int BH, L, NB, RM;
  int NBt;  // rows per (b,h) of the tail gradient tables (NB, or NB padded to 64 by the TK path)
};
void launch_rep_unpool(const RepUnpoolParams& p, int D, DType dt, cudaStream_t s);

// Tier-B CSR lists from the A|B and A selections: row i of tier B is row i of A|B with the tier-A
// columns removed (ascending). The global top-k selects exactly its budget per (b,h), so
// ptr_B = ptr_AB - ptr_A is exact and absolute, and the lists need no count pass. One warp per row.
// The TK kernels walk these lists instead of scanning bitmaps inside the attention loop.
struct TierBListParams {
  const int* ab_ptr;  // [BH*(NB+1)] A|B q2k CSR
  const int* ab_idx;
  const int* a_ptr;   // [BH*(NB+1)] tier-A q2k CSR
  const uint32_t* bits_a;
  int* b_ptr;  // [BH*(NB+1)] written
  int* b_idx;  // [BH*(budget_ab - budget_a)] written
  int BH, NB, W;
};
void launch_tier_b_lists(const TierBListParams& p, cudaStream_t s);

// delta[row] = sum_d dO[row, d] * O[row, d], 0 on padded rows. (Backward needs the global delta of
// the fused output; this is the only statistic the three tiers share besides lse.)
struct RowDotParams {
  const void* a;
  const void* b;
  float* out;
  const int* blk_cnt;
  int BH, L, NB;
};
void launch_row_dot(const RowDotParams& p, int D, DType dt, cudaStream_t s);

// ---------------------------------------------------------------- three-tier forward
struct RepAttnFwdParams {
  const void* q;
  const void* k;
  const void* v;
  const void* k_mid;
  const void* v_mid;
  const float* lc_mid;
  const void* k_tail;
  const void* v_tail;
  const float* lc_tail;
  void* o;
  float* lse;
  const int* row_ptr;  // tier-A q2k CSR, [BH * (NB + 1)]
  const int* col_idx;
  const uint32_t* bits_a;
  const uint32_t* bits_ab;
  const int* blk_cnt;
  int BH, L, NB, RM, W;
  float scale_log2;
  int phases;
  // Diagnostic only, to isolate what a gathered tile pays over a contiguous one (no hardware
  // counters are available on the test host). 0 = normal. >=1: a gathered tile's columns carry no
  // log2-count bias, so the softmax skips its shared-memory bias reads. >=2: additionally the tile
  // gathers table rows 0..63, i.e. contiguously, so the scattered row gather and expand_tile are
  // bypassed. Both give wrong results; never use outside measurement.
  int diag;
};
void launch_rep_attn_fwd(const RepAttnFwdParams& p, int D, DType dt, cudaStream_t s);

// ---------------------------------------------------------------- three-tier backward (query owner)
// One CTA per (b*h, query block) walks the same column stream as the forward and recomputes P from
// the forward lse. dQ is accumulated in registers over all three tiers and written once: exact and
// deterministic, no atomics. The representative key/value gradients of tiers B and C are reduced into
// the (small) fp32 rep-gradient tables with atomics, since a representative row is shared by many
// query blocks; that reduction order is nondeterministic, like VSA2's own TK dK/dV accumulators.
// Tier-A token dK/dV are not produced here: vsa2::launch_attn_bwd on the tier-A k2q lists computes
// them exactly, with bias 0 on every tier-A column, which is what this kernel's tier A also uses.
struct RepAttnBwdQParams {
  const void* q;
  const void* k;
  const void* v;
  const void* k_mid;
  const void* v_mid;
  const float* lc_mid;
  const void* k_tail;
  const void* v_tail;
  const float* lc_tail;
  const void* dout;
  const float* lse;    // natural
  const float* delta;  // rowsum(dO * O)
  void* dq;            // [BH, L, D] Elt, written once
  const float* dq_acc_in;  // optional [BH, L, D] fp32 dQ of the tiers this call does not walk (added,
                           // not atomically: a CTA owns its query block). Lets tier A's dQ come from
                           // VSA2's KV-owner kernel while this call covers only tiers B and C.
  const float* add_dq;  // optional [BH, add_nb, D] per-block gradient, added as add/cnt (coarse branch)
  int add_nb;
  float* dk_mid;       // [BH, NB * RM, D] fp32, atomically accumulated (must be zeroed)
  float* dv_mid;
  float* dk_tail;  // [BH, NB, D] fp32, atomically accumulated (must be zeroed)
  float* dv_tail;
  const int* row_ptr;  // tier-A q2k CSR
  const int* col_idx;
  const uint32_t* bits_a;
  const uint32_t* bits_ab;
  const int* blk_cnt;
  int BH, L, NB, RM, W;
  float scale;
  float scale_log2;
  int phases;
  // Diagnostic only (like VSA2's dq_word), to attribute the cost of the representative tiers without
  // hardware counters. 0 = normal. 1 = keep the two representative MMAs but skip the atomic scatter.
  // 2 = skip both. Either value makes dK/dV wrong; never use outside measurement.
  int diag;
};
void launch_rep_attn_bwd_q(const RepAttnBwdQParams& p, int D, DType dt, cudaStream_t s);

}  // namespace vsa2rep
