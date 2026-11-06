// VSA2 native kernels: host-visible parameter structs and launchers.
//
// Tensor conventions (all contiguous):
//   token tensors  q, k, v, o, dO : [B*H, L, D]      L = NB * 64 (block-padded layout)
//   blk_cnt                       : [NB] int32       valid tokens per 64-token block, 1..64
//                                                     (valid tokens first inside a block)
//   CSR lists                     : row_ptr [B*H*(NB+1)] int32 absolute offsets into col_idx
//   lse                           : [B*H, L] fp32    natural log-sum-exp of scale*q.k over the
//                                                     selected keys; -inf for rows with no
//                                                     selected key and for padded rows
//                                                     (generic backend).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace vsa2 {

constexpr int kBlock = 64;  // hardware block size B (tokens per block)
inline constexpr float kLog2eHost() { return 1.4426950408889634f; }

enum class DType : int { kBF16 = 0, kFP16 = 1 };

struct AttnFwdParams {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  float* lse;
  const int* row_ptr;  // q2k
  const int* col_idx;
  const int* blk_cnt;
  int BH;
  int L;
  int NB;
  float scale;
  float scale_log2;
  // Split-KV task mode (split.cu plan), all nullptr in the baseline (NB, BH) grid. tasks[t] =
  // {row = bh*NB + qb, edge_begin, edge_end, slot}; slot < 0 writes o/lse directly, slot >= 0
  // writes an fp32 normalized partial part_o[slot] [64, D] and its log2-domain lse part_lse[slot].
  const int* tasks;      // [n_tasks_cap, 4]
  const int* task_info;  // device [4]: total tasks, split rows, split slots, -
  int n_tasks_cap;
  float* part_o;
  float* part_lse;
};

// ---------------------------------------------------------------- split-KV planning
// Rows (b*h, block) of a CSR list are eligible when n >= thr_h (below); each eligible row becomes
// ceil(n/chunk) balanced chunk tasks with one partial slot each, while the running slot count stays
// <= max_slots (later long rows run unsplit), and only if the number of eligible rows E <= budget
// (otherwise every row runs direct). Task order: split chunks first (row order), then one direct task
// per remaining row. Every count lives on the device; launch grids use the host bounds
// n_rows + max_slots (tasks) and min(n_rows, max_slots / 2) (merges); CTAs past the device count exit.
//   thr_h = max(min_len, ceil(rel_mean * nnz_h / NB), ceil(tail_frac * W / resident))   (doubles)
//   nnz_h = edges of row h's (b, h), W = all edges of the list. min_len >= chunk + 1, so a split row
//   has >= 2 chunks. Manual SplitKV: min_len = chunk + 1, rel_mean = tail_frac = 0, budget = INT_MAX.
struct SplitPlanParams {
  const int* ptr;  // [BH*(NB+1)]
  int BH, NB, chunk, max_slots;
  int min_len;           // absolute floor (>= chunk + 1)
  double rel_mean;       // relative to the (b, h) mean list length; 0 = off
  double tail_frac;      // fraction of W / resident (work per resident CTA); 0 = off
  int resident;          // resident CTAs of the consuming kernel (SMs x CTAs/SM), >= 1
  int budget;            // max eligible rows for any split; INT_MAX = unlimited
  int* cnt;              // [2, NR] chunks if eligible, eligible flag
  const int* cnt_incl;   // [2, NR] inclusive scans
  int* flags;            // [3, NR] direct, split chunks, split flag
  const int* flags_incl; // [3, NR] inclusive scans
  int* tasks;            // [NR + max_slots, 4]
  int* merges;           // [max(1, min(NR, max_slots/2)), 4] {row, slot0, chunks, -}
  int* info;             // [8] tasks, split rows, split slots, NR, eligible rows E, E <= budget,
                         //     thr without the per-head term, W
};
void launch_split_count(const SplitPlanParams& p, cudaStream_t s);
void launch_split_mark(const SplitPlanParams& p, cudaStream_t s);
void launch_split_fill(const SplitPlanParams& p, cudaStream_t s);

// Forward merge of split rows: o = sum_i w_i part_o[i], lse = log sum_i 2^part_lse[i] (natural).
struct SplitMergeFwdParams {
  const int* merges;
  const int* info;
  int n_merge_cap;
  const float* part_o;    // [slots, 64, D]
  const float* part_lse;  // [slots, 64] log2 domain
  void* o;                // [BH, L, D]
  float* lse;             // [BH, L]
  const int* blk_cnt;
  int L, NB;
  int mask_rows;  // 1: rows >= blk_cnt get o = 0, lse = -inf (generic); 0: merged like any row (TK keeps
                  //    padded query rows finite, which its backward needs)
};
void launch_split_merge_fwd(const SplitMergeFwdParams& p, int D, DType dt, cudaStream_t s);

// Backward merge of split rows/columns: out[t] = valid ? sum_i part[slot0+i][t] + add[blk]/cnt : 0,
// summed in chunk order (deterministic). out is Elt (out_f32 == nullptr) or fp32.
struct SplitMergeSumParams {
  const int* merges;
  const int* info;
  int n_merge_cap;
  const float* part;  // [slots, 64, D]
  const float* add;   // optional [BH, add_nb, D]
  int add_nb;
  void* out;
  float* out_f32;
  const int* blk_cnt;
  int L, NB;
};
void launch_split_merge_sum(const SplitMergeSumParams& p, int D, DType dt, cudaStream_t s);

struct AttnBwdParams {
  const void* q;
  const void* k;
  const void* v;
  const void* dout;    // masked dO of this attention branch
  const float* lse;    // [BH, L] natural
  const float* delta;  // [BH, L] rowsum(dO * O)
  float* dq_acc;       // [BH, L, D] fp32, accumulated with atomics (must be zeroed), or nullptr:
                       // dQ not computed, attn_bwd_q supplies it
  bool dq_word;        // atomically add dQ one float at a time instead of float2 (diagnostic)
  void* dk;            // [BH, L, D] Elt output (used when dk_f32 == nullptr)
  void* dv;
  float* dk_f32;       // optional fp32 outputs instead of Elt
  float* dv_f32;
  const float* add_dk;  // optional [BH, add_nb, D] per-block gradients, added as add/cnt to
  const float* add_dv;  // every valid token of the block (coarse unpooling)
  int add_nb;
  const int* row_ptr;  // k2q
  const int* col_idx;
  const int* blk_cnt;
  int BH;
  int L;
  int NB;
  float scale;
  float scale_log2;
  // Split-KV task mode over k2q columns (see AttnFwdParams): slot >= 0 writes fp32 partial dK/dV to
  // part_dk/part_dv[slot] [64, D] with no coarse add (split_merge_sum adds it once).
  const int* tasks;
  const int* task_info;
  int n_tasks_cap;
  float* part_dk;
  float* part_dv;
};

void launch_attn_fwd(const AttnFwdParams& p, int D, DType dt, cudaStream_t s);
// generic forward variant: 1 = 4 warps x 16 rows (default), 2 = 2 warps x 32 rows
void set_fwd_variant(int mt);
int get_fwd_variant();
void launch_attn_bwd(const AttnBwdParams& p, int D, DType dt, cudaStream_t s);

// dQ from q->kv lists with no atomics; echosparse/vsa/dispatch.py chooses between this and atomic dQ per call.
struct AttnBwdQParams {
  const void* q;
  const void* k;
  const void* v;
  const void* dout;
  const float* lse;
  const float* delta;
  void* dq;             // [BH, L, D] Elt output
  const float* add_dq;  // optional [BH, add_nb, D] per-block gradient, added as add/cnt
  int add_nb;
  const int* row_ptr;   // q2k
  const int* col_idx;
  const int* blk_cnt;
  int BH;
  int L;
  int NB;
  float scale;
  float scale_log2;
  // Split-KV task mode over q2k rows: slot >= 0 writes fp32 partial dQ to part_dq[slot] [64, D].
  const int* tasks;
  const int* task_info;
  int n_tasks_cap;
  float* part_dq;
};
void launch_attn_bwd_q(const AttnBwdQParams& p, int D, DType dt, cudaStream_t s);

// ---------------------------------------------------------------- pooling
struct PoolParams {
  const void* q;
  const void* k;
  const void* v;
  void* qr;  // [BH, Lr_pad, D]  router pooled rows, row = blk*G + g
  void* kr;
  void* qc;  // [BH, NBc_pad, D] coarse block means
  void* kc;
  void* vc;
  void* vr;  // [BH, Lr_pad, D] V group means, same layout as kr; null when not requested
  const int* blk_cnt;
  int BH, L, NB, R, G, Lr_pad, NBc_pad;
};
void launch_pool(const PoolParams& p, int D, DType dt, cudaStream_t s);

// ---------------------------------------------------------------- router
struct RouterParams {
  const void* qr;  // [BH, Lr_pad, D]
  const void* kr;
  float* lse2;     // [BH, Lr_pad] log2-domain LSE per pooled query row
  float* scores;   // [bh_count, NB, NB] block scores for bh in [bh0, bh0+bh_count)
  const int* blk_cnt;
  int BH, NB, R, G, Lr, Lr_pad;
  int bh0, bh_count;
  int row_norm;  // 0: divide by #valid query groups of the block, 1: paper 1/G^2
  float scale_log2;
};
void launch_router_lse(const RouterParams& p, int D, DType dt, cudaStream_t s);
void launch_router_scores(const RouterParams& p, int D, DType dt, cudaStream_t s);

// ---------------------------------------------------------------- global top-k
struct TopkWorkspace {
  uint32_t* hist;     // [3, bh_count, 2048]  (zeroed by launcher)
  int* state;         // [bh_count, 8]
  uint32_t* gtbits;   // [bh_count, NB, W]
  uint32_t* eqbits;   // [bh_count, NB, W]
  uint32_t* selbits;  // [bh_count, NB, W]
  int* gtcnt;         // [bh_count, NB]
  int* eqcnt;         // [bh_count, NB]
  int* colcnt;        // [bh_count, NB]
};
size_t topk_workspace_bytes(int bh_count, int NB);
void topk_carve_workspace(void* base, int bh_count, int NB, TopkWorkspace* ws);

struct TopkParams {
  const float* scores;  // [bh_count, NB, NB]
  int NB;
  int bh0;
  int bh_count;
  long long budget;  // per (b,h); clamped to NB*NB
  int* q2k_ptr;      // [BH*(NB+1)] (rows bh0.. written)
  int* q2k_idx;      // [BH*budget]
  int* k2q_ptr;
  int* k2q_idx;
};
void launch_global_topk(const TopkParams& p, const TopkWorkspace& ws, cudaStream_t s);

// ---------------------------------------------------------------- original VSA router
// Whole-block coarse scores in the original's rounding: S = rnd(rnd(qc . kc) * scale), rnd = round to the
// token dtype, scale = 1/sqrt(D) in fp32 (Torch's bf16 `(q_c @ k_c^T) / sqrt(D)`: fp32-accumulated matmul
// rounded once, then multiplied by the fp32 reciprocal of the CPU scalar and rounded again).
struct BlockScoreParams {
  const void* qc;  // [BH, NBc_pad, D] coarse block means (pool_fwd)
  const void* kc;
  void* scores;    // [bh_count, NB, NB] token dtype, for bh in [bh0, bh0+bh_count)
  int NB, NBc_pad, bh0, bh_count;
  float scale;
};
void launch_block_scores(const BlockScoreParams& p, int D, DType dt, cudaStream_t s);

// Per-query-row top-K of 16-bit (bf16/fp16) scores: exactly K = min(topk, NB) kv blocks per row, larger
// score first, ties to the smaller kv index (-0 == +0). Fills q2k (ascending) and the inverted k2q lists.
struct RowTopkParams {
  const uint16_t* scores;  // [bh_count, NB, NB]
  int NB;
  int bh0;
  int bh_count;
  int K;
  int* q2k_ptr;  // [BH*(NB+1)], row r of (b,h) starts at ((b,h)*NB + r)*K
  int* q2k_idx;  // [BH*NB*K]
  int* k2q_ptr;
  int* k2q_idx;
};
void launch_row_topk(const RowTopkParams& p, const TopkWorkspace& ws, cudaStream_t s);

// ---------------------------------------------------------------- gate fusion
// gate_mode: 0 none (alpha=1, VSA ungated), 1 per-token scalar [BH, L], 2 per-element [BH, L, D]
// gate_act : 0 tanh (VSA2), 1 identity (original VSA compress_attn_weight)
struct FuseFwdParams {
  const void* of;  // [BH, L, D]
  const void* oc;  // [BH, NBc_pad, D]
  const void* gate;
  void* o;
  const int* blk_cnt;
  int BH, L, NB, NBc_pad, gate_mode, gate_act;
};
void launch_fuse_fwd(const FuseFwdParams& p, int D, DType dt, cudaStream_t s);

struct FuseBwdParams {
  const void* dout;  // [BH, L, D]
  const void* of;
  const void* oc;    // [BH, NBc_pad, D]
  const void* gate;
  void* dout_f;      // [BH, L, D]  masked dO for the fine branch, or nullptr (not written)
  float* delta_f;    // [BH, L]
  void* dout_c;      // [BH, NBc_pad, D] (rows >= NB untouched; caller zero-fills)
  float* delta_c;    // [BH, NBc_pad]
  void* dgate;       // like gate
  const int* blk_cnt;
  int BH, L, NB, NBc_pad, gate_mode, gate_act;
};
void launch_fuse_bwd(const FuseBwdParams& p, int D, DType dt, cudaStream_t s);

// dX = valid ? acc[t] + add[blk]/cnt : 0, cast to Elt. acc may be nullptr (treated as 0).
struct FinalizeParams {
  const float* acc;  // [BH, L, D] or nullptr
  const float* add;  // [BH, add_nb, D] or nullptr
  void* out;         // [BH, L, D]
  const int* blk_cnt;
  int BH, L, NB, add_nb;
};
void launch_finalize(const FinalizeParams& p, int D, DType dt, cudaStream_t s);

}  // namespace vsa2
