// VSA2 split-KV scheduling (experimental, opt-in): task planning on a CSR list and the merges.
//
// Plan (device only, no host readback), for NR = BH*NB rows of q2k (forward, Q-owner backward)
// or k2q (KV-owner backward):
//   count : elig[r]  = n_r >= thr_h (vsa2_kernels.h; manual SplitKV: n_r > chunk)
//           cnt[r]   = elig[r] ? ceil(n_r / chunk) : 0
//   scan  : cnt_incl, elig_incl = inclusive_scan                             (torch cumsum)
//   mark  : split[r] = E <= budget && cnt[r] > 0 && cnt_incl[r] <= max_slots   (E = #eligible)
//           flags    = [!split, split ? cnt : 0, split]
//   scan  : flags_incl                                                       (torch cumsum)
//   fill  : split chunk j of row r -> task (slot base + j) with slot = task index, kv range
//           [beg + j*n/c, beg + (j+1)*n/c); direct row r -> task T_split + (#direct rows before r)
// Split rows form a prefix of the rows with cnt > 0 (cnt_incl is monotone), so slots and split
// tasks are contiguous: tasks [0, T_split) are the split chunks (heavy first), then direct rows.
//
// Merges (one CTA per split row, 128 threads, chunks combined in order: deterministic):
//   forward : o = sum_i 2^(lse_i - M) o_i / sum_i 2^(lse_i - M), lse = (M + log2 sum) * ln 2
//             (padded query rows: o = 0, lse = -inf with mask_rows, else merged like the TK kernel's)
//   backward: out = valid ? sum_i part_i + add[blk]/cnt : 0
#include "common.cuh"
#include "vsa2_kernels.h"

namespace vsa2 {

namespace {

constexpr int kPlanThreads = 256;

// thr without the per-(b, h) term: max(min_len, ceil(tail_frac * W / resident))
__device__ __forceinline__ int split_global_thr(const SplitPlanParams& p, long long W) {
  int thr = p.min_len;
  if (p.tail_frac > 0.0) thr = max(thr, (int)ceil(p.tail_frac * (double)W / (double)p.resident));
  return thr;
}

__global__ void split_count_kernel(const SplitPlanParams p) {
  const int NR = p.BH * p.NB;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= NR) return;
  const int bh = r / p.NB, i = r - bh * p.NB;
  const int* hp = p.ptr + (size_t)bh * (p.NB + 1);
  const int n = hp[i + 1] - hp[i];
  const long long W = (long long)p.ptr[(size_t)p.BH * (p.NB + 1) - 1] - p.ptr[0];
  int thr = split_global_thr(p, W);
  if (p.rel_mean > 0.0) thr = max(thr, (int)ceil(p.rel_mean * (double)(hp[p.NB] - hp[0]) / (double)p.NB));
  const bool elig = n >= thr;
  p.cnt[r] = elig ? (n + p.chunk - 1) / p.chunk : 0;
  p.cnt[NR + r] = elig ? 1 : 0;
}

__global__ void split_mark_kernel(const SplitPlanParams p) {
  const int NR = p.BH * p.NB;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= NR) return;
  const int c = p.cnt[r];
  const bool split = p.cnt_incl[2 * NR - 1] <= p.budget && c > 0 && p.cnt_incl[r] <= p.max_slots;
  p.flags[r] = split ? 0 : 1;
  p.flags[NR + r] = split ? c : 0;
  p.flags[2 * NR + r] = split ? 1 : 0;
}

__global__ void split_fill_kernel(const SplitPlanParams p) {
  const int NR = p.BH * p.NB;
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= NR) return;
  const int t_split = p.flags_incl[2 * NR - 1];
  const int bh = r / p.NB, i = r - bh * p.NB;
  const int* rp = p.ptr + (size_t)bh * (p.NB + 1) + i;
  const int beg = rp[0], n = rp[1] - beg;
  int4* tasks = reinterpret_cast<int4*>(p.tasks);
  if (p.flags[2 * NR + r]) {
    const int c = p.flags[NR + r];
    const int base = p.flags_incl[NR + r] - c;
    for (int j = 0; j < c; ++j) {
      const int b = beg + (int)((long long)j * n / c);
      const int e = beg + (int)((long long)(j + 1) * n / c);
      tasks[base + j] = make_int4(r, b, e, base + j);
    }
    reinterpret_cast<int4*>(p.merges)[p.flags_incl[2 * NR + r] - 1] = make_int4(r, base, c, 0);
  } else {
    tasks[t_split + p.flags_incl[r] - 1] = make_int4(r, beg, beg + n, -1);
  }
  if (r == NR - 1) {
    p.info[0] = t_split + p.flags_incl[NR - 1];  // tasks
    p.info[1] = p.flags_incl[3 * NR - 1];        // split rows (merges)
    p.info[2] = t_split;                          // split slots
    p.info[3] = NR;
    const long long W = (long long)p.ptr[(size_t)p.BH * (p.NB + 1) - 1] - p.ptr[0];
    const int E = p.cnt_incl[2 * NR - 1];
    p.info[4] = E;                                // eligible rows
    p.info[5] = E <= p.budget ? 1 : 0;            // within the heavy-row budget
    p.info[6] = split_global_thr(p, W);
    p.info[7] = (int)W;
  }
}

inline int plan_blocks(const SplitPlanParams& p) { return (p.BH * p.NB + kPlanThreads - 1) / kPlanThreads; }

template <typename Elt>
__device__ __forceinline__ void store8(Elt* dst, const float* x) {
  using T = EltTraits<Elt>;
  uint4 u;
  u.x = T::pack2(x[0], x[1]);
  u.y = T::pack2(x[2], x[3]);
  u.z = T::pack2(x[4], x[5]);
  u.w = T::pack2(x[6], x[7]);
  *reinterpret_cast<uint4*>(dst) = u;
}

__device__ __forceinline__ void load8f(const float* src, float* x) {
  const float4 a = reinterpret_cast<const float4*>(src)[0], b = reinterpret_cast<const float4*>(src)[1];
  x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
}

}  // namespace

template <typename Elt, int D>
__global__ void __launch_bounds__(128) split_merge_fwd_kernel(const SplitMergeFwdParams p) {
  constexpr int CPR = D / 8, RSTEP = 128 / CPR, NPER = kBlock / RSTEP;
  if ((int)blockIdx.x >= p.info[1]) return;
  const int4 mg = reinterpret_cast<const int4*>(p.merges)[blockIdx.x];
  const int bh = mg.x / p.NB, qb = mg.x - bh * p.NB, slot0 = mg.y, c = mg.z;
  const int ch = threadIdx.x % CPR, r0 = threadIdx.x / CPR;
  const int qcnt = p.blk_cnt[qb];
#pragma unroll
  for (int it = 0; it < NPER; ++it) {
    const int row = r0 + it * RSTEP;
    const size_t tok = (size_t)bh * p.L + (size_t)qb * kBlock + row;
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    float lse = neg_inf();
    if (row < qcnt || !p.mask_rows) {
      float mx = neg_inf();
      for (int j = 0; j < c; ++j) mx = fmaxf(mx, p.part_lse[(size_t)(slot0 + j) * kBlock + row]);
      if (mx != neg_inf()) {
        float sum = 0.f;
        for (int j = 0; j < c; ++j) {
          const float w = exp2f(p.part_lse[(size_t)(slot0 + j) * kBlock + row] - mx);
          float x[8];
          load8f(p.part_o + ((size_t)(slot0 + j) * kBlock + row) * D + ch * 8, x);
          sum += w;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[e] += w * x[e];
        }
        const float inv = 1.f / sum;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] *= inv;
        lse = (mx + log2f(sum)) * kLn2;
      }
    }
    store8(reinterpret_cast<Elt*>(p.o) + tok * D + ch * 8, acc);
    if (ch == 0) p.lse[tok] = lse;
  }
}

template <typename Elt, int D>
__global__ void __launch_bounds__(128) split_merge_sum_kernel(const SplitMergeSumParams p) {
  constexpr int CPR = D / 8, RSTEP = 128 / CPR, NPER = kBlock / RSTEP;
  if ((int)blockIdx.x >= p.info[1]) return;
  const int4 mg = reinterpret_cast<const int4*>(p.merges)[blockIdx.x];
  const int bh = mg.x / p.NB, blk = mg.x - bh * p.NB, slot0 = mg.y, c = mg.z;
  const int ch = threadIdx.x % CPR, r0 = threadIdx.x / CPR;
  const int cnt = p.blk_cnt[blk];
  float add[8];
  if (p.add) {
    load8f(p.add + ((size_t)bh * p.add_nb + blk) * D + ch * 8, add);
    const float inv = 1.f / (float)cnt;
#pragma unroll
    for (int e = 0; e < 8; ++e) add[e] *= inv;
  } else {
#pragma unroll
    for (int e = 0; e < 8; ++e) add[e] = 0.f;
  }
#pragma unroll
  for (int it = 0; it < NPER; ++it) {
    const int row = r0 + it * RSTEP;
    const size_t tok = (size_t)bh * p.L + (size_t)blk * kBlock + row;
    float acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    if (row < cnt) {
      for (int j = 0; j < c; ++j) {
        float x[8];
        load8f(p.part + ((size_t)(slot0 + j) * kBlock + row) * D + ch * 8, x);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] += x[e];
      }
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[e] += add[e];
    }
    if (p.out_f32) {
      float4* dst = reinterpret_cast<float4*>(p.out_f32 + tok * D + ch * 8);
      dst[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      dst[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    } else {
      store8(reinterpret_cast<Elt*>(p.out) + tok * D + ch * 8, acc);
    }
  }
}

void launch_split_count(const SplitPlanParams& p, cudaStream_t s) {
  split_count_kernel<<<plan_blocks(p), kPlanThreads, 0, s>>>(p);
}
void launch_split_mark(const SplitPlanParams& p, cudaStream_t s) {
  split_mark_kernel<<<plan_blocks(p), kPlanThreads, 0, s>>>(p);
}
void launch_split_fill(const SplitPlanParams& p, cudaStream_t s) {
  split_fill_kernel<<<plan_blocks(p), kPlanThreads, 0, s>>>(p);
}

#define VSA2_SPLIT_DISPATCH(D, dt, KERNEL, ...)                            \
  do {                                                                     \
    if (dt == DType::kBF16) {                                              \
      if (D == 64) KERNEL<__nv_bfloat16, 64><<<__VA_ARGS__>>>(p);          \
      else KERNEL<__nv_bfloat16, 128><<<__VA_ARGS__>>>(p);                 \
    } else {                                                               \
      if (D == 64) KERNEL<__half, 64><<<__VA_ARGS__>>>(p);                 \
      else KERNEL<__half, 128><<<__VA_ARGS__>>>(p);                        \
    }                                                                      \
  } while (0)

void launch_split_merge_fwd(const SplitMergeFwdParams& p, int D, DType dt, cudaStream_t s) {
  if (p.n_merge_cap <= 0) return;
  VSA2_SPLIT_DISPATCH(D, dt, split_merge_fwd_kernel, p.n_merge_cap, 128, 0, s);
}
void launch_split_merge_sum(const SplitMergeSumParams& p, int D, DType dt, cudaStream_t s) {
  if (p.n_merge_cap <= 0) return;
  VSA2_SPLIT_DISPATCH(D, dt, split_merge_sum_kernel, p.n_merge_cap, 128, 0, s);
}

}  // namespace vsa2
