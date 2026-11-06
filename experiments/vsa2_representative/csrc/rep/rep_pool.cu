// Representative construction, its adjoint, and the row-dot statistic the backward needs.
//
// rep_build : plain means of K and V over the contiguous token groups of each 64-token kv block --
//             RM groups of 64/RM tokens (the "mid" table) and the whole block (the "tail" table) --
//             together with log2 of each group's valid token count. No clustering and no reordering:
//             a group is exactly one of VSA2's own R-token router groups (R = 64/RM), so under the
//             block layout of layout.py the representatives are genuine spatio-temporal sub-cubes.
//             An empty group (past blk_cnt) gets a zero mean and log2(c) = -inf, which removes its
//             column from the softmax.
// rep_unpool: the adjoint. A representative is a mean over c tokens, so d rep / d x_i = 1 / c and a
//             representative's gradient is spread as drep / c over exactly the tokens it averaged.
//             Padded slots receive zero.
// row_dot   : delta[row] = sum_d dO[row, d] * O[row, d] over the fused output, 0 on padded rows.
#include "common.cuh"
#include "rep_kernels.h"

namespace vsa2rep {

using vsa2::EltTraits;
using vsa2::neg_inf;

template <typename Elt, int D>
__global__ void rep_build_kernel(const RepBuildParams p) {
  using T = EltTraits<Elt>;
  const int blk = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
  const int RM = p.RM, GS = kBlock / RM;
  const int cnt = p.blk_cnt[blk];
  const size_t base = ((size_t)bh * p.L + (size_t)blk * kBlock) * D + d;
  const Elt* gk = reinterpret_cast<const Elt*>(p.k) + base;
  const Elt* gv = reinterpret_cast<const Elt*>(p.v) + base;
  Elt* mk = reinterpret_cast<Elt*>(p.k_mid) + ((size_t)bh * p.NB + blk) * RM * D + d;
  Elt* mv = reinterpret_cast<Elt*>(p.v_mid) + ((size_t)bh * p.NB + blk) * RM * D + d;
  float* lcm = p.lc_mid + ((size_t)bh * p.NB + blk) * RM;

  float tk = 0.f, tv = 0.f;
  for (int g = 0; g < RM; ++g) {
    const int n = min(max(cnt - g * GS, 0), GS);
    float ak = 0.f, av = 0.f;
    for (int s = 0; s < n; ++s) {
      const size_t off = (size_t)(g * GS + s) * D;
      ak += T::to_f(gk[off]);
      av += T::to_f(gv[off]);
    }
    tk += ak;
    tv += av;
    const float inv = n > 0 ? 1.f / (float)n : 0.f;
    mk[(size_t)g * D] = T::from_f(ak * inv);
    mv[(size_t)g * D] = T::from_f(av * inv);
    if (d == 0) lcm[g] = n > 0 ? __log2f((float)n) : neg_inf();
  }
  const float invc = cnt > 0 ? 1.f / (float)cnt : 0.f;
  const size_t toff = ((size_t)bh * p.NB + blk) * D + d;
  reinterpret_cast<Elt*>(p.k_tail)[toff] = T::from_f(tk * invc);
  reinterpret_cast<Elt*>(p.v_tail)[toff] = T::from_f(tv * invc);
  if (d == 0) p.lc_tail[(size_t)bh * p.NB + blk] = cnt > 0 ? __log2f((float)cnt) : neg_inf();
}

template <typename Elt, int D>
__global__ void __launch_bounds__(256) rep_unpool_kernel(const RepUnpoolParams p) {
  using T = EltTraits<Elt>;
  const int blk = blockIdx.x, bh = blockIdx.y;
  const int RM = p.RM, GS = kBlock / RM;
  const int cnt = p.blk_cnt[blk];
  const size_t tok = ((size_t)bh * p.L + (size_t)blk * kBlock) * D;
  const size_t mid = ((size_t)bh * p.NB + blk) * RM * D;
  const size_t tail = ((size_t)bh * p.NBt + blk) * D;
  const float invc = cnt > 0 ? 1.f / (float)cnt : 0.f;
#pragma unroll
  for (int j = 0; j < kBlock * D / 256; ++j) {
    const int i = (int)threadIdx.x + j * 256;
    const int row = i / D, d = i - row * D;
    const int gi = row / GS;
    const int n = min(max(cnt - gi * GS, 0), GS);
    const float invg = n > 0 ? 1.f / (float)n : 0.f;
    float gk = 0.f, gv = 0.f;
    if (row < cnt) {
      if (p.acc_k) {
        gk = p.acc_k[tok + i];
        gv = p.acc_v[tok + i];
      }
      gk += p.dk_mid[mid + (size_t)gi * D + d] * invg + p.dk_tail[tail + d] * invc;
      gv += p.dv_mid[mid + (size_t)gi * D + d] * invg + p.dv_tail[tail + d] * invc;
    }
    reinterpret_cast<Elt*>(p.dk)[tok + i] = T::from_f(gk);
    reinterpret_cast<Elt*>(p.dv)[tok + i] = T::from_f(gv);
  }
}

template <typename Elt, int D>
__global__ void __launch_bounds__(256) row_dot_kernel(const RowDotParams p) {
  using T = EltTraits<Elt>;
  constexpr int PER = D / 32;
  const int blk = blockIdx.x, bh = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cnt = p.blk_cnt[blk];
  const Elt* ga = reinterpret_cast<const Elt*>(p.a);
  const Elt* gb = reinterpret_cast<const Elt*>(p.b);
  for (int row = warp; row < kBlock; row += 8) {
    const size_t off = ((size_t)bh * p.L + (size_t)blk * kBlock + row) * D;
    float acc = 0.f;
    if (row < cnt) {
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int d = lane + i * 32;
        acc += T::to_f(ga[off + d]) * T::to_f(gb[off + d]);
      }
#pragma unroll
      for (int s = 16; s; s >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, s);
    }
    if (lane == 0) p.out[(size_t)bh * p.L + (size_t)blk * kBlock + row] = acc;
  }
}

__global__ void __launch_bounds__(256) tier_b_lists_kernel(const TierBListParams p) {
  const int bh = blockIdx.y;
  const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= p.NB) return;
  const size_t r = (size_t)bh * (p.NB + 1) + row;
  const int ab0 = p.ab_ptr[r], ab1 = p.ab_ptr[r + 1];
  const int b0 = ab0 - p.a_ptr[r];
  if (lane == 0) {
    p.b_ptr[r] = b0;
    if (row == p.NB - 1) p.b_ptr[r + 1] = ab1 - p.a_ptr[r + 1];
  }
  const uint32_t* bits = p.bits_a + ((size_t)bh * p.NB + row) * p.W;
  int out = b0;
  for (int base = ab0; base < ab1; base += 32) {
    const int i = base + lane;
    int col = -1;
    bool keep = false;
    if (i < ab1) {
      col = p.ab_idx[i];
      keep = !((bits[col >> 5] >> (col & 31)) & 1u);
    }
    const unsigned m = __ballot_sync(0xffffffffu, keep);
    if (keep) p.b_idx[out + __popc(m & ((1u << lane) - 1u))] = col;
    out += __popc(m);
  }
}

void launch_tier_b_lists(const TierBListParams& p, cudaStream_t s) {
  tier_b_lists_kernel<<<dim3((p.NB + 7) / 8, p.BH), 256, 0, s>>>(p);
}

template <typename Elt, int D>
static void run_build(const RepBuildParams& p, cudaStream_t s) {
  rep_build_kernel<Elt, D><<<dim3(p.NB, p.BH), D, 0, s>>>(p);
}

void launch_rep_build(const RepBuildParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_build<__nv_bfloat16, 64>(p, s);
    else run_build<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_build<__half, 64>(p, s);
    else run_build<__half, 128>(p, s);
  }
}

void launch_rep_unpool(const RepUnpoolParams& p, int D, DType dt, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  if (dt == DType::kBF16) {
    if (D == 64) rep_unpool_kernel<__nv_bfloat16, 64><<<grid, 256, 0, s>>>(p);
    else rep_unpool_kernel<__nv_bfloat16, 128><<<grid, 256, 0, s>>>(p);
  } else {
    if (D == 64) rep_unpool_kernel<__half, 64><<<grid, 256, 0, s>>>(p);
    else rep_unpool_kernel<__half, 128><<<grid, 256, 0, s>>>(p);
  }
}

void launch_row_dot(const RowDotParams& p, int D, DType dt, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  if (dt == DType::kBF16) {
    if (D == 64) row_dot_kernel<__nv_bfloat16, 64><<<grid, 256, 0, s>>>(p);
    else row_dot_kernel<__nv_bfloat16, 128><<<grid, 256, 0, s>>>(p);
  } else {
    if (D == 64) row_dot_kernel<__half, 64><<<grid, 256, 0, s>>>(p);
    else row_dot_kernel<__half, 128><<<grid, 256, 0, s>>>(p);
  }
}

}  // namespace vsa2rep
