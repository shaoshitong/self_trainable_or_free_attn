// Three-tier representative attention backward, query-owner direction.
//
// One CTA = one (b*h, 64-row query block), 8 warps, walking the same column stream as the forward and
// recomputing P from the forward lse (no saved score matrix). Per tile, with bias_j = log2(c_j):
//
//   dP = dO V^T,  S = Q K^T,  P = exp2(scale_log2 * S + bias - log2e * lse),  dS = P * (dP - delta) * scale
//   dQ     += dS K                          (registers, all three tiers, written once)
//   dV_rep += P^T dO,  dK_rep += dS^T Q     (tiers B and C only, fp32 atomics into the rep tables)
//
// log2(c) is a constant of the tier assignment, so it contributes no gradient of its own; it only
// shapes P. Tier-A token dK/dV are not computed here -- vsa2::launch_attn_bwd over the tier-A k2q
// lists produces them exactly, and its unbiased `col < blk_cnt` masking is precisely this kernel's
// tier-A bias. Representative dK/dV are later spread over the tokens each mean averaged (rep_unpool).
//
// dQ is exact and deterministic (accumulated in registers, one store). The representative tables are
// reduced with fp32 atomics because one representative row is shared by many query blocks, so their
// summation order is nondeterministic -- the same tradeoff VSA2's own TK backward makes for dK/dV.
//
// Stage order follows VSA2's attn_bwd_q_kernel (next V load overlaps S and dQ, next K load overlaps
// the next dP); Q, dO, lse and delta belong to the CTA's own query block and are loaded once. The next
// tile's column metadata is produced by thread 0 during dP, so the barrier that waits for K publishes
// it without an extra sync.
#include "common.cuh"
#include "rep_common.cuh"
#include "rep_kernels.h"

namespace vsa2rep {

using vsa2::atomic_add_f2;
using vsa2::cp_async_commit;
using vsa2::cp_async_wait;
using vsa2::EltTraits;
using vsa2::fast_exp2;
using vsa2::kLog2e;
using vsa2::ldsm_x4;
using vsa2::ldsm_x4_t;
using vsa2::load_tile_async;

namespace {

// acc[4][4] = A B^T, warp tile: rows 16*wm of A, columns 32*wn = rows of B
template <typename Elt, int D>
__device__ __forceinline__ void mm_abt(float (&acc)[4][4], const Elt* sA, const Elt* sB, int wm, int wn, int lane) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) {
    uint32_t a[4];
    ldsm_x4(a[0], a[1], a[2], a[3], smem_u32(sA + swz<D>(16 * wm + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
    for (int np = 0; np < 2; ++np) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4(b0, b1, b2, b3,
              smem_u32(sB + swz<D>(32 * wn + np * 16 + (lane & 7) + ((lane >> 4) << 3),
                                   kk * 16 + ((lane >> 3) & 1) * 8)));
      T::mma(acc[2 * np], a, b0, b1);
      T::mma(acc[2 * np + 1], a, b2, b3);
    }
  }
}

// P and dS from S and dP into shared memory. Every column carries its log2-domain bias, which both
// masks absent columns (-inf) and gives a representative the mass of the c tokens it averages.
template <typename Elt, int D, bool STORE_P>
__device__ __forceinline__ void store_p_ds(float (&s)[4][4], float (&dp)[4][4], Elt* sP, Elt* sdS, const float* sL,
                                           const float* sDl, const float* bias, int kcA, float scale_log2,
                                           float scale, int wm, int wn, int g, int t) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = 16 * wm + g + 8 * r;
    const float lse = sL[row];
    const float dl = sDl[row];
    const bool rv = lse != neg_inf();
    const float lse2 = lse * kLog2e;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int col = 32 * wn + nt * 8 + 2 * t;
      float b0, b1;
      tile_bias2(kcA, bias, col, b0, b1);
      float pv[2], dsv[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float x = s[nt][2 * r + e] * scale_log2 + (e ? b1 : b0);
        pv[e] = rv ? fast_exp2(x - lse2) : 0.f;
        dsv[e] = pv[e] * (dp[nt][2 * r + e] - dl) * scale;
      }
      if constexpr (STORE_P) *reinterpret_cast<uint32_t*>(sP + swz<kBlock>(row, col)) = T::pack2(pv[0], pv[1]);
      *reinterpret_cast<uint32_t*>(sdS + swz<kBlock>(row, col)) = T::pack2(dsv[0], dsv[1]);
    }
  }
}

// acc = X^T B, warp tile: rows 16*wm of the result (columns of X), head-dim half wn
template <typename Elt, int D>
__device__ __forceinline__ void mm_xtb(float (&acc)[D / 16][4], const Elt* sX, const Elt* sB, int wm, int wn,
                                       int lane) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int i = 0; i < D / 16; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t a[4];
    ldsm_x4_t(a[0], a[1], a[2], a[3],
              smem_u32(sX + swz<kBlock>(kk * 16 + (lane & 7) + ((lane >> 4) << 3), 16 * wm + ((lane >> 3) & 1) * 8)));
#pragma unroll
    for (int dq2 = 0; dq2 < D / 32; ++dq2) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4_t(b0, b1, b2, b3,
                smem_u32(sB + swz<D>(kk * 16 + (lane & 15), wn * (D / 2) + dq2 * 16 + (lane >> 4) * 8)));
      T::mma(acc[2 * dq2], a, b0, b1);
      T::mma(acc[2 * dq2 + 1], a, b2, b3);
    }
  }
}

// acc += dS K, warp tile: rows 16*wm of dS, head-dim half wn
template <typename Elt, int D>
__device__ __forceinline__ void mma_dsk(float (&acc)[D / 16][4], const Elt* sdS, const Elt* sK, int wm, int wn,
                                        int lane) {
  using T = EltTraits<Elt>;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    uint32_t a[4];
    ldsm_x4(a[0], a[1], a[2], a[3], smem_u32(sdS + swz<kBlock>(16 * wm + (lane & 15), kk * 16 + (lane >> 4) * 8)));
#pragma unroll
    for (int dq2 = 0; dq2 < D / 32; ++dq2) {
      uint32_t b0, b1, b2, b3;
      ldsm_x4_t(b0, b1, b2, b3,
                smem_u32(sK + swz<D>(kk * 16 + (lane & 15), wn * (D / 2) + dq2 * 16 + (lane >> 4) * 8)));
      T::mma(acc[2 * dq2], a, b0, b1);
      T::mma(acc[2 * dq2 + 1], a, b2, b3);
    }
  }
}

}  // namespace

template <typename Elt, int D>
__global__ void __launch_bounds__(256) rep_attn_bwd_q_kernel(const RepAttnBwdQParams p) {
  using T = EltTraits<Elt>;
  constexpr int BM = kBlock, BN = kBlock, NT = 256;
  constexpr int NH = D / 16;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sQ = reinterpret_cast<Elt*>(smem_raw);
  Elt* sdO = sQ + BM * D;
  Elt* sK = sdO + BM * D;
  Elt* sV = sK + BN * D;
  Elt* sP = sV + BN * D;
  Elt* sdS = sP + BM * BN;
  float* sL = reinterpret_cast<float*>(sdS + BM * BN);
  float* sDl = sL + BM;
  int* sIdx = reinterpret_cast<int*>(sDl + BM);             // [2][64] gathered table rows
  float* sBias = reinterpret_cast<float*>(sIdx + 2 * BM);   // [2][64] log2 counts
  int* sCol = reinterpret_cast<int*>(sBias + 2 * BM);       // [2][64] packed kv block ids
  int* sMeta = sCol + 2 * BM;                               // [2][kMetaStride]

  const int qb = blockIdx.x, bh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t tok_base = (size_t)bh * p.L;
  const size_t q_base = tok_base + (size_t)qb * BM;
  const Elt* gK = reinterpret_cast<const Elt*>(p.k) + tok_base * D;
  const Elt* gV = reinterpret_cast<const Elt*>(p.v) + tok_base * D;
  const size_t mid_base = (size_t)bh * p.NB * p.RM * D;
  const size_t tail_base = (size_t)bh * p.NB * D;
  const Elt* tKm = reinterpret_cast<const Elt*>(p.k_mid) + mid_base;
  const Elt* tVm = reinterpret_cast<const Elt*>(p.v_mid) + mid_base;
  const Elt* tKt = reinterpret_cast<const Elt*>(p.k_tail) + tail_base;
  const Elt* tVt = reinterpret_cast<const Elt*>(p.v_tail) + tail_base;
  float* gdKm = p.dk_mid + mid_base;
  float* gdVm = p.dv_mid + mid_base;
  float* gdKt = p.dk_tail + tail_base;
  float* gdVt = p.dv_tail + tail_base;
  const int qcnt = p.blk_cnt[qb];
  const int wm = warp & 3, wn = warp >> 2;
  const float* cs_lc_mid = p.lc_mid + (size_t)bh * p.NB * p.RM;
  const float* cs_lc_tail = p.lc_tail + (size_t)bh * p.NB;

  ColStream cs;
  if (warp == 0) {  // every lane of warp 0 keeps the same stream state; the bit scan is lane-parallel
    const int* rp = p.row_ptr + (size_t)bh * (p.NB + 1) + qb;
    cs.a_idx = p.col_idx + rp[0];
    cs.row_a = p.bits_a + ((size_t)bh * p.NB + qb) * p.W;
    cs.row_ab = p.bits_ab + ((size_t)bh * p.NB + qb) * p.W;
    cs.blk_cnt = p.blk_cnt;
    cs.init(rp[1] - rp[0], p.NB, p.RM, p.phases);
    cs.next(sMeta, sCol, lane);
  }
  __syncthreads();

  float dq[NH][4];
#pragma unroll
  for (int i = 0; i < NH; ++i) dq[i][0] = dq[i][1] = dq[i][2] = dq[i][3] = 0.f;

  int tier = sMeta[kMetaTier], par = 0;
  if (tier != kTileA && tier != kTileNone) {  // first tile: nothing to overlap with yet
    int row;
    float bias;
    expand_tile_fetch(tier, sMeta[kMetaCnt], p.RM, sCol, cs_lc_mid, cs_lc_tail, tid, row, bias);
    expand_tile_publish(sIdx, sBias, tid, row, bias);
    __syncthreads();
  }
  if (tier != kTileNone) {
    auto issue_v = [&](int tr, int pr) {
      if (tr == kTileA) load_tile_async<BN, D, NT>(sV, gV + (size_t)sMeta[pr * kMetaStride + kMetaBlk] * BN * D, tid);
      else if (tr == kTileB) load_tile_gather_async<BN, D, NT>(sV, tVm, sIdx + pr * BM, tid);
      else load_tile_gather_async<BN, D, NT>(sV, tVt, sIdx + pr * BM, tid);
      cp_async_commit();
    };
    auto issue_k = [&](int tr, int pr) {
      if (tr == kTileA) load_tile_async<BN, D, NT>(sK, gK + (size_t)sMeta[pr * kMetaStride + kMetaBlk] * BN * D, tid);
      else if (tr == kTileB) load_tile_gather_async<BN, D, NT>(sK, tKm, sIdx + pr * BM, tid);
      else load_tile_gather_async<BN, D, NT>(sK, tKt, sIdx + pr * BM, tid);
      cp_async_commit();
    };

    load_tile_async<BM, D, NT>(sQ, reinterpret_cast<const Elt*>(p.q) + q_base * D, tid);
    load_tile_async<BM, D, NT>(sdO, reinterpret_cast<const Elt*>(p.dout) + q_base * D, tid);
    if (tid < 16) vsa2::cp_async_16(smem_u32(sL + tid * 4), p.lse + q_base + tid * 4);
    else if (tid < 32) vsa2::cp_async_16(smem_u32(sDl + (tid - 16) * 4), p.delta + q_base + (tid - 16) * 4);
    issue_v(tier, par);  // first group: Q, dO, lse, delta, V
    issue_k(tier, par);  // second group: K

    while (tier != kTileNone) {
      const int cur = par, nxt = par ^ 1;
      float s[4][4], dp[4][4];

      // The column stream only walks lists and bitmaps, so the next tile is decided before this
      // tile's data arrives and expanded between the two barriers this stage already has: gathered
      // tiers add no synchronization of their own.
      if (warp == 0) cs.next(sMeta + nxt * kMetaStride, sCol + nxt * BM, lane);
      cp_async_wait<1>();
      __syncthreads();  // this tile's V landed; warps finished dS K of the previous tile; next decided
      const int ntier = sMeta[nxt * kMetaStride + kMetaTier];
      const bool ngather = ntier != kTileA && ntier != kTileNone;  // uniform across the CTA
      int nrow = -1;
      float nbias = neg_inf();
      if (ngather)
        expand_tile_fetch(ntier, sMeta[nxt * kMetaStride + kMetaCnt], p.RM, sCol + nxt * BM, cs_lc_mid,
                          cs_lc_tail, tid, nrow, nbias);
      mm_abt<Elt, D>(dp, sdO, sV, wm, wn, lane);
      if (ngather) expand_tile_publish(sIdx + nxt * BM, sBias + nxt * BM, tid, nrow, nbias);

      cp_async_wait<0>();
      __syncthreads();  // this tile's K landed; V free; the next tile's columns are now visible
      if (ntier != kTileNone) issue_v(ntier, nxt);

      mm_abt<Elt, D>(s, sQ, sK, wm, wn, lane);
      const bool reps = (tier != kTileA) && p.diag < 2;
      const int kcA = (tier != kTileA) ? -1 : sMeta[cur * kMetaStride + kMetaCnt];
      if (reps) {
        store_p_ds<Elt, D, true>(s, dp, sP, sdS, sL, sDl, sBias + cur * BM, kcA, p.scale_log2, p.scale, wm, wn,
                                 g, t);
      } else {
        store_p_ds<Elt, D, false>(s, dp, nullptr, sdS, sL, sDl, sBias + cur * BM, kcA, p.scale_log2, p.scale, wm,
                                  wn, g, t);
      }
      __syncthreads();  // P and dS visible to every warp

      mma_dsk<Elt, D>(dq, sdS, sK, wm, wn, lane);
      if (reps) {
        float dkr[NH][4], dvr[NH][4];
        mm_xtb<Elt, D>(dvr, sP, sdO, wm, wn, lane);
        mm_xtb<Elt, D>(dkr, sdS, sQ, wm, wn, lane);
        float* dkt = (tier == kTileB) ? gdKm : gdKt;
        float* dvt = (tier == kTileB) ? gdVm : gdVt;
        const int* idx = sIdx + cur * BM;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          if (p.diag) break;
          const int row = 16 * wm + g + 8 * r;
          const int rr = idx[row];
          if (rr < 0) continue;
          const size_t off = (size_t)rr * D;
#pragma unroll
          for (int nd = 0; nd < NH; ++nd) {
            const int col = wn * (D / 2) + nd * 8 + 2 * t;
            atomic_add_f2(dkt + off + col, dkr[nd][2 * r], dkr[nd][2 * r + 1]);
            atomic_add_f2(dvt + off + col, dvr[nd][2 * r], dvr[nd][2 * r + 1]);
          }
        }
      }
      __syncthreads();  // K, P and dS free
      if (ntier != kTileNone) issue_k(ntier, nxt);
      tier = ntier;
      par = nxt;
    }
  }

  // ---- epilogue: dQ, written once (+ coarse unpooling add; zero on padded rows)
  const float inv_cnt = 1.f / (float)qcnt;
  const float* add = p.add_dq ? p.add_dq + ((size_t)bh * p.add_nb + qb) * D : nullptr;
  const float* acc = p.dq_acc_in ? p.dq_acc_in + q_base * D : nullptr;
  Elt* out = reinterpret_cast<Elt*>(p.dq) + q_base * D;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int row = 16 * wm + g + 8 * r;
    const bool valid = row < qcnt;
#pragma unroll
    for (int nd = 0; nd < NH; ++nd) {
      const int col = wn * (D / 2) + nd * 8 + 2 * t;
      float q0 = dq[nd][2 * r], q1 = dq[nd][2 * r + 1];
      if (acc) {
        const float2 a2 = *reinterpret_cast<const float2*>(acc + (size_t)row * D + col);
        q0 += a2.x;
        q1 += a2.y;
      }
      if (add) { q0 += add[col] * inv_cnt; q1 += add[col + 1] * inv_cnt; }
      if (!valid) q0 = q1 = 0.f;
      *reinterpret_cast<uint32_t*>(out + (size_t)row * D + col) = T::pack2(q0, q1);
    }
  }
}

template <typename Elt, int D>
static void run_bwd_q(const RepAttnBwdQParams& p, cudaStream_t s) {
  constexpr int smem = (4 * kBlock * D + 2 * kBlock * kBlock) * (int)sizeof(Elt) + 2 * kBlock * 4 +
                       3 * 2 * kBlock * 4 + 2 * kMetaStride * 4;
  auto kern = rep_attn_bwd_q_kernel<Elt, D>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(p.NB, p.BH);
  kern<<<grid, 256, smem, s>>>(p);
}

void launch_rep_attn_bwd_q(const RepAttnBwdQParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_bwd_q<__nv_bfloat16, 64>(p, s);
    else run_bwd_q<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_bwd_q<__half, 64>(p, s);
    else run_bwd_q<__half, 128>(p, s);
  }
}

}  // namespace vsa2rep
