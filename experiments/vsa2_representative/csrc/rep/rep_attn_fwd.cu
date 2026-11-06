// Three-tier representative attention forward.
//
// One CTA = one (b*h, 64-row query block), 4 warps x 16 rows, mma.sync.m16n8k16 with Q fragments held
// in registers: the MMA and online-softmax body is VSA2's generic attn_fwd_kernel (MT = 1 variant),
// with one change -- every kv column carries an additive log2-domain bias from shared memory instead
// of a plain `col < blk_cnt` mask. The bias is 0 for a real token, log2(c) for a representative of c
// tokens, and -inf for a masked or absent column, so one online softmax spans all three tiers and
// normalizes over their union. No intermediate lse is stored or reused.
//
// Column stream per CTA (see rep_common.cuh): the tier-A CSR list one 64-token block per tile, then
// tier B packing 64/RM blocks x RM representatives per tile, then tier C packing 64 block means per
// tile. Tier A loads a contiguous [64, D] tile; tiers B and C gather 64 table rows.
//
// Pipelining keeps VSA2's single-buffer FlashAttention-2 order (V_j load overlaps S_j = Q K_j^T,
// K_{j+1} load overlaps softmax and P V_j). The column metadata of tile j+1 is produced by thread 0
// right after the softmax of tile j, so the existing __syncthreads() that waits for V_j publishes it:
// the gathered tiers cost no extra barrier. Metadata buffers are double-buffered by parity, so the
// producer never touches the tile being consumed.
#include "common.cuh"
#include "rep_common.cuh"
#include "rep_kernels.h"

namespace vsa2rep {

using vsa2::cp_async_commit;
using vsa2::cp_async_wait;
using vsa2::EltTraits;
using vsa2::fast_exp2;
using vsa2::kLn2;
using vsa2::ldsm_x4;
using vsa2::ldsm_x4_t;
using vsa2::load_tile_async;
using vsa2::warp_quad_max;
using vsa2::warp_quad_sum;

template <typename Elt, int D>
__global__ void __launch_bounds__(128) rep_attn_fwd_kernel(const RepAttnFwdParams p) {
  using T = EltTraits<Elt>;
  constexpr int BM = kBlock, BN = kBlock, NT = 128;
  constexpr int KD = D / 16;
  constexpr int ND = D / 8;
  extern __shared__ __align__(128) uint8_t smem_raw[];
  Elt* sQ = reinterpret_cast<Elt*>(smem_raw);
  Elt* sK = sQ + BM * D;
  Elt* sV = sK + BN * D;
  int* sIdx = reinterpret_cast<int*>(sV + BN * D);           // [2][64] gathered table rows
  float* sBias = reinterpret_cast<float*>(sIdx + 2 * BM);    // [2][64] log2 counts
  int* sCol = reinterpret_cast<int*>(sBias + 2 * BM);        // [2][64] packed kv block ids
  int* sMeta = sCol + 2 * BM;                                // [2][kMetaStride]

  const int qb = blockIdx.x, bh = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t tok_base = (size_t)bh * p.L;
  const Elt* gQ = reinterpret_cast<const Elt*>(p.q) + (tok_base + (size_t)qb * BM) * D;
  const Elt* gK = reinterpret_cast<const Elt*>(p.k) + tok_base * D;
  const Elt* gV = reinterpret_cast<const Elt*>(p.v) + tok_base * D;
  const size_t mid_base = (size_t)bh * p.NB * p.RM * D;
  const size_t tail_base = (size_t)bh * p.NB * D;
  const Elt* tKm = reinterpret_cast<const Elt*>(p.k_mid) + mid_base;
  const Elt* tVm = reinterpret_cast<const Elt*>(p.v_mid) + mid_base;
  const Elt* tKt = reinterpret_cast<const Elt*>(p.k_tail) + tail_base;
  const Elt* tVt = reinterpret_cast<const Elt*>(p.v_tail) + tail_base;
  Elt* gO = reinterpret_cast<Elt*>(p.o) + (tok_base + (size_t)qb * BM) * D;
  float* gL = p.lse + tok_base + (size_t)qb * BM;
  const int qcnt = p.blk_cnt[qb];
  const float* cs_lc_mid = p.lc_mid + (size_t)bh * p.NB * p.RM;
  const float* cs_lc_tail = p.lc_tail + (size_t)bh * p.NB;

  ColStream cs;
  if (warp == 0) {  // every lane of warp 0 keeps the same stream state; the bit scan is lane-parallel
    const int* rp = p.row_ptr + (size_t)bh * (p.NB + 1) + qb;
    cs.a_idx = p.col_idx + rp[0];
    cs.row_a = p.bits_a + ((size_t)bh * p.NB + qb) * p.W;
    cs.row_ab = p.bits_ab + ((size_t)bh * p.NB + qb) * p.W;
    cs.blk_cnt = p.blk_cnt;
    cs.init(rp[1] - rp[0], p.NB, p.RM, p.phases, p.diag);
    cs.next(sMeta, sCol, lane);
  }
  __syncthreads();

  int tier = sMeta[kMetaTier], par = 0;
  if (tier != kTileA && tier != kTileNone) {  // first tile: nothing to overlap with yet
    int row = tid < kBlock ? tid : -1;
    float bias = 0.f;
    if (p.diag < 2)
      expand_tile_fetch(tier, sMeta[kMetaCnt], p.RM, sCol, cs_lc_mid, cs_lc_tail, tid, row, bias);
    expand_tile_publish(sIdx, sBias, tid, row, bias);
    __syncthreads();
  }
  if (tier == kTileNone) {  // no column in any tier: o = 0, lse = -inf
    constexpr int CH = BM * D / 8;
    for (int i = tid; i < CH; i += NT) reinterpret_cast<uint4*>(gO)[i] = make_uint4(0, 0, 0, 0);
    for (int i = tid; i < BM; i += NT) gL[i] = neg_inf();
    return;
  }

  auto issue_k = [&](int tr, int pr) {
    if (tr == kTileA) {
      load_tile_async<BN, D, NT>(sK, gK + (size_t)sMeta[pr * kMetaStride + kMetaBlk] * BN * D, tid);
    } else if (p.diag >= 3) {  // diagnostic: contiguous loader, no per-chunk index read
      load_tile_async<BN, D, NT>(sK, (tr == kTileB) ? tKm : tKt, tid);
    } else if (tr == kTileB) {
      load_tile_gather_async<BN, D, NT>(sK, tKm, sIdx + pr * BM, tid);
    } else {
      load_tile_gather_async<BN, D, NT>(sK, tKt, sIdx + pr * BM, tid);
    }
  };
  auto issue_v = [&](int tr, int pr) {
    if (tr == kTileA) {
      load_tile_async<BN, D, NT>(sV, gV + (size_t)sMeta[pr * kMetaStride + kMetaBlk] * BN * D, tid);
    } else if (p.diag >= 3) {
      load_tile_async<BN, D, NT>(sV, (tr == kTileB) ? tVm : tVt, tid);
    } else if (tr == kTileB) {
      load_tile_gather_async<BN, D, NT>(sV, tVm, sIdx + pr * BM, tid);
    } else {
      load_tile_gather_async<BN, D, NT>(sV, tVt, sIdx + pr * BM, tid);
    }
  };

  load_tile_async<BM, D, NT>(sQ, gQ, tid);
  issue_k(tier, par);
  cp_async_commit();

  float o_acc[ND][4];
#pragma unroll
  for (int i = 0; i < ND; ++i) o_acc[i][0] = o_acc[i][1] = o_acc[i][2] = o_acc[i][3] = 0.f;
  float m_r[2] = {neg_inf(), neg_inf()}, l_r[2] = {0.f, 0.f};
  uint32_t qf[KD][4];
  const int r0 = warp * 16;
  bool first = true;

  while (tier != kTileNone) {
    const int cur = par, nxt = par ^ 1;
    // The column stream only walks lists and bitmaps, so the next tile can be decided before this
    // tile's data has even arrived; the barrier below then publishes it, and its expansion lands on
    // the barrier after the softmax. Gathered tiers therefore cost no extra synchronization.
    if (warp == 0) cs.next(sMeta + nxt * kMetaStride, sCol + nxt * BM, lane);
    cp_async_wait<0>();
    __syncthreads();  // K of this tile (and Q on the first pass) landed; every warp finished P V
    if (first) {
#pragma unroll
      for (int kk = 0; kk < KD; ++kk)
        ldsm_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                smem_u32(sQ + swz<D>(r0 + (lane & 15), kk * 16 + (lane >> 4) * 8)));
      first = false;
    }
    issue_v(tier, cur);
    cp_async_commit();

    const int ntier = sMeta[nxt * kMetaStride + kMetaTier];
    const bool ngather = ntier != kTileA && ntier != kTileNone;  // uniform across the CTA
    int nrow = -1;
    float nbias = neg_inf();
    if (ngather) {
      if (p.diag >= 2) {  // diagnostic: contiguous rows, no scattered gather and no expand
        nrow = tid < kBlock ? tid : -1;
        nbias = 0.f;
      } else {
        expand_tile_fetch(ntier, sMeta[nxt * kMetaStride + kMetaCnt], p.RM, sCol + nxt * BM, cs_lc_mid,
                          cs_lc_tail, tid, nrow, nbias);
      }
    }

    float s[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < KD; ++kk) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3,
                smem_u32(sK + swz<D>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 16 + ((lane >> 3) & 1) * 8)));
        T::mma(s[2 * np], qf[kk], b0, b1);
        T::mma(s[2 * np + 1], qf[kk], b2, b3);
      }
    }

    // online softmax over this tile's 64 columns, each biased by log2(represented tokens)
    const float* bias = sBias + cur * BM;
    // kcA >= 0 means "mask from a token count, no bias array"; diag >= 1 forces that for every tier
    const int kcA = (tier == kTileA) ? sMeta[cur * kMetaStride + kMetaCnt] : (p.diag >= 1 ? kBlock : -1);
    float mx[2] = {neg_inf(), neg_inf()};
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      float b0, b1;
      tile_bias2(kcA, bias, nt * 8 + 2 * t, b0, b1);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = s[nt][e] * p.scale_log2 + ((e & 1) ? b1 : b0);
        s[nt][e] = x;
        mx[e >> 1] = fmaxf(mx[e >> 1], x);
      }
    }
    float mu[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float mrow = warp_quad_max(mx[r]);
      const float m_new = fmaxf(m_r[r], mrow);
      const float m_use = (m_new == neg_inf()) ? 0.f : m_new;
      const float alpha = fast_exp2(m_r[r] - m_use);
      m_r[r] = m_new;
      mu[r] = m_use;
      l_r[r] *= alpha;
#pragma unroll
      for (int nd = 0; nd < ND; ++nd) {
        o_acc[nd][2 * r] *= alpha;
        o_acc[nd][2 * r + 1] *= alpha;
      }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float pv = fast_exp2(s[nt][e] - mu[e >> 1]);
        s[nt][e] = pv;
        l_r[e >> 1] += pv;
      }
    }

    if (ngather) expand_tile_publish(sIdx + nxt * BM, sBias + nxt * BM, tid, nrow, nbias);

    cp_async_wait<0>();
    __syncthreads();  // V landed; warps finished reading K; the next tile's columns are now visible
    if (ntier != kTileNone) {
      issue_k(ntier, nxt);
      cp_async_commit();
    }

    uint32_t pf[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      pf[kk][0] = T::pack2(s[2 * kk][0], s[2 * kk][1]);
      pf[kk][1] = T::pack2(s[2 * kk][2], s[2 * kk][3]);
      pf[kk][2] = T::pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pf[kk][3] = T::pack2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
      for (int dp = 0; dp < D / 16; ++dp) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4_t(b0, b1, b2, b3, smem_u32(sV + swz<D>(kk * 16 + (lane & 15), dp * 16 + (lane >> 4) * 8)));
        T::mma(o_acc[2 * dp], pf[kk], b0, b1);
        T::mma(o_acc[2 * dp + 1], pf[kk], b2, b3);
      }
    }
    tier = ntier;
    par = nxt;
  }

  // Epilogue: normalize, stage through this warp's rows of sQ, coalesced store.
  __syncwarp();
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float lsum = warp_quad_sum(l_r[r]);
    const int row = r0 + g + 8 * r;
    const bool valid = row < qcnt && lsum > 0.f;
    const float inv = valid ? 1.f / lsum : 0.f;
#pragma unroll
    for (int nd = 0; nd < ND; ++nd) {
      const int col = nd * 8 + 2 * t;
      *reinterpret_cast<uint32_t*>(sQ + swz<D>(row, col)) = T::pack2(o_acc[nd][2 * r] * inv, o_acc[nd][2 * r + 1] * inv);
    }
    if (t == 0) gL[row] = valid ? (m_r[r] + __log2f(lsum)) * kLn2 : neg_inf();
  }
  __syncwarp();
  constexpr int CPR = D / 8;
#pragma unroll
  for (int i = lane; i < 16 * CPR; i += 32) {
    const int row = r0 + i / CPR;
    const int ch = i % CPR;
    *reinterpret_cast<uint4*>(gO + (size_t)row * D + ch * 8) =
        *reinterpret_cast<const uint4*>(sQ + swz<D>(row, ch * 8));
  }
}

template <typename Elt, int D>
static void run_fwd(const RepAttnFwdParams& p, cudaStream_t s) {
  constexpr int smem = 3 * kBlock * D * (int)sizeof(Elt) + 3 * 2 * kBlock * 4 + 2 * kMetaStride * 4;
  auto kern = rep_attn_fwd_kernel<Elt, D>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(p.NB, p.BH);
  kern<<<grid, 128, smem, s>>>(p);
}

void launch_rep_attn_fwd(const RepAttnFwdParams& p, int D, DType dt, cudaStream_t s) {
  if (dt == DType::kBF16) {
    if (D == 64) run_fwd<__nv_bfloat16, 64>(p, s);
    else run_fwd<__nv_bfloat16, 128>(p, s);
  } else {
    if (D == 64) run_fwd<__half, 64>(p, s);
    else run_fwd<__half, 128>(p, s);
  }
}

}  // namespace vsa2rep
