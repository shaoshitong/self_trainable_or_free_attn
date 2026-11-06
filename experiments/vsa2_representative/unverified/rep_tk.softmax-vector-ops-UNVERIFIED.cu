// VSA2 three-tier representative attention, H100 (sm_90a) ThunderKittens wgmma + TMA kernels.
//
// Same semantics as the generic CUDA path (csrc/rep): tier A = the native fine block-sparse attention
// (64 tokens per selected kv block), tier B = RM representatives per kv block, tier C = one block mean
// per kv block, every representative of c tokens entering the one online softmax with + log2(c) on its
// log2-domain logit. What changes is how each tier's tiles reach shared memory:
//
//   tier A  contiguous [64, D] token tiles: tensor TMA, exactly the native VSA2 TK kernel's loads.
//   tier C  densely masked over the contiguous block-mean table: tile t is block means 64t..64t+63, one
//           tensor-TMA load; a column already in A|B gets -inf from the A|B bitmap. That trades a few
//           masked columns (NB/64 tiles instead of ceil(#C/64)) for loads TMA can express natively, and
//           a tile whose 64 blocks are all in A|B is skipped outright.
//   tier B  scattered: 64/RM selected blocks x RM representatives per tile. A tensor-TMA box cannot
//           gather rows, so the mid table is stored in global memory *already in TK's st_bf<64, D>
//           byte layout* (128-byte swizzle, chunk ^= row % 8, one copy per swizzle phase a slot can
//           land on), and each block's RM rows are moved with 1-D cp.async.bulk -- the TMA engine,
//           completing on the tile's mbarrier -- straight into wgmma-ready shared memory. No thread
//           touches the data; empty slots copy a zero block.
//
// Tier-B lists come precomputed from routing (CSR, see launch_tier_b_lists), so no kernel here scans a
// bitmap except for tier C's mask, which is two words per tile.
//
// Backward: tier A is VSA2's own TK backward on the tier-A k2q lists (unmodified; bias 0). Tier C runs
// KV-owner like that kernel -- the CTA owns 64 block means, accumulates their dK/dV in registers and
// reduces dQ with tensor TMA. Tier B runs query-owner: dQ in registers, representative dK/dV reduced
// into the fp32 mid-gradient table with 1-D bulk reduce-add (cp.reduce.async.bulk .add.f32), one per
// slot, instead of per-element atomics. All tiers share one fp32 dQ accumulator.
#include "kittens.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdlib>

using namespace kittens;

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define REP_TK_HOPPER 1
#else
#define REP_TK_HOPPER 0
#endif

namespace rep_tk {

constexpr int kB = 64;  // block / tile rows
enum { kTierA = 0, kTierB = 1, kTierC = 2, kTierNone = -1 };
enum { kPhaseA = 1, kPhaseB = 2, kPhaseC = 4 };
constexpr float kLog2e = 1.44269504089f;
constexpr float kLn2 = 0.69314718056f;

__device__ __forceinline__ float ninf() { return __uint_as_float(0xff800000u); }

// ---------------------------------------------------------------- raw TMA-engine helpers
__device__ __forceinline__ void bulk_load(uint32_t dst_smem, const void* src, uint32_t bytes, semaphore& bar) {
  const uint32_t mb = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(
                   dst_smem),
               "l"(src), "r"(bytes), "r"(mb)
               : "memory");
}
__device__ __forceinline__ void bulk_reduce_add_f32(void* dst, uint32_t src_smem, uint32_t bytes) {
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n" ::"l"(dst),
               "r"(src_smem), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait_read0() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
__device__ __forceinline__ void bulk_wait0() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// Number of distinct swizzle phases a tier-B slot can start on: slot s begins at tile row RM*s, and the
// 128-byte swizzle repeats every 8 rows.
__host__ __device__ constexpr int num_phases(int RM) { return RM >= 8 ? 1 : 8 / RM; }

// ---------------------------------------------------------------- the per-thread tile stream
// Every thread walks the same deterministic sequence -- tier A list, tier-B tiles, unskipped tier-C
// tiles -- so no shared metadata or barrier is needed to agree on it.
struct Stream {
  const int* a_idx;
  const int* b_idx;
  const uint32_t* ab_row;  // bits_ab of this query block
  int na, nb_list, nbt_tiles, nc_tiles, NB, W, G, phases;
  int ia, ib, tc;

  // C columns of tail tile t: blocks 64t..64t+63 that exist and are not in A|B.
  __device__ __forceinline__ uint2 cmask(int t) const {
    const int w0 = 2 * t, w1 = 2 * t + 1;
    uint32_t lo = w0 < W ? ~ab_row[w0] : 0u, hi = w1 < W ? ~ab_row[w1] : 0u;
    const int n0 = NB - 64 * t;  // real blocks in this tile
    if (n0 < 32) lo &= (n0 <= 0) ? 0u : ((1u << n0) - 1u);
    if (n0 < 64) hi &= (n0 <= 32) ? 0u : ((n0 - 32 >= 32) ? 0xffffffffu : ((1u << (n0 - 32)) - 1u));
    return make_uint2(lo, hi);
  }

  // Returns the next tile as (tier, arg): A -> kv block, B -> tile index into the B list, C -> tail tile.
  __device__ __forceinline__ int2 next() {
    if ((phases & kPhaseA) && ia < na) return make_int2(kTierA, a_idx[ia++]);
    if ((phases & kPhaseB) && ib < nbt_tiles) return make_int2(kTierB, ib++);
    if (phases & kPhaseC) {
      while (tc < nc_tiles) {
        const int t = tc++;
        const uint2 m = cmask(t);
        if (m.x | m.y) return make_int2(kTierC, t);
      }
    }
    return make_int2(kTierNone, 0);
  }
};

// Per-thread log2 biases of the 16 columns this thread holds in an rt_fl<16, 64> tile: for 16-column
// subtile i, columns 16i + 2t + {0, 1, 8, 9} (t = lane % 4) -> v[i][0..3]. `masked` is false for a full
// tier-A tile, which needs no bias at all.
//
// Tier B reads lcb, the log2 counts flattened per tier-B list entry (lcb[e * RM + g] = lc_mid[b_idx[e] *
// RM + g], built once per forward), so a tile's 64 biases are one contiguous run instead of a b_idx ->
// lc_mid dependent pair: measured 27% of the tier-B forward when it sat on the critical path.
struct TileBias {
  float v[4][4];  // tiers B and C only
  int kc;         // tier A: valid tokens of the kv block (the mask is formed inline, only when < 64)
  bool masked;
};

__device__ __forceinline__ TileBias tile_bias(int2 tile, const Stream& s, const int* blk_cnt, const float* lcb,
                                              const float* lc_tail, int RM) {
  TileBias tb;
  const int t4 = laneid() & 3;
  tb.masked = true;
  tb.kc = kB;
  if (tile.x == kTierA) {
    tb.kc = blk_cnt[tile.y];
    tb.masked = false;  // a partial block is masked from kc in the softmax, like VSA2's right_fill
  } else if (tile.x == kTierB) {
    const int G = 64 / RM;
    const int first = tile.y * G;
    const int ncol = min(G, s.nb_list - first) * RM;  // real columns of this tile
    const float* l = lcb + (size_t)first * RM;
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = 16 * i + 2 * t4 + (j & 1) + 8 * (j >> 1);
        tb.v[i][j] = c < ncol ? l[c] : ninf();
      }
  } else if (tile.x == kTierC) {
    const uint2 m = s.cmask(tile.y);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = 16 * i + 2 * t4 + (j & 1) + 8 * (j >> 1);
        const bool on = ((c < 32 ? m.x >> c : m.y >> (c - 32)) & 1u) != 0u;
        tb.v[i][j] = on ? lc_tail[64 * tile.y + c] : ninf();
      }
  } else {
    tb.masked = false;
  }
  return tb;
}

// Flattened tier-B log2 counts: lcb[e * RM + g] for every absolute tier-B entry e. One CTA per (row, bh),
// walking that row's CSR range, so heads need not have equal tier-B counts (hand-made selections).
__global__ void lcb_kernel(const float* __restrict__ lc_mid, const int* __restrict__ b_ptr,
                           const int* __restrict__ b_idx, float* __restrict__ lcb, int NB, int RM) {
  const int row = blockIdx.x, bh = blockIdx.y;
  const size_t r = (size_t)bh * (NB + 1) + row;
  const int e0 = b_ptr[r], e1 = b_ptr[r + 1];
  const float* lm = lc_mid + (size_t)bh * NB * RM;
  for (int i = threadIdx.x; i < (e1 - e0) * RM; i += blockDim.x) {
    const int e = e0 + i / RM, gg = i % RM;
    lcb[(size_t)e * RM + gg] = lm[b_idx[e] * RM + gg];
  }
}

// ---------------------------------------------------------------- representative tables
// k_sw/v_sw: [BH][NP][NB+1][D/64][RM][64] bf16 -- block j's RM representatives, one record per swizzle
// phase, each panel's RM rows pre-swizzled for tile rows phi..phi+RM-1 (phi = pi*RM % 8). Record NB is
// zeros (empty slots). kt/vt: [BH][NBt][D] block means, rows >= NB zero. lc_mid [BH][NB*RM] and lc_tail
// [BH][NBt] are log2 of the valid-token counts, -inf for empty groups and padded rows.
template <int D>
__global__ void build_kernel(const bf16* __restrict__ k, const bf16* __restrict__ v, const int* __restrict__ blk_cnt,
                             bf16* __restrict__ k_sw, bf16* __restrict__ v_sw, float* __restrict__ lc_mid,
                             bf16* __restrict__ kt, bf16* __restrict__ vt, float* __restrict__ lc_tail, int L, int NB,
                             int NBt, int RM) {
  constexpr int NPAN = D / 64;
  const int blk = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
  const int NP = num_phases(RM), GS = kB / RM;
  const int p = d >> 6, c = d & 63;
  const size_t rec = (size_t)NPAN * RM * 64;
  auto sw_off = [&](int pi, int b, int g) {
    const int phi = (pi * RM) & 7;
    const int chunk = (c >> 3) ^ ((phi + g) & 7);
    return ((((size_t)bh * NP + pi) * (NB + 1) + b) * rec) + ((size_t)p * RM + g) * 64 + chunk * 8 + (c & 7);
  };
  if (blk >= NB) {  // padding: zero record NB of the mid tables, zero / -inf tail rows NB..NBt-1
    if (blk == NB)
      for (int pi = 0; pi < NP; ++pi)
        for (int g = 0; g < RM; ++g) {
          k_sw[sw_off(pi, NB, g)] = __float2bfloat16(0.f);
          v_sw[sw_off(pi, NB, g)] = __float2bfloat16(0.f);
        }
    if (blk < NBt) {
      kt[((size_t)bh * NBt + blk) * D + d] = __float2bfloat16(0.f);
      vt[((size_t)bh * NBt + blk) * D + d] = __float2bfloat16(0.f);
      if (d == 0) lc_tail[(size_t)bh * NBt + blk] = ninf();
    }
    return;
  }
  const int cnt = blk_cnt[blk];
  const bf16* gk = k + ((size_t)bh * L + (size_t)blk * kB) * D + d;
  const bf16* gv = v + ((size_t)bh * L + (size_t)blk * kB) * D + d;
  float tk = 0.f, tv = 0.f;
  for (int g = 0; g < RM; ++g) {
    const int n = min(max(cnt - g * GS, 0), GS);
    float ak = 0.f, av = 0.f;
    for (int s = 0; s < n; ++s) {
      ak += __bfloat162float(gk[(size_t)(g * GS + s) * D]);
      av += __bfloat162float(gv[(size_t)(g * GS + s) * D]);
    }
    tk += ak;
    tv += av;
    const float inv = n > 0 ? 1.f / (float)n : 0.f;
    const bf16 mk = __float2bfloat16(ak * inv), mv = __float2bfloat16(av * inv);
    for (int pi = 0; pi < NP; ++pi) {
      k_sw[sw_off(pi, blk, g)] = mk;
      v_sw[sw_off(pi, blk, g)] = mv;
    }
    if (d == 0) lc_mid[((size_t)bh * NB + blk) * RM + g] = n > 0 ? __log2f((float)n) : ninf();
  }
  const float invc = cnt > 0 ? 1.f / (float)cnt : 0.f;
  kt[((size_t)bh * NBt + blk) * D + d] = __float2bfloat16(tk * invc);
  vt[((size_t)bh * NBt + blk) * D + d] = __float2bfloat16(tv * invc);
  if (d == 0) lc_tail[(size_t)bh * NBt + blk] = cnt > 0 ? __log2f((float)cnt) : ninf();
}

// The block ids of the slots this lane copies for tier-B tile `tb` (slots lane and lane + 32): loaded
// one stage ahead of the copies, so warp 0 never stalls on them while it is also a consumer.
__device__ __forceinline__ void slot_blocks(const int* b_idx, int nb_list, int tb, int NB, int RM, int (&blk)[2]) {
  const int G = kB / RM;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int sl = laneid() + 32 * u, pos = tb * G + sl;
    blk[u] = (sl < G && pos < nb_list) ? b_idx[pos] : NB;  // empty slot: the zero record
  }
}

// Issue the bulk copies of a tier-B tile into a st_bf<64, D> buffer. Called by all 32 lanes of one
// warp with their slot_blocks(); lane 0 has already armed `bar` for the full tile.
template <int D>
__device__ __forceinline__ void issue_tier_b(uint32_t dst, const bf16* sw_bh, const int (&blk)[2], int NB, int RM,
                                             semaphore& bar, bool one_per_lane = false) {
  constexpr int NPAN = D / 64;
  const int G = kB / RM, NP = num_phases(RM);
  const size_t rec = (size_t)NPAN * RM * 64;
  const uint32_t bytes = (uint32_t)RM * 128u;
  if (one_per_lane && NPAN == 2 && G <= 16) {  // diagnostic layout: lane = 2 * slot + panel
    const int sl = laneid() >> 1, p = laneid() & 1;
    const int b_ = __shfl_sync(0xffffffffu, blk[0], sl);
    if (sl < G) {
      const int pi = (NP == 1) ? 0 : (sl % NP);
      const bf16* src = sw_bh + ((size_t)pi * (NB + 1) + b_) * rec;
      bulk_load(dst + p * (kB * 128) + (uint32_t)(RM * sl) * 128u, src + (size_t)p * RM * 64, bytes, bar);
    }
    return;
  }
#pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int sl = laneid() + 32 * u;
    if (sl >= G) break;
    const int pi = (NP == 1) ? 0 : (sl % NP);
    const bf16* src = sw_bh + ((size_t)pi * (NB + 1) + blk[u]) * rec;
#pragma unroll
    for (int p = 0; p < NPAN; ++p)
      bulk_load(dst + p * (kB * 128) + (uint32_t)(RM * sl) * 128u, src + (size_t)p * RM * 64, bytes, bar);
  }
}

// ---------------------------------------------------------------- forward
template <int D>
struct fwd_globals {
  using tile = st_bf<kB, D>;
  using tgl = gl<bf16, -1, -1, -1, -1, tile>;
  using l_col_vec = col_vec<st_fl<kB, D>>;
  using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
  tgl q, k, v, o, kt, vt;
  const bf16* k_sw;
  const bf16* v_sw;
  const float* lc_mid;
  const float* lc_tail;
  const float* lcb;  // [n_B * RM] flattened tier-B log2 counts per absolute entry (lcb_kernel)
  l_gl lse;
  const int* a_ptr;
  const int* a_idx;
  const int* b_ptr;
  const int* b_idx;
  const uint32_t* bits_ab;
  const int* blk_cnt;
  int H, L, NB, NBt, RM, W, phases;
  // Measurement only (wrong results): bit 0 = tier-B tiles take bias 0 (no bias loads), bit 1 = tier-B
  // tiles are loaded as one tensor-TMA tile (tail tile 0) instead of per-slot bulk copies. Bit 2 (correct
  // results) = spread a tier-B tile's copies one per lane instead of one slot per lane.
  int diag;
};

template <int D>
__device__ __forceinline__ void issue_tile(int2 tile, typename fwd_globals<D>::tile& dst,
                                           const typename fwd_globals<D>::tgl& tok,
                                           const typename fwd_globals<D>::tgl& tail, const bf16* sw_bh,
                                           const int (&blk)[2], int b, int h, int NB, int RM, semaphore& bar,
                                           bool diag_tma = false, bool lane_split = false) {
  using tile_t = typename fwd_globals<D>::tile;
  // warp-wide: lane 0 arms the barrier and issues tensor TMA; all lanes share the tier-B bulk copies
  if (laneid() == 0) tma::expect_bytes(bar, sizeof(tile_t));
  if (tile.x == kTierB && diag_tma) {
    if (laneid() == 0) {
      coord<tile_t> c = {b, h, 0, 0};
      tma::load_async(dst, tail, c, bar);
    }
  } else if (tile.x == kTierB) {
    issue_tier_b<D>(static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0])), sw_bh, blk, NB, RM, bar, lane_split);
  } else if (laneid() == 0) {
    coord<tile_t> c = {b, h, tile.y, 0};
    if (tile.x == kTierA) tma::load_async(dst, tok, c, bar);
    else tma::load_async(dst, tail, c, bar);
  }
}

// MINB: CTAs per SM the register budget is sized for. 4 is VSA2's TK forward, capping at 128 registers;
// this kernel carries more live state (the tile stream, a tile of biases ahead) and measured 0.723 ms at
// MINB 4 vs 0.521 ms at MINB 3 for tier A alone (L=16k H=16 D=128; native TK 0.487). REP_TK_FWD_CFG=0
// selects 4 for comparison.
template <int D, int MINB, bool A_ONLY = false>
__global__ __launch_bounds__(128, MINB) void fwd_ker(const __grid_constant__ fwd_globals<D> g) {
#if REP_TK_HOPPER
  constexpr float SL2 = kLog2e * (D == 64 ? 0.125f : 0.08838834764f);
  constexpr int ND = D / 16;
  using tile_t = typename fwd_globals<D>::tile;
  const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int bh = b * g.H + h;
  const int warp = warpid(), lane = laneid(), gq = lane >> 2;

  Stream s;
  {
    const size_t r = (size_t)bh * (g.NB + 1) + qb;
    const int a0 = g.a_ptr[r], b0 = g.b_ptr[r];
    s.a_idx = g.a_idx + a0;
    s.na = g.a_ptr[r + 1] - a0;
    s.b_idx = g.b_idx + b0;
    s.nb_list = g.b_ptr[r + 1] - b0;
    s.ab_row = g.bits_ab + ((size_t)bh * g.NB + qb) * g.W;
    s.G = kB / g.RM;
    s.nbt_tiles = (s.nb_list + s.G - 1) / s.G;
    s.nc_tiles = g.NBt / kB;
    s.NB = g.NB;
    s.W = g.W;
    s.phases = g.phases;
    s.ia = s.ib = s.tc = 0;
  }
  const float* lc_tail = g.lc_tail + (size_t)bh * g.NBt;
  const float* lcb = g.lcb + (size_t)(s.b_idx - g.b_idx) * g.RM;  // this row's entries
  const bf16* ksw = g.k_sw + (size_t)bh * num_phases(g.RM) * (g.NB + 1) * (D / 64) * g.RM * 64;
  const bf16* vsw = g.v_sw + (size_t)bh * num_phases(g.RM) * (g.NB + 1) * (D / 64) * g.RM * 64;
  const int qcnt = g.blk_cnt[qb];

  int2 cur = s.next();
  if (cur.x == kTierNone) {
    bf16* o = g.o.raw_ptr + ((size_t)bh * g.L + (size_t)qb * kB) * D;
    for (int i = threadIdx.x; i < kB * D / 8; i += blockDim.x) reinterpret_cast<uint4*>(o)[i] = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < kB) g.lse.raw_ptr[(size_t)bh * g.L + (size_t)qb * kB + threadIdx.x] = ninf();
    return;
  }

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);
  tile_t(&q_smem) = al.allocate<tile_t>();
  tile_t(&k_smem) = al.allocate<tile_t>();
  tile_t(&v_smem) = al.allocate<tile_t>();
  typename fwd_globals<D>::l_col_vec(&l_smem) = al.allocate<typename fwd_globals<D>::l_col_vec>();
  __shared__ semaphore q_bar, k_bar, v_bar;
  if (threadIdx.x == 0) {
    init_semaphore(q_bar, 0, 1);
    init_semaphore(k_bar, 0, 1);
    init_semaphore(v_bar, 0, 1);
    tma::expect_bytes(q_bar, sizeof(tile_t));
    coord<tile_t> c = {b, h, qb, 0};
    tma::load_async(q_smem, g.q, c, q_bar);
  }
  __syncthreads();  // barriers initialized before warp 0's other lanes use them
  int slots[2] = {g.NB, g.NB};
  if (warp == 0) {
    if (cur.x == kTierB) slot_blocks(s.b_idx, s.nb_list, cur.y, g.NB, g.RM, slots);
    issue_tile<D>(cur, k_smem, g.k, g.kt, ksw, slots, b, h, g.NB, g.RM, k_bar, g.diag & 2, g.diag & 4);
    issue_tile<D>(cur, v_smem, g.v, g.vt, vsw, slots, b, h, g.NB, g.RM, v_bar, g.diag & 2, g.diag & 4);
  }
  TileBias cb;
  cb.masked = false;
  cb.kc = kB;

  rt_fl<16, 64> att;
  rt_bf<16, 64> att_mma;
  rt_fl<16, D> o_reg;
  // Running statistics in TK col_vecs, so the reductions and rescalings are its row_max / row_sum /
  // sub_row / mul_row rather than hand-rolled shuffles. Unlike VSA2's kernel, max_vec is kept already
  // in log2 units: a representative's + log2(c) bias is per column and cannot be factored out of the
  // max, so the bias has to be added before the row max is taken.
  col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_last, alpha;
  zero(o_reg);
  neg_infty(max_vec);
  neg_infty(max_last);
  zero(norm_vec);
  wait(q_bar, 0);

  // One pipeline step: S = Q K^T for `cur`, online softmax, O += P V, with the loads of `nxt` issued as
  // soon as their buffers free up (VSA2's TK order). BC = false is a tier-A tile: no bias state, the
  // mask formed from its token count only when the block is partial (like VSA2's right_fill). BC = true
  // is a tier-B/C tile with its biases `cb` computed one tile earlier.
  int it = 0;
  auto step = [&]<bool BC>(int2 tile, int2 nxt, int kcA, const TileBias& cb, const int (&nsl)[2]) {
    const bool masked = BC && cb.masked && !(tile.x == kTierB && (g.diag & 1));
    wait(k_bar, it & 1);
    warpgroup::mm_ABt(att, q_smem, k_smem);
    warpgroup::mma_async_wait();
    if (nxt.x != kTierNone && warp == 0)
      issue_tile<D>(nxt, k_smem, g.k, g.kt, ksw, nsl, b, h, g.NB, g.RM, k_bar, g.diag & 2, g.diag & 4);

    mul(att, att, SL2);  // raw q.k -> log2 units
    if constexpr (BC) {
      if (masked) {  // + log2(c) per column, -inf where the column is absent
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float2& e = att.tiles[0][i].data[j];
            e.x += cb.v[i][(j >> 1) * 2];
            e.y += cb.v[i][(j >> 1) * 2 + 1];
          }
      }
    } else if (kcA < kB) {  // partial tier-A block: VSA2's own column mask
      right_fill(att, att, kcA, base_types::constants<float>::neg_infty());
    }
    row_max(max_vec, att, max_vec);
    sub_row(att, att, max_vec);
    exp2(att, att);
    sub(alpha, max_last, max_vec);
    exp2(alpha, alpha);
    mul(norm_vec, norm_vec, alpha);
    row_sum(norm_vec, att, norm_vec);
    copy(att_mma, att);
    mul_row(o_reg, o_reg, alpha);
    copy(max_last, max_vec);

    wait(v_bar, it & 1);
    warpgroup::mma_AB(o_reg, att_mma, v_smem);
    warpgroup::mma_async_wait();
    if (nxt.x != kTierNone && warp == 0)
      issue_tile<D>(nxt, v_smem, g.v, g.vt, vsw, nsl, b, h, g.NB, g.RM, v_bar, g.diag & 2, g.diag & 4);
    ++it;
  };

  // Tier A: no bias registers live at all, as in VSA2's own kernel.
  while (cur.x == kTierA) {
    const int2 nxt = s.next();
    int nsl[2] = {g.NB, g.NB};
    if (warp == 0 && nxt.x == kTierB) slot_blocks(s.b_idx, s.nb_list, nxt.y, g.NB, g.RM, nsl);
    step.template operator()<false>(cur, nxt, g.blk_cnt[cur.y], cb, nsl);
    cur = nxt;
  }
  if constexpr (A_ONLY) goto epilogue;  // diagnostic specialisation: no B/C state compiled in
  // Tiers B and C: biases and slot ids loaded one tile ahead, consumed one tile later, so their latency
  // hides behind a tile of MMAs instead of sitting on the critical path.
  if (cur.x != kTierNone) cb = tile_bias(cur, s, g.blk_cnt, lcb, lc_tail, g.RM);
  while (cur.x != kTierNone) {
    const int2 nxt = s.next();
    const TileBias nb_ = tile_bias(nxt, s, g.blk_cnt, lcb, lc_tail, g.RM);
    int nsl[2] = {g.NB, g.NB};
    if (warp == 0 && nxt.x == kTierB) slot_blocks(s.b_idx, s.nb_list, nxt.y, g.NB, g.RM, nsl);
    step.template operator()<true>(cur, nxt, kB, cb, nsl);
    cur = nxt;
    cb = nb_;
  }

epilogue:
  // o = o / l, natural lse = max * ln2 + ln(l). Padded query rows get o = 0 (as the generic backend
  // does) but a finite lse, which the TK backward relies on, like VSA2's own kernel.
  div_row(o_reg, o_reg, norm_vec);
  {
    const int r0 = 16 * warp + gq;
#pragma unroll
    for (int n = 0; n < ND; ++n)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (r0 + 8 * (j & 1) >= qcnt) {
          o_reg.tiles[0][n].data[j].x = 0.f;
          o_reg.tiles[0][n].data[j].y = 0.f;
        }
      }
  }
  mul(max_vec, max_vec, kLn2);
  log(norm_vec, norm_vec);
  add(norm_vec, norm_vec, max_vec);

  __syncthreads();  // every warp is done with q_smem
  warpgroup::store(q_smem, o_reg);
  warpgroup::store(l_smem, norm_vec);
  __syncthreads();
  if (warp == 0) {  // TK's store/wait helpers are warp-collective (lane-0 guard + __syncwarp inside)
    coord<tile_t> c = {b, h, qb, 0};
    tma::store_async(g.o, q_smem, c);
    coord<typename fwd_globals<D>::l_col_vec> lc = {b, h, 0, qb};
    tma::store_async(g.lse, l_smem, lc);
    tma::store_async_wait();
  }
#endif
}

// ---------------------------------------------------------------- backward, tier C (KV-owner)
// One CTA per (tail tile t, h, b): it owns block means 64t..64t+63, keeps their dK/dV in registers over
// every query block that has a tier-C column in the tile, and reduces each pair's dQ into the shared fp32
// accumulator with tensor TMA. Body and pipeline are VSA2's TK backward; the only change is a per-kv-row
// bias -- log2 of the block's token count, or -inf when the block is not tier C for this query block --
// in place of its `lower_fill` padding mask.
template <int D>
struct bwdc_globals {
  using tb = st_bf<kB, D>;
  using tf = st_fl<kB, D>;
  using lv = row_vec<st_fl<kB, kB>>;
  gl<bf16, -1, -1, -1, -1, tb> q, og, kt, vt;
  gl<float, -1, -1, -1, -1, tf> qg, kgt, vgt;
  gl<float, -1, -1, -1, -1, lv> l, d;
  const float* lc_tail;
  const uint32_t* bits_ab;
  int H, NB, NBt, W;
  int splits;  // query range of a tail tile split over this many CTAs (gridDim.z = B * splits)
};

__device__ static inline void stream_tile(auto& reg_tile, auto& smem_vec) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int base_col = 16 * i + 2 * (laneid() % 4);
    reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[base_col + 0];
    reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[base_col + 0];
    reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[base_col + 8];
    reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[base_col + 8];
  }
}
__device__ static inline void stream_sub_tile(auto& reg_tile, auto& smem_vec) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int base_col = 16 * i + 2 * (laneid() % 4);
    const float2 a = *(float2*)&smem_vec[base_col + 0], c = *(float2*)&smem_vec[base_col + 8];
    reg_tile.tiles[0][i].data[0].x -= a.x; reg_tile.tiles[0][i].data[0].y -= a.y;
    reg_tile.tiles[0][i].data[1].x -= a.x; reg_tile.tiles[0][i].data[1].y -= a.y;
    reg_tile.tiles[0][i].data[2].x -= c.x; reg_tile.tiles[0][i].data[2].y -= c.y;
    reg_tile.tiles[0][i].data[3].x -= c.x; reg_tile.tiles[0][i].data[3].y -= c.y;
  }
}

// C columns of tail tile t for query block qb (same rule as Stream::cmask).
__device__ __forceinline__ uint2 c_mask(const uint32_t* bits_ab, int qb, int t, int NB, int W) {
  const uint32_t* row = bits_ab + (size_t)qb * W;
  const int w0 = 2 * t, w1 = 2 * t + 1;
  uint32_t lo = w0 < W ? ~row[w0] : 0u, hi = w1 < W ? ~row[w1] : 0u;
  const int n0 = NB - 64 * t;
  if (n0 < 32) lo &= (n0 <= 0) ? 0u : ((1u << n0) - 1u);
  if (n0 < 64) hi &= (n0 <= 32) ? 0u : ((n0 - 32 >= 32) ? 0xffffffffu : ((1u << (n0 - 32)) - 1u));
  return make_uint2(lo, hi);
}
__device__ __forceinline__ int next_c_query(const uint32_t* bits_ab, int from, int t, int NB, int W, int end) {
  for (int qb = from; qb < end; ++qb) {
    const uint2 m = c_mask(bits_ab, qb, t, NB, W);
    if (m.x | m.y) return qb;
  }
  return -1;
}

template <int D>
__global__ __launch_bounds__(128, (D == 64) ? 3 : 2) void bwd_c_ker(const __grid_constant__ bwdc_globals<D> g) {
#if REP_TK_HOPPER
  constexpr float SCALE = (D == 64) ? 0.125f : 0.08838834764f;
  constexpr float SL2 = kLog2e * SCALE;
  constexpr float NEG_INV_SCALE = (D == 64) ? -8.0f : -11.313708499f;
  using tb = typename bwdc_globals<D>::tb;
  using tf = typename bwdc_globals<D>::tf;
  using lv = typename bwdc_globals<D>::lv;
  using attn_tile = st_bf<kB, kB>;
  const int t = blockIdx.x, h = blockIdx.y, b = blockIdx.z / g.splits, sp = blockIdx.z % g.splits;
  const int bh = b * g.H + h;
  // query blocks [q_lo, q_hi) of this split; the CTA's dK/dV partial is reduce-added like every split's
  const int per = (g.NB + g.splits - 1) / g.splits, q_lo = sp * per, q_hi = min(g.NB, q_lo + per);
  const uint32_t* bits = g.bits_ab + (size_t)bh * g.NB * g.W;
  int qb_next = next_c_query(bits, q_lo, t, g.NB, g.W, q_hi);
  if (qb_next < 0) return;  // no query block uses this tile: its dK/dV stay zero

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);
  tb(&k_smem) = al.allocate<tb>();
  tb(&v_smem) = al.allocate<tb>();
  tb(&q_smem) = al.allocate<tb>();
  tb(&og_smem) = al.allocate<tb>();
  tf(&qg_smem) = al.allocate<tf>();
  lv(&l_smem) = al.allocate<lv>();
  lv(&d_smem) = al.allocate<lv>();
  attn_tile(&ds_smem) = al.allocate<attn_tile>();
  tf(&kg_smem) = *reinterpret_cast<tf*>(&k_smem.data[0]);  // spans k_smem + v_smem
  tf(&vg_smem) = *reinterpret_cast<tf*>(&q_smem.data[0]);  // spans q_smem + og_smem

  __shared__ semaphore kv_b, q_b, o_b, vec_b;
  if (threadIdx.x == 0) {
    init_semaphore(kv_b, 0, 1);
    init_semaphore(q_b, 0, 1);
    init_semaphore(o_b, 0, 1);
    init_semaphore(vec_b, 0, 1);
    tma::expect_bytes(kv_b, 2 * sizeof(tb));
    coord<tb> ck = {b, h, t, 0};
    tma::load_async(k_smem, g.kt, ck, kv_b);
    tma::load_async(v_smem, g.vt, ck, kv_b);
    coord<tb> cq = {b, h, qb_next, 0};
    coord<lv> cv = {b, h, 0, qb_next};
    tma::expect_bytes(o_b, sizeof(tb));
    tma::load_async(og_smem, g.og, cq, o_b);
    tma::expect_bytes(vec_b, 2 * sizeof(lv));
    tma::load_async(l_smem, g.l, cv, vec_b);
    tma::load_async(d_smem, g.d, cv, vec_b);
    tma::expect_bytes(q_b, sizeof(tb));
    tma::load_async(q_smem, g.q, cq, q_b);
  }
  __syncthreads();

  // this thread's two kv rows of S^T, their log2 counts
  const int r0 = 16 * warpid() + (laneid() >> 2), r1 = r0 + 8;
  const float lc0 = g.lc_tail[(size_t)bh * g.NBt + 64 * t + r0];
  const float lc1 = g.lc_tail[(size_t)bh * g.NBt + 64 * t + r1];

  rt_fl<16, D> kg_reg, vg_reg;
  rt_fl<16, kB> s_t, dp_t, ds_t;
  rt_bf<16, kB> ds_mma, p_mma;
  zero(kg_reg);
  zero(vg_reg);
  wait(kv_b, 0);

  for (int it = 0; qb_next >= 0; ++it) {
    const int qb = qb_next;
    qb_next = next_c_query(bits, qb + 1, t, g.NB, g.W, q_hi);
    const bool has_next = qb_next >= 0;
    const uint2 m = c_mask(bits, qb, t, g.NB, g.W);
    const bool on0 = ((r0 < 32 ? m.x >> r0 : m.y >> (r0 - 32)) & 1u) != 0u;
    const bool on1 = ((r1 < 32 ? m.x >> r1 : m.y >> (r1 - 32)) & 1u) != 0u;
    const float b0 = on0 ? lc0 : ninf(), b1 = on1 ? lc1 : ninf();

    wait(o_b, it & 1);
    warpgroup::mm_ABt(dp_t, v_smem, og_smem);  // dP^T = V dO^T
    warpgroup::mma_commit_group();
    wait(vec_b, it & 1);
    stream_tile(s_t, l_smem);
    mul(s_t, s_t, NEG_INV_SCALE);
    wait(q_b, it & 1);
    warpgroup::mma_ABt(s_t, k_smem, q_smem);  // S^T - lse/scale
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    mul(s_t, s_t, SL2);
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // rows r0 (data 0, 2) and r1 (data 1, 3)
      s_t.tiles[0][i].data[0].x += b0; s_t.tiles[0][i].data[0].y += b0;
      s_t.tiles[0][i].data[2].x += b0; s_t.tiles[0][i].data[2].y += b0;
      s_t.tiles[0][i].data[1].x += b1; s_t.tiles[0][i].data[1].y += b1;
      s_t.tiles[0][i].data[3].x += b1; s_t.tiles[0][i].data[3].y += b1;
    }
    exp2(s_t, s_t);
    copy(p_mma, s_t);
    stream_sub_tile(dp_t, d_smem);   // dP - delta
    mul(ds_t, s_t, dp_t);
    mul(ds_t, ds_t, SCALE);

    if (has_next && threadIdx.x == 0) {
      coord<lv> cv = {b, h, 0, qb_next};
      tma::expect_bytes(vec_b, 2 * sizeof(lv));
      tma::load_async(l_smem, g.l, cv, vec_b);
      tma::load_async(d_smem, g.d, cv, vec_b);
    }
    warpgroup::mma_AB(vg_reg, p_mma, og_smem);  // dV += P^T dO
    warpgroup::mma_commit_group();
    copy(ds_mma, ds_t);
    warpgroup::store(ds_smem, ds_t);
    warpgroup::mma_async_wait();
    if (has_next && threadIdx.x == 0) {
      coord<tb> cq = {b, h, qb_next, 0};
      tma::expect_bytes(o_b, sizeof(tb));
      tma::load_async(og_smem, g.og, cq, o_b);
    }
    warpgroup::mma_AB(kg_reg, ds_mma, q_smem);  // dK += dS^T Q
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    if (has_next && threadIdx.x == 0) {
      coord<tb> cq = {b, h, qb_next, 0};
      tma::expect_bytes(q_b, sizeof(tb));
      tma::load_async(q_smem, g.q, cq, q_b);
    }
    rt_fl<16, D> qg_reg;
    __syncthreads();
    warpgroup::mm_AtB(qg_reg, ds_smem, k_smem);  // dQ = dS K
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    warpgroup::store(qg_smem, qg_reg);
    __syncthreads();
    if (warpid() == 0) {
      coord<tf> cq = {b, h, qb, 0};
      tma::store_add_async(g.qg, qg_smem, cq);
      tma::store_async_read_wait();
    }
  }

  __syncthreads();
  warpgroup::store(kg_smem, kg_reg);
  __syncthreads();
  if (warpid() == 0) {
    coord<tf> c = {b, h, t, 0};
    tma::store_add_async(g.kgt, kg_smem, c);
    tma::store_commit_group();
  }
  warpgroup::store(vg_smem, vg_reg);
  __syncthreads();
  if (warpid() == 0) {
    coord<tf> c = {b, h, t, 0};
    tma::store_add_async(g.vgt, vg_smem, c);
    tma::store_commit_group();
    tma::store_async_wait();
  }
#endif
}

// ---------------------------------------------------------------- backward, tier B (query-owner)
// One CTA per (query block, h, b), walking its tier-B tiles. dQ accumulates in registers and is reduced
// once at the end. Representative dK/dV of a tile (64 rows = 64/RM slots x RM) are staged in shared
// memory as plain row-major fp32 and reduced into the fp32 mid-gradient table with one 1-D bulk
// reduce-add per slot: a slot's RM rows are contiguous both in the staging buffer and in the table.
// The staging buffer reuses the K/V tiles, which are free once dQ's MMA has read K.
template <int D>
struct bwdb_globals {
  using tb = st_bf<kB, D>;
  using tf = st_fl<kB, D>;
  gl<bf16, -1, -1, -1, -1, tb> q, og;
  gl<float, -1, -1, -1, -1, tf> qg;
  const bf16* k_sw;
  const bf16* v_sw;
  const float* lcb;  // flattened tier-B log2 counts (rep_tk_lcb)
  const float* lse;
  const float* delta;
  float* dk_mid;  // [BH, NB*RM, D] fp32
  float* dv_mid;
  const int* b_ptr;
  const int* b_idx;
  int H, L, NB, RM;
};

template <int D>
__global__ __launch_bounds__(128, (D == 64) ? 3 : 2) void bwd_b_ker(const __grid_constant__ bwdb_globals<D> g) {
#if REP_TK_HOPPER
  constexpr float SCALE = (D == 64) ? 0.125f : 0.08838834764f;
  constexpr float SL2 = kLog2e * SCALE;
  constexpr int ND = D / 16;
  using tb = typename bwdb_globals<D>::tb;
  using tf = typename bwdb_globals<D>::tf;
  using attn_tile = st_bf<kB, kB>;
  const int qb = blockIdx.x, h = blockIdx.y, b = blockIdx.z, bh = b * g.H + h;
  const size_t r = (size_t)bh * (g.NB + 1) + qb;
  const int b0 = g.b_ptr[r], nb_list = g.b_ptr[r + 1] - b0;
  if (nb_list <= 0) return;
  const int RM = g.RM, G = kB / RM;
  const int ntiles = (nb_list + G - 1) / G;
  const int* b_idx = g.b_idx + b0;
  const size_t sw_bh = (size_t)bh * num_phases(RM) * (g.NB + 1) * (D / 64) * RM * 64;
  const bf16* ksw = g.k_sw + sw_bh;
  const bf16* vsw = g.v_sw + sw_bh;
  const float* lcb = g.lcb + (size_t)b0 * RM;
  float* dkm = g.dk_mid + (size_t)bh * g.NB * RM * D;
  float* dvm = g.dv_mid + (size_t)bh * g.NB * RM * D;
  const int warp = warpid(), lane = laneid(), gq = lane >> 2, t4 = lane & 3;

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);
  tb(&q_smem) = al.allocate<tb>();
  tb(&og_smem) = al.allocate<tb>();
  tb(&k_smem) = al.allocate<tb>();
  tb(&v_smem) = al.allocate<tb>();
  attn_tile(&p_smem) = al.allocate<attn_tile>();
  attn_tile(&ds_smem) = al.allocate<attn_tile>();
  float* stage = reinterpret_cast<float*>(&k_smem.data[0]);  // [64][D] fp32, spans k_smem + v_smem
  tf(&qg_smem) = *reinterpret_cast<tf*>(&k_smem.data[0]);
  const uint32_t stage_s = static_cast<uint32_t>(__cvta_generic_to_shared(stage));

  __shared__ semaphore qo_b, kv_b;
  if (threadIdx.x == 0) {
    init_semaphore(qo_b, 0, 1);
    init_semaphore(kv_b, 0, 1);
    tma::expect_bytes(qo_b, 2 * sizeof(tb));
    coord<tb> c = {b, h, qb, 0};
    tma::load_async(q_smem, g.q, c, qo_b);
    tma::load_async(og_smem, g.og, c, qo_b);
  }
  __syncthreads();
  auto issue_kv = [&](const int (&sl)[2]) {
    if (lane == 0) tma::expect_bytes(kv_b, 2 * sizeof(tb));
    const uint32_t kd = static_cast<uint32_t>(__cvta_generic_to_shared(&k_smem.data[0]));
    const uint32_t vd = static_cast<uint32_t>(__cvta_generic_to_shared(&v_smem.data[0]));
    issue_tier_b<D>(kd, ksw, sl, g.NB, RM, kv_b);
    issue_tier_b<D>(vd, vsw, sl, g.NB, RM, kv_b);
  };
  if (warp == 0) {
    int sl0[2];
    slot_blocks(b_idx, nb_list, 0, g.NB, RM, sl0);
    issue_kv(sl0);
  }

  // this thread's query rows (gq, gq + 8 of its warp): lse in log2 units, delta
  const size_t row0 = (size_t)bh * g.L + (size_t)qb * kB + 16 * warp + gq;
  const float lse2[2] = {g.lse[row0] * kLog2e, g.lse[row0 + 8] * kLog2e};
  const float dl[2] = {g.delta[row0], g.delta[row0 + 8]};

  rt_fl<16, D> dq_reg;
  rt_fl<16, kB> s_t, dp_t;
  rt_bf<16, kB> ds_mma;
  zero(dq_reg);
  wait(qo_b, 0);

  for (int it = 0; it < ntiles; ++it) {
    const int first = it * G, cnt = min(G, nb_list - first);
    int nsl[2] = {g.NB, g.NB};  // next tile's slot blocks, loaded a tile ahead of their copies
    if (warp == 0 && it + 1 < ntiles) slot_blocks(b_idx, nb_list, it + 1, g.NB, RM, nsl);
    float bias[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c = 16 * i + 2 * t4 + (j & 1) + 8 * (j >> 1);
        bias[i][j] = c < cnt * RM ? lcb[first * RM + c] : ninf();
      }

    wait(kv_b, it & 1);
    warpgroup::mm_ABt(s_t, q_smem, k_smem);    // S  = Q K^T
    warpgroup::mma_commit_group();
    warpgroup::mm_ABt(dp_t, og_smem, v_smem);  // dP = dO V^T
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float2& e = s_t.tiles[0][i].data[j];
        float2& dpe = dp_t.tiles[0][i].data[j];
        const int rr = j & 1;
        e.x = exp2f(e.x * SL2 + bias[i][(j >> 1) * 2] - lse2[rr]);
        e.y = exp2f(e.y * SL2 + bias[i][(j >> 1) * 2 + 1] - lse2[rr]);
        dpe.x = e.x * (dpe.x - dl[rr]) * SCALE;  // dS, in place
        dpe.y = e.y * (dpe.y - dl[rr]) * SCALE;
      }
    copy(ds_mma, dp_t);
    warpgroup::mma_AB(dq_reg, ds_mma, k_smem);  // dQ += dS K
    warpgroup::mma_commit_group();
    warpgroup::store(p_smem, s_t);
    warpgroup::store(ds_smem, dp_t);
    warpgroup::mma_async_wait();
    __syncthreads();  // P, dS visible; every warp's dQ MMA has read K

    rt_fl<16, D> gr;
    warpgroup::mm_AtB(gr, p_smem, og_smem);  // dV_rep = P^T dO   (rows = the tile's 64 representatives)
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    auto stage_and_reduce = [&](float* table) {
      // gr rows 16*warp + gq (+8), columns 16n + 2t4 (+1, +8, +9)
#pragma unroll
      for (int n = 0; n < ND; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int row = 16 * warp + gq + 8 * (j & 1), col = 16 * n + 2 * t4 + 8 * (j >> 1);
          *reinterpret_cast<float2*>(stage + row * D + col) = gr.tiles[0][n].data[j];
        }
      fence_async_smem();
      __syncthreads();
      if (warp == 0) {
        for (int sl = lane; sl < cnt; sl += 32)
          bulk_reduce_add_f32(table + (size_t)b_idx[first + sl] * RM * D, stage_s + (uint32_t)(sl * RM * D * 4),
                              (uint32_t)(RM * D * 4));
        bulk_commit();
      }
    };
    stage_and_reduce(dvm);
    warpgroup::mm_AtB(gr, ds_smem, q_smem);  // dK_rep = dS^T Q, overlapping the dV reduce's reads
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    if (warp == 0) bulk_wait_read0();
    __syncthreads();
    stage_and_reduce(dkm);
    if (warp == 0) bulk_wait_read0();
    __syncthreads();  // staging (= K/V) free again
    if (it + 1 < ntiles && warp == 0) issue_kv(nsl);
  }

  // dQ of the tier-B columns into the shared fp32 accumulator (K/V region is idle)
  warpgroup::store(qg_smem, dq_reg);
  __syncthreads();
  if (warp == 0) {
    coord<tf> c = {b, h, qb, 0};
    tma::store_add_async(g.qg, qg_smem, c);
    tma::store_async_wait();
  }
#endif
}


// ---------------------------------------------------------------- diagnostic: native forward, copied
// VSA2's TK forward loop body (csrc/vsa/sm90_tk/fine_tk_h100.cu, unsplit path) copied verbatim, so its
// time can be compared with this file's tier-A path inside one build. MY_SOFTMAX = true swaps in this
// file's softmax (log2-domain, deferred row sums) and nothing else. Measurement only; o/lse as native.
template <int D, bool MY_SOFTMAX>
__global__ __launch_bounds__(128, 4) void nat_copy_ker(const __grid_constant__ fwd_globals<D> g) {
#if REP_TK_HOPPER
  constexpr float SCALE_LOG2 = (D == 64) ? 1.44269504089f * 0.125f : 1.44269504089f * 0.08838834764f;
  using tile_t = typename fwd_globals<D>::tile;
  const int seq_idx = blockIdx.x, h = blockIdx.y, b = blockIdx.z, bh = b * g.H + h;
  const int* rp = g.a_ptr + (size_t)bh * (g.NB + 1) + seq_idx;
  const int beg = rp[0], kv_blocks = rp[1] - beg;
  const int* idx = g.a_idx + beg;
  if (kv_blocks <= 0) return;
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);
  tile_t(&q_smem) = al.allocate<tile_t>();
  tile_t(&k_smem) = al.allocate<tile_t>();
  tile_t(&v_smem) = al.allocate<tile_t>();
  __shared__ semaphore qsmem_semaphore, k_smem_arrived, v_smem_arrived;
  if (threadIdx.x == 0) {
    const int kvb = idx[0];
    init_semaphore(qsmem_semaphore, 0, 1);
    init_semaphore(k_smem_arrived, 0, 1);
    init_semaphore(v_smem_arrived, 0, 1);
    coord<tile_t> qc = {b, h, seq_idx, 0};
    tma::expect_bytes(qsmem_semaphore, sizeof(tile_t));
    tma::load_async(q_smem, g.q, qc, qsmem_semaphore);
    coord<tile_t> kc = {b, h, kvb, 0};
    tma::expect_bytes(k_smem_arrived, sizeof(tile_t));
    tma::load_async(k_smem, g.k, kc, k_smem_arrived);
    tma::expect_bytes(v_smem_arrived, sizeof(tile_t));
    tma::load_async(v_smem, g.v, kc, v_smem_arrived);
  }
  __syncthreads();
  rt_fl<16, 64> att_block;
  rt_bf<16, 64> att_block_mma;
  rt_fl<16, D> o_reg;
  col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
  neg_infty(max_vec);
  zero(norm_vec);
  zero(o_reg);
  float m_r[2] = {ninf(), ninf()}, l_r[2] = {0.f, 0.f};
  wait(qsmem_semaphore, 0);
  for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
    const bool has_next = kv_idx + 1 < kv_blocks;
    const int nkv = has_next ? idx[kv_idx + 1] : 0;
    wait(k_smem_arrived, kv_idx % 2);
    warpgroup::mm_ABt(att_block, q_smem, k_smem);
    if constexpr (!MY_SOFTMAX) {
      copy(max_vec_last_scaled, max_vec);
      mul(max_vec_last_scaled, max_vec_last_scaled, SCALE_LOG2);
    }
    warpgroup::mma_async_wait();
    if (has_next && threadIdx.x == 0) {
      tma::expect_bytes(k_smem_arrived, sizeof(tile_t));
      coord<tile_t> kc = {b, h, nkv, 0};
      tma::load_async(k_smem, g.k, kc, k_smem_arrived);
    }
    if constexpr (!MY_SOFTMAX) {
      right_fill(att_block, att_block, g.blk_cnt[idx[kv_idx]], base_types::constants<float>::neg_infty());
      row_max(max_vec, att_block, max_vec);
      mul(att_block, att_block, SCALE_LOG2);
      mul(max_vec_scaled, max_vec, SCALE_LOG2);
      sub_row(att_block, att_block, max_vec_scaled);
      exp2(att_block, att_block);
      sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
      exp2(max_vec_last_scaled, max_vec_last_scaled);
      mul(norm_vec, norm_vec, max_vec_last_scaled);
      row_sum(norm_vec, att_block, norm_vec);
      copy(att_block_mma, att_block);
      mul_row(o_reg, o_reg, max_vec_last_scaled);
    } else {
      const int kcA = g.blk_cnt[idx[kv_idx]];
      const int lane = laneid();
      float mx[2] = {ninf(), ninf()};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2& e = att_block.tiles[0][i].data[j];
          float b0 = 0.f, b1 = 0.f;
          if (kcA < kB) {
            const int c = 16 * i + 2 * (lane & 3) + 8 * (j >> 1);
            b0 = c < kcA ? 0.f : ninf();
            b1 = c + 1 < kcA ? 0.f : ninf();
          }
          e.x = e.x * SCALE_LOG2 + b0;
          e.y = e.y * SCALE_LOG2 + b1;
          mx[j & 1] = fmaxf(mx[j & 1], fmaxf(e.x, e.y));
        }
      float mu[2], alpha[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        float v2 = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
        v2 = fmaxf(v2, __shfl_xor_sync(0xffffffffu, v2, 2));
        const float mn = fmaxf(m_r[r], v2);
        mu[r] = mn == ninf() ? 0.f : mn;
        alpha[r] = exp2f(m_r[r] - mu[r]);
        m_r[r] = mn;
        l_r[r] *= alpha[r];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float2& e = att_block.tiles[0][i].data[j];
          e.x = exp2f(e.x - mu[j & 1]);
          e.y = exp2f(e.y - mu[j & 1]);
          l_r[j & 1] += e.x + e.y;
        }
#pragma unroll
      for (int n = 0; n < D / 16; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          o_reg.tiles[0][n].data[j].x *= alpha[j & 1];
          o_reg.tiles[0][n].data[j].y *= alpha[j & 1];
        }
      copy(att_block_mma, att_block);
    }
    wait(v_smem_arrived, kv_idx % 2);
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem);
    warpgroup::mma_async_wait();
    if (has_next && threadIdx.x == 0) {
      tma::expect_bytes(v_smem_arrived, sizeof(tile_t));
      coord<tile_t> vc = {b, h, nkv, 0};
      tma::load_async(v_smem, g.v, vc, v_smem_arrived);
    }
  }
  if constexpr (MY_SOFTMAX) {
    float l = l_r[0] + __shfl_xor_sync(0xffffffffu, l_r[0], 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    for (int n = 0; n < D / 16; ++n) for (int j = 0; j < 4; ++j) { o_reg.tiles[0][n].data[j].x /= l; }
  } else {
    div_row(o_reg, o_reg, norm_vec);
  }
  warpgroup::store(q_smem, o_reg);
  __syncthreads();
  if (warpid() == 0) {
    coord<tile_t> c = {b, h, seq_idx, 0};
    tma::store_async(g.o, q_smem, c);
    tma::store_async_wait();
  }
#endif
}
}  // namespace rep_tk

// ---------------------------------------------------------------- host: tables and forward
namespace {

inline bf16* bfp(const torch::Tensor& t) { return reinterpret_cast<bf16*>(t.data_ptr<c10::BFloat16>()); }
inline int words_of(int nb) { return (nb + 31) >> 5; }
inline cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }
inline void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "rep_tk ", what, " launch failed: ", cudaGetErrorString(e));
}
void check_bf16(const torch::Tensor& t, const char* n) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == torch::kBFloat16, n,
              " must be a contiguous CUDA bf16 tensor");
}
void check_rm(int64_t RM) {
  TORCH_CHECK(RM == 1 || RM == 2 || RM == 4 || RM == 8 || RM == 16, "rep_tk: RM must be 1, 2, 4, 8 or 16");
}

}  // namespace

// Returns k_sw, v_sw, lc_mid, kt, vt, lc_tail (see build_kernel).
std::vector<torch::Tensor> rep_tk_build(torch::Tensor k, torch::Tensor v, torch::Tensor blk_cnt, int64_t RM) {
  check_bf16(k, "k");
  check_bf16(v, "v");
  check_rm(RM);
  const c10::cuda::OptionalCUDAGuard guard(k.device());
  const int B = k.size(0), H = k.size(1), L = k.size(2), D = k.size(3), NB = L / 64;
  TORCH_CHECK(D == 64 || D == 128, "rep_tk: D must be 64 or 128");
  const int NBt = (NB + 63) / 64 * 64, NP = rep_tk::num_phases((int)RM);
  auto opt = k.options();
  auto f32 = opt.dtype(torch::kFloat32);
  auto k_sw = torch::empty({B, H, NP, NB + 1, D / 64, RM, 64}, opt);
  auto v_sw = torch::empty_like(k_sw);
  auto lc_mid = torch::empty({B, H, NB * RM}, f32);
  auto kt = torch::empty({B, H, NBt, D}, opt);
  auto vt = torch::empty_like(kt);
  auto lc_tail = torch::empty({B, H, NBt}, f32);
  dim3 grid(std::max(NB + 1, NBt), B * H);
  if (D == 64)
    rep_tk::build_kernel<64><<<grid, 64, 0, stream()>>>(bfp(k), bfp(v), blk_cnt.data_ptr<int>(), bfp(k_sw), bfp(v_sw),
                                                        lc_mid.data_ptr<float>(), bfp(kt), bfp(vt),
                                                        lc_tail.data_ptr<float>(), L, NB, NBt, (int)RM);
  else
    rep_tk::build_kernel<128><<<grid, 128, 0, stream()>>>(bfp(k), bfp(v), blk_cnt.data_ptr<int>(), bfp(k_sw),
                                                          bfp(v_sw), lc_mid.data_ptr<float>(), bfp(kt), bfp(vt),
                                                          lc_tail.data_ptr<float>(), L, NB, NBt, (int)RM);
  check_launch("build");
  return {k_sw, v_sw, lc_mid, kt, vt, lc_tail};
}

// Flattened tier-B log2 counts for a selection's tier-B lists, [b_idx.numel() * RM] fp32. Built once per
// forward, reused by the backward.
torch::Tensor rep_tk_lcb(torch::Tensor lc_mid, torch::Tensor b_ptr, torch::Tensor b_idx, int64_t NB, int64_t RM) {
  const int B = lc_mid.size(0), H = lc_mid.size(1);
  auto lcb = torch::empty({std::max<long long>(1, b_idx.numel()) * RM}, lc_mid.options());
  rep_tk::lcb_kernel<<<dim3((unsigned)NB, B * H), 128, 0, stream()>>>(
      lc_mid.data_ptr<float>(), b_ptr.data_ptr<int>(), b_idx.data_ptr<int>(), lcb.data_ptr<float>(), (int)NB, (int)RM);
  check_launch("lcb");
  return lcb;
}

template <int D>
static void fwd_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, const std::vector<torch::Tensor>& tab,
                     torch::Tensor a_ptr, torch::Tensor a_idx, torch::Tensor b_ptr, torch::Tensor b_idx,
                     torch::Tensor bits_ab, torch::Tensor blk_cnt, int RM, int phases, torch::Tensor o,
                     torch::Tensor lse, int diag) {
  using G = rep_tk::fwd_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2);
  const int NB = L / 64, NBt = tab[3].size(2);
  typename G::tgl gq{bfp(q), B, H, L, (unsigned)D};
  typename G::tgl gk{bfp(k), B, H, L, (unsigned)D};
  typename G::tgl gv{bfp(v), B, H, L, (unsigned)D};
  typename G::tgl go{bfp(o), B, H, L, (unsigned)D};
  typename G::tgl gkt{bfp(tab[3]), B, H, (unsigned)NBt, (unsigned)D};
  typename G::tgl gvt{bfp(tab[4]), B, H, (unsigned)NBt, (unsigned)D};
  G g{gq, gk, gv, go, gkt, gvt, bfp(tab[0]), bfp(tab[1]), tab[2].data_ptr<float>(), tab[5].data_ptr<float>(),
      tab[6].data_ptr<float>(), typename G::l_gl{lse.data_ptr<float>(), B, H, 1U, L}, a_ptr.data_ptr<int>(), a_idx.data_ptr<int>(), b_ptr.data_ptr<int>(),
      b_idx.data_ptr<int>(), reinterpret_cast<const uint32_t*>(bits_ab.data_ptr<int>()), blk_cnt.data_ptr<int>(),
      (int)H, (int)L, NB, NBt, RM, words_of(NB), phases, diag};
  constexpr int mem = 3 * 64 * D * 2 + 2048;  // q, k, v tiles, the lse vector + alignment slack
  static int cfg = -1;  // 0: MINB 4, else MINB 3 (default, measured best); 2-5 diagnostics
  if (cfg < 0) {
    const char* e = std::getenv("REP_TK_FWD_CFG");
    cfg = e ? std::atoi(e) : 1;
  }
  auto launch = [&](auto kern) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, mem);
    kern<<<dim3(NB, H, B), 128, mem, stream()>>>(g);
  };
  // cfg 2 / 3: tier-A-only specialisation at MINB 4 / 3 (diagnostic; valid only for phases == A)
  // cfg 4 / 5: verbatim copy of VSA2's TK forward with its own / this file's softmax (diagnostic, tier A)
  if (cfg == 4) launch(rep_tk::nat_copy_ker<D, false>);
  else if (cfg == 5) launch(rep_tk::nat_copy_ker<D, true>);
  else if (cfg == 0) launch(rep_tk::fwd_ker<D, 4>);
  else if (cfg == 2 && phases == 1) launch(rep_tk::fwd_ker<D, 4, true>);
  else if (cfg == 3 && phases == 1) launch(rep_tk::fwd_ker<D, 3, true>);
  else launch(rep_tk::fwd_ker<D, 3>);
  check_launch("fwd");
}

// tab = rep_tk_build(...). Returns o (padded query rows 0) and the natural lse (finite on padded rows).
std::vector<torch::Tensor> rep_tk_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, std::vector<torch::Tensor> tab,
                                      torch::Tensor a_ptr, torch::Tensor a_idx, torch::Tensor b_ptr,
                                      torch::Tensor b_idx, torch::Tensor bits_ab, torch::Tensor blk_cnt, int64_t RM,
                                      int64_t phases, int64_t diag) {
  check_bf16(q, "q");
  check_bf16(k, "k");
  check_bf16(v, "v");
  check_rm(RM);
  TORCH_CHECK(tab.size() == 7, "tab must be rep_tk_build(...) + [rep_tk_lcb(...)]");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(torch::kFloat32));
  if (q.size(3) == 64)
    fwd_impl<64>(q, k, v, tab, a_ptr, a_idx, b_ptr, b_idx, bits_ab, blk_cnt, (int)RM, (int)phases, o, lse, (int)diag);
  else
    fwd_impl<128>(q, k, v, tab, a_ptr, a_idx, b_ptr, b_idx, bits_ab, blk_cnt, (int)RM, (int)phases, o, lse, (int)diag);
  return {o, lse};
}


template <int D>
static void bwd_c_impl(torch::Tensor q, torch::Tensor kt, torch::Tensor vt, torch::Tensor lc_tail, torch::Tensor dout,
                       torch::Tensor lse, torch::Tensor delta, torch::Tensor bits_ab, torch::Tensor qg,
                       torch::Tensor kgt, torch::Tensor vgt) {
  using G = rep_tk::bwdc_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2), NBt = kt.size(2);
  const int NB = L / 64;
  decltype(G::q) a_q{bfp(q), B, H, L, (unsigned)D};
  decltype(G::og) a_og{bfp(dout), B, H, L, (unsigned)D};
  decltype(G::kt) a_kt{bfp(kt), B, H, NBt, (unsigned)D};
  decltype(G::vt) a_vt{bfp(vt), B, H, NBt, (unsigned)D};
  decltype(G::qg) a_qg{qg.data_ptr<float>(), B, H, L, (unsigned)D};
  decltype(G::kgt) a_kg{kgt.data_ptr<float>(), B, H, NBt, (unsigned)D};
  decltype(G::vgt) a_vg{vgt.data_ptr<float>(), B, H, NBt, (unsigned)D};
  decltype(G::l) a_l{lse.data_ptr<float>(), B, H, 1U, L};
  decltype(G::d) a_d{delta.data_ptr<float>(), B, H, 1U, L};
  // One CTA per tail tile would be NBt/64 * H * B CTAs (64 at L=16k, H=16: under half of an H100's SMs).
  // Split each tile's query range so the grid fills one wave of resident CTAs (2/SM at D=128, 3 at
  // D=64, from the launch bounds) without spilling into a second, keeping >= 16 query blocks per CTA.
  // Measured at L=16k H=16 D=128 (64 tiles, 264 slots): 1 split 0.792 ms, 4 (= one wave) 0.247 ms,
  // 8 0.254, 9 (2.2 waves) 0.313. REP_TK_BWDC_SPLITS overrides.
  int splits;
  if (const char* e = std::getenv("REP_TK_BWDC_SPLITS")) {
    splits = std::max(1, std::atoi(e));
  } else {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const long long ctas = (long long)(NBt / 64) * H * B, resident = (long long)sms * (D == 64 ? 3 : 2);
    splits = (int)std::max<long long>(1, std::min<long long>(resident / ctas, NB / 16));
  }
  G g{a_q, a_og, a_kt, a_vt, a_qg, a_kg, a_vg, a_l, a_d, lc_tail.data_ptr<float>(),
      reinterpret_cast<const uint32_t*>(bits_ab.data_ptr<int>()), (int)H, NB, (int)NBt, words_of(NB), splits};
  constexpr int mem = (D == 64) ? 72000 : 113000;  // as VSA2's TK backward
  cudaFuncSetAttribute(rep_tk::bwd_c_ker<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem);
  rep_tk::bwd_c_ker<D><<<dim3(NBt / 64, H, B * splits), 128, mem, stream()>>>(g);
  check_launch("bwd_c");
}

template <int D>
static void bwd_b_impl(torch::Tensor q, torch::Tensor k_sw, torch::Tensor v_sw, torch::Tensor lc_mid,
                       torch::Tensor dout, torch::Tensor lse, torch::Tensor delta, torch::Tensor b_ptr,
                       torch::Tensor b_idx, torch::Tensor qg, torch::Tensor dk_mid, torch::Tensor dv_mid, int RM) {
  // lc_mid here is the flattened per-entry table from rep_tk_lcb
  using G = rep_tk::bwdb_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2);
  const int NB = L / 64;
  decltype(G::q) a_q{bfp(q), B, H, L, (unsigned)D};
  decltype(G::og) a_og{bfp(dout), B, H, L, (unsigned)D};
  decltype(G::qg) a_qg{qg.data_ptr<float>(), B, H, L, (unsigned)D};
  G g{a_q, a_og, a_qg, bfp(k_sw), bfp(v_sw), lc_mid.data_ptr<float>(), lse.data_ptr<float>(),
      delta.data_ptr<float>(), dk_mid.data_ptr<float>(), dv_mid.data_ptr<float>(), b_ptr.data_ptr<int>(),
      b_idx.data_ptr<int>(), (int)H, (int)L, NB, RM};
  constexpr int mem = 4 * 64 * D * 2 + 2 * 64 * 64 * 2 + 1024;
  cudaFuncSetAttribute(rep_tk::bwd_b_ker<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem);
  rep_tk::bwd_b_ker<D><<<dim3(NB, H, B), 128, mem, stream()>>>(g);
  check_launch("bwd_b");
}

// Tier C, KV-owner. dout must be the masked dO (0 on padded query rows). qg (fp32 [B,H,L,D]) receives
// dQ by reduce-add; kgt/vgt (fp32 [B,H,NBt,D], zeroed) receive the block-mean gradients.
void rep_tk_bwd_c(torch::Tensor q, torch::Tensor kt, torch::Tensor vt, torch::Tensor lc_tail, torch::Tensor dout,
                  torch::Tensor lse, torch::Tensor delta, torch::Tensor bits_ab, torch::Tensor qg, torch::Tensor kgt,
                  torch::Tensor vgt) {
  check_bf16(q, "q");
  check_bf16(dout, "dout");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  if (q.size(3) == 64) bwd_c_impl<64>(q, kt, vt, lc_tail, dout, lse, delta, bits_ab, qg, kgt, vgt);
  else bwd_c_impl<128>(q, kt, vt, lc_tail, dout, lse, delta, bits_ab, qg, kgt, vgt);
}

// Tier B, query-owner. dk_mid/dv_mid: fp32 [B,H,NB*RM,D], zeroed (bulk reduce-add).
void rep_tk_bwd_b(torch::Tensor q, torch::Tensor k_sw, torch::Tensor v_sw, torch::Tensor lc_mid, torch::Tensor dout,
                  torch::Tensor lse, torch::Tensor delta, torch::Tensor b_ptr, torch::Tensor b_idx, torch::Tensor qg,
                  torch::Tensor dk_mid, torch::Tensor dv_mid, int64_t RM) {
  check_bf16(q, "q");
  check_bf16(dout, "dout");
  check_rm(RM);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  if (q.size(3) == 64) bwd_b_impl<64>(q, k_sw, v_sw, lc_mid, dout, lse, delta, b_ptr, b_idx, qg, dk_mid, dv_mid, (int)RM);
  else bwd_b_impl<128>(q, k_sw, v_sw, lc_mid, dout, lse, delta, b_ptr, b_idx, qg, dk_mid, dv_mid, (int)RM);
}

// ---------------------------------------------------------------- VSA2's own TK fine kernels
// Compiled in from csrc/vsa/sm90_tk/fine_tk_h100.cu unchanged: the native baseline, and tier A's backward.
void vsa2_tk_fine_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                      torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt);
void vsa2_tk_fine_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                      torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                      torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, int opts);

std::vector<torch::Tensor> tk_native_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor row_ptr,
                                         torch::Tensor col_idx, torch::Tensor blk_cnt) {
  check_bf16(q, "q");
  auto o = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(torch::kFloat32));
  vsa2_tk_fine_fwd(q, k, v, o, lse, row_ptr, col_idx, blk_cnt);
  check_launch("native fwd");
  return {o, lse};
}

void tk_native_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                   torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                   torch::Tensor qg, torch::Tensor kg, torch::Tensor vg) {
  check_bf16(q, "q");
  vsa2_tk_fine_bwd(q, k, v, dout_f, lse, delta, row_ptr, col_idx, blk_cnt, qg, kg, vg, 0);
  check_launch("native bwd");
}
