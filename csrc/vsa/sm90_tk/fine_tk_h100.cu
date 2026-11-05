// VSA2 fine branch, H100 (sm_90a) ThunderKittens wgmma + TMA kernels.
//
// Adapted from FastVideo fastvideo-kernel/csrc/attention/block_sparse_h100.cu
// (https://github.com/hao-ai-lab/FastVideo @ 661d18ba55ddd2fde1802960cf4343a2ca84f57e,
// Apache License 2.0, Copyright the FastVideo authors), built against
// ThunderKittens @ 6c27e28c8115d1839d9eeeb530913c184a75fc87 (MIT).
//
// Changes relative to the original VSA kernels (marked "VSA2:"):
//   * fixed-width q2k/k2q index tables -> CSR lists (row_ptr/col_idx), which hold
//     the variable per-query-block counts produced by global top-k
//   * zero-edge query blocks (empty list): o = 0, lse = -inf, no TMA issued
//     (the original kernel unconditionally read index[0] / index[-1])
//   * lse is stored as the natural log-sum-exp of scale*q.k (the original stored
//     -lse*sqrt(D)); the backward converts on load
//   * the backward prep kernel is removed: delta (and the masked dO) come from
//     the VSA2 fusion-backward kernel, which also folds in the gate
//   * MHA only (hr = 1), batch handled through gridDim.z
// The math, tiling, pipelining and TMA usage are otherwise unchanged.
//
// Split-KV task mode (TASK = true, experimental, plan from csrc/vsa/split.cu): a 1-D grid over planned
// tasks, each one query block (forward) or kv block (backward) and a contiguous range of its CSR list.
//   forward : direct tasks finish as above. Split chunks store the fp32 normalized partial (TMA, through
//             the K/V shared memory) and its log2-domain lse; split_merge_fwd (mask_rows = 0, keeping TK's
//             finite padded rows) writes O_f and the global lse.
//   backward: every task reduce-adds its dQ, dK, dV into the fp32 accumulators as before, so chunks of a
//             split kv block sum in the accumulator (fp32, order not fixed). No partial slots are needed.
//
// Backward runtime options (bwd_globals::opts, 0 = the kernel above unchanged):
//   kDqReadWait (1): after each dQ reduce-add wait only until the shared tile has been read
//                    (cp.async.bulk.wait_group.read) instead of until the reduction completed in global
//                    memory. qg_smem reuse needs no more; the final store_async_wait still drains all.
//   kRotate     (2): each CTA walks its k2q list from a hashed start offset, so CTAs of one head with long
//                    lists (hot kv columns) do not reduce into the same dQ tiles in lockstep. Same pairs; only
//                    the fp32 summation order of dK/dV within a CTA and of dQ across CTAs changes.
//   kSkipDq     (4): DIAGNOSTIC ONLY, dQ is not written (wrong gradients): measures the dQ reduce cost.

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

using namespace kittens;

#if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define VSA2_TK_HOPPER 1
#else
#define VSA2_TK_HOPPER 0
#endif

namespace vsa2_tk {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;

template <int D>
struct fwd_attend_ker_tile_dims {
  constexpr static int tile_width = D;
  constexpr static int qo_height = 4 * 16;
  constexpr static int kv_height = 4 * 16;
};

template <int D>
struct fwd_globals {
  using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
  using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  l_gl l;
  o_gl o;

  const int L;
  const int NB;
  const int* __restrict__ row_ptr;  // VSA2: CSR q->kv
  const int* __restrict__ col_idx;
  const int* __restrict__ block_size;
  // split-KV task mode (TASK = true only)
  using p_tile = st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
  using p_gl = gl<float, -1, -1, -1, -1, p_tile>;
  p_gl po;                       // [1, 1, slots*64, D] fp32 partials
  float* __restrict__ part_lse;  // [slots, 64] log2 domain
  const int H;
  const int* __restrict__ tasks;
  const int* __restrict__ task_info;
};

template <int D, bool TASK>
__global__ __launch_bounds__(128, 4) void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g) {
#if VSA2_TK_HOPPER
  constexpr float SCALE_LOG2 = (D == 64) ? 1.44269504089f * 0.125f : 1.44269504089f * 0.08838834764f;
  int b, h, seq_idx, beg, kv_blocks, slot = -1;
  if constexpr (TASK) {
    if ((int)blockIdx.x >= g.task_info[0]) return;
    const int4 tk = reinterpret_cast<const int4*>(g.tasks)[blockIdx.x];
    const int r_bh = tk.x / g.NB;
    seq_idx = tk.x - r_bh * g.NB;
    b = r_bh / g.H;
    h = r_bh - b * g.H;
    beg = tk.y;
    kv_blocks = tk.z - tk.y;
    slot = tk.w;
  } else {
    b = blockIdx.z;
    h = blockIdx.y;
    seq_idx = blockIdx.x;
    // VSA2: CSR row and zero-edge handling
    const int* rp = g.row_ptr + ((size_t)b * g.H + h) * (g.NB + 1) + seq_idx;
    beg = rp[0];
    kv_blocks = rp[1] - beg;
  }
  const int bh = b * g.H + h;
  const int* q2k_block_sparse_index_ptr = g.col_idx + beg;
  if (kv_blocks <= 0) {
    bf16* o = g.o.raw_ptr + ((size_t)bh * g.L + (size_t)seq_idx * BLOCK_M) * D;
    for (int i = threadIdx.x; i < BLOCK_M * D / 8; i += blockDim.x) reinterpret_cast<uint4*>(o)[i] = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < BLOCK_M) g.l.raw_ptr[(size_t)bh * g.L + (size_t)seq_idx * BLOCK_M + threadIdx.x] = -INFINITY;
    return;
  }

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);

  using K = fwd_attend_ker_tile_dims<D>;
  using q_tile = st_bf<64, K::tile_width>;
  using k_tile = st_bf<64, K::tile_width>;
  using v_tile = st_bf<64, K::tile_width>;
  using l_col_vec = col_vec<st_fl<64, K::tile_width>>;
  using o_tile = st_bf<64, K::tile_width>;

  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();
  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();
  l_col_vec(&l_smem)[1] = al.allocate<l_col_vec, 1>();
  auto(*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

  const int kv_head_idx = h;  // MHA

  __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived, v_smem_arrived;
  if (threadIdx.x == 0) {
    int32_t kv_block_index = q2k_block_sparse_index_ptr[0];
    init_semaphore(qsmem_semaphore, 0, 1);
    init_semaphore(k_smem_arrived, 0, 1);
    init_semaphore(v_smem_arrived, 0, 1);

    coord<q_tile> q_tile_idx = {b, h, seq_idx, 0};
    tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
    tma::load_async(q_smem[0], g.q, q_tile_idx, qsmem_semaphore);

    tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
    coord<k_tile> k_tile_idx = {b, kv_head_idx, kv_block_index, 0};
    tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

    tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
    coord<v_tile> v_tile_idx = {b, kv_head_idx, kv_block_index, 0};
    tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
  }
  __syncthreads();

  rt_fl<16, 64> att_block;
  rt_bf<16, 64> att_block_mma;
  rt_fl<16, K::tile_width> o_reg;
  col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

  neg_infty(max_vec);
  zero(norm_vec);
  zero(o_reg);

  wait(qsmem_semaphore, 0);

  for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
    const bool has_next = kv_idx + 1 < kv_blocks;
    int32_t kv_block_index = has_next ? q2k_block_sparse_index_ptr[kv_idx + 1] : 0;

    wait(k_smem_arrived, kv_idx % 2);
    warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

    copy(max_vec_last_scaled, max_vec);
    mul(max_vec_last_scaled, max_vec_last_scaled, SCALE_LOG2);
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
      coord<k_tile> k_tile_idx = {b, kv_head_idx, kv_block_index, 0};
      tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);
    }

    right_fill(att_block, att_block, g.block_size[q2k_block_sparse_index_ptr[kv_idx]],
               base_types::constants<float>::neg_infty());
    row_max(max_vec, att_block, max_vec);
    mul(att_block, att_block, SCALE_LOG2);
    mul(max_vec_scaled, max_vec, SCALE_LOG2);

    sub_row(att_block, att_block, max_vec_scaled);
    exp2(att_block, att_block);
    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    exp2(max_vec_last_scaled, max_vec_last_scaled);
    mul(norm_vec, norm_vec, max_vec_last_scaled);
    row_sum(norm_vec, att_block, norm_vec);
    add(att_block, att_block, 0.f);
    copy(att_block_mma, att_block);
    mul_row(o_reg, o_reg, max_vec_last_scaled);

    wait(v_smem_arrived, kv_idx % 2);
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
      coord<v_tile> v_tile_idx = {b, kv_head_idx, kv_block_index, 0};
      tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
    }
  }

  div_row(o_reg, o_reg, norm_vec);

  if constexpr (TASK) {
    if (slot >= 0) {  // split chunk: fp32 normalized partial + log2-domain lse
      using p_tile = typename fwd_globals<D>::p_tile;
      // K/V tiles are idle (no load in flight after the last chunk); together they hold one fp32 tile,
      // the same reuse as kg_smem in the backward
      p_tile(*p_smem) = reinterpret_cast<p_tile*>(&k_smem[0].data[0]);
      __syncthreads();
      warpgroup::store(p_smem[0], o_reg);
      mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
      log(norm_vec, norm_vec);
      add(norm_vec, norm_vec, max_vec_scaled);
      mul(norm_vec, norm_vec, 1.44269504089f);  // natural -> log2
      warpgroup::store(l_smem[0], norm_vec);
      __syncthreads();
      if (threadIdx.x / 32 == 0) {
        coord<p_tile> p_tile_idx = {0, 0, slot, 0};
        tma::store_async(g.po, p_smem[0], p_tile_idx);
      }
      if (threadIdx.x < BLOCK_M) g.part_lse[(size_t)slot * BLOCK_M + threadIdx.x] = l_smem[0][threadIdx.x];
      tma::store_async_wait();
      return;
    }
  }

  warpgroup::store(o_smem[0], o_reg);
  __syncthreads();

  if (threadIdx.x / 32 == 0) {
    coord<o_tile> o_tile_idx = {b, h, seq_idx, 0};
    tma::store_async(g.o, o_smem[0], o_tile_idx);
  }

  // VSA2: natural lse = max*scale + ln(norm)
  mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
  log(norm_vec, norm_vec);
  add(norm_vec, norm_vec, max_vec_scaled);

  warpgroup::store(l_smem[0], norm_vec);
  __syncthreads();

  if (threadIdx.x / 32 == 0) {
    coord<l_col_vec> tile_idx = {b, h, 0, seq_idx};
    tma::store_async(g.l, l_smem[0], tile_idx);
  }
  tma::store_async_wait();
#endif
}

template <int D>
struct bwd_attend_ker_tile_dims {
  constexpr static int tile_width = D;
  constexpr static int tile_h = 4 * 16;
  constexpr static int tile_h_qo = 4 * 16;
};

template <int D>
struct bwd_globals {
  using G = bwd_attend_ker_tile_dims<D>;
  using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using k_tile = st_bf<G::tile_h, G::tile_width>;
  using v_tile = st_bf<G::tile_h, G::tile_width>;
  using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using kg_tile = st_fl<G::tile_h, G::tile_width>;
  using vg_tile = st_fl<G::tile_h, G::tile_width>;
  using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using og_gl = gl<bf16, -1, -1, -1, -1, og_tile>;
  using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
  using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
  using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;
  using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
  using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  og_gl og;
  qg_gl qg;
  kg_gl kg;
  vg_gl vg;
  l_gl l;
  d_gl d;

  const int NB;
  const int* __restrict__ row_ptr;  // VSA2: CSR kv->q
  const int* __restrict__ col_idx;
  const int* __restrict__ block_size;
  const int H;
  const int* __restrict__ tasks;  // split-KV task mode (TASK = true only)
  const int* __restrict__ task_info;
  const int opts;                 // kDqReadWait | kRotate | kSkipDq
};

constexpr int kDqReadWait = 1, kRotate = 2, kSkipDq = 4;

__device__ static inline void stream_tile(auto& reg_tile, auto& smem_vec, int tic) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int base_col = 16 * i + 2 * (kittens::laneid() % 4);
    reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
    reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
  }
}

__device__ static inline void stream_sub_tile(auto& reg_tile, auto& smem_vec, int tic) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int base_col = 16 * i + 2 * (laneid() % 4);
    reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
    reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
  }
}

template <int D, bool TASK>
__global__ __launch_bounds__(128, (D == 64) ? 3 : 2) void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g) {
#if VSA2_TK_HOPPER
  constexpr float SCALE = (D == 64) ? 0.125f : 0.08838834764f;
  constexpr float SCALE_LOG2 = 1.44269504089f * SCALE;
  constexpr float NEG_INV_SCALE = (D == 64) ? -8.0f : -11.313708499f;  // VSA2: natural lse -> -lse*sqrt(D)

  int b, h, kb, beg, qo_blocks;
  if constexpr (TASK) {
    if ((int)blockIdx.x >= g.task_info[0]) return;
    const int4 tk = reinterpret_cast<const int4*>(g.tasks)[blockIdx.x];
    const int r_bh = tk.x / g.NB;
    kb = tk.x - r_bh * g.NB;
    b = r_bh / g.H;
    h = r_bh - b * g.H;
    beg = tk.y;
    qo_blocks = tk.z - tk.y;
  } else {
    b = blockIdx.z;
    h = blockIdx.y;
    kb = blockIdx.x;
    const int* rp = g.row_ptr + ((size_t)b * g.H + h) * (g.NB + 1) + kb;
    beg = rp[0];
    qo_blocks = rp[1] - beg;
  }
  const int* k2q_block_sparse_index_ptr = g.col_idx + beg;
  if (qo_blocks <= 0) return;  // dK/dV accumulators stay zero
  // kRotate: list position i -> (i + rot) mod qo_blocks
  int rot = 0;
  if ((g.opts & kRotate) && qo_blocks > 1)
    rot = (int)(((unsigned)kb * 2654435761u + (unsigned)h * 40503u + (unsigned)beg * 2246822519u) >> 8) % qo_blocks;
  auto q_at = [&](int i) {
    int j = i + rot;
    if (j >= qo_blocks) j -= qo_blocks;
    return k2q_block_sparse_index_ptr[j];
  };

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);

  using G = bwd_attend_ker_tile_dims<D>;
  using kg_tile = st_fl<G::tile_h, G::tile_width>;
  using vg_tile = st_fl<G::tile_h, G::tile_width>;
  using k_tile = st_bf<G::tile_h, G::tile_width>;
  using v_tile = st_bf<G::tile_h, G::tile_width>;
  using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();
  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();
  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
  og_tile(&og_smem)[1] = al.allocate<og_tile, 1>();
  qg_tile(&qg_smem) = al.allocate<qg_tile>();
  l_tile(&l_smem)[1] = al.allocate<l_tile, 1>();
  d_tile(&d_smem)[1] = al.allocate<d_tile, 1>();
  kg_tile(*kg_smem) = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
  vg_tile(*vg_smem) = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);
  attn_tile(&ds_smem_t)[1] = al.allocate<attn_tile, 1>();

  const int kv_head_idx = h;  // MHA

  __shared__ kittens::semaphore kv_b, q_b[1], o_b[1], vec_b[1];
  int32_t store_qg_block_index;
  int32_t load_q_block_index = q_at(0);

  if (threadIdx.x == 0) {
    init_semaphore(kv_b, 0, 1);
    init_semaphore(q_b[0], 0, 1);
    init_semaphore(o_b[0], 0, 1);
    init_semaphore(vec_b[0], 0, 1);

    tma::expect_bytes(kv_b, sizeof(k_smem[0]) + sizeof(v_smem[0]));
    coord<k_tile> tile_idx_kv = {b, kv_head_idx, kb, 0};
    tma::load_async(k_smem[0], g.k, tile_idx_kv, kv_b);
    tma::load_async(v_smem[0], g.v, tile_idx_kv, kv_b);

    coord<q_tile> tile_idx_qo = {b, h, load_q_block_index, 0};
    coord<l_tile> vec_idx = {b, h, 0, load_q_block_index};

    tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
    tma::load_async(og_smem[0], g.og, tile_idx_qo, o_b[0]);

    tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
    tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
    tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);

    tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
    tma::load_async(q_smem[0], g.q, tile_idx_qo, q_b[0]);
  }
  __syncthreads();

  rt_fl<16, G::tile_width> kg_reg, vg_reg;
  rt_fl<16, 64> s_block_t, p_block_t;
  rt_fl<16, 64> ds_block_t, dp_block_t;
  rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

  zero(kg_reg);
  zero(vg_reg);

  wait(kv_b, 0);
  int fill_start = g.block_size[kb] - 16 * kittens::warpid();

  for (int qo_idx = 0; qo_idx < qo_blocks; qo_idx++) {
    const bool has_next = qo_idx + 1 < qo_blocks;
    store_qg_block_index = load_q_block_index;
    if (has_next) load_q_block_index = q_at(qo_idx + 1);

    wait(o_b[0], qo_idx % 2);
    warpgroup::mm_ABt(dp_block_t, v_smem[0], og_smem[0]);  // dP^T = V dO^T
    warpgroup::mma_commit_group();

    wait(vec_b[0], qo_idx % 2);
    stream_tile(s_block_t, l_smem, 0);
    mul(s_block_t, s_block_t, NEG_INV_SCALE);  // VSA2: -lse/scale
    wait(q_b[0], qo_idx % 2);
    warpgroup::mma_ABt(s_block_t, k_smem[0], q_smem[0]);  // S^T = K Q^T - lse/scale
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    mul(s_block_t, s_block_t, SCALE_LOG2);
    lower_fill(s_block_t, s_block_t, fill_start, base_types::constants<float>::neg_infty());
    exp2(s_block_t, s_block_t);
    copy(p_block_t, s_block_t);
    copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, 0);  // dP - delta
    mul(ds_block_t, p_block_t, dp_block_t);  // dS = P * (dP - delta)
    mul(ds_block_t, ds_block_t, SCALE);

    if (has_next && threadIdx.x == 0) {
      coord<l_tile> vec_idx = {b, h, 0, load_q_block_index};
      tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
      tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
      tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);
    }

    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[0]);  // dV += P^T dO
    warpgroup::mma_commit_group();
    copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem_t[0], ds_block_t);
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      coord<q_tile> tile_idx = {b, h, load_q_block_index, 0};
      tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
      tma::load_async(og_smem[0], g.og, tile_idx, o_b[0]);
    }

    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[0]);  // dK += dS^T Q
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    if (has_next && threadIdx.x == 0) {
      coord<q_tile> q_tile_idx = {b, h, load_q_block_index, 0};
      tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
      tma::load_async(q_smem[0], g.q, q_tile_idx, q_b[0]);
    }

    rt_fl<16, G::tile_width> qg_reg;
    __syncthreads();
    warpgroup::mm_AtB(qg_reg, ds_smem_t[0], k_smem[0]);  // dQ = dS K
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    warpgroup::store(qg_smem, qg_reg);
    __syncthreads();

    if (threadIdx.x / 32 == 0 && !(g.opts & kSkipDq)) {
      coord<qg_tile> tile_idx = {b, h, store_qg_block_index, 0};
      tma::store_add_async(g.qg, qg_smem, tile_idx);
      if (g.opts & kDqReadWait) tma::store_async_read_wait();  // qg_smem reusable; reduction still in flight
      else tma::store_async_wait();
    }
  }

  __syncthreads();
  warpgroup::store(kg_smem[0], kg_reg);
  __syncthreads();
  if (threadIdx.x / 32 == 0) {
    coord<kg_tile> tile_idx = {b, kv_head_idx, kb, 0};
    tma::store_add_async(g.kg, kg_smem[0], tile_idx);
    tma::store_commit_group();
  }

  warpgroup::store(vg_smem[0], vg_reg);
  __syncthreads();
  if (kittens::warpid() % 4 == 0) {
    coord<vg_tile> tile_idx = {b, kv_head_idx, kb, 0};
    tma::store_add_async(g.vg, vg_smem[0], tile_idx);
    tma::store_commit_group();
  }
  tma::store_async_wait();
#endif
}

// ------------------------------------------------------------------ host

// Unsplit: tasks == nullptr, grid (NB, H, B). Task mode: 1-D grid of n_tasks (split.cu plan); part_o
// ([part_rows / 64 >= 1, 64, D] fp32) and part_lse receive the split chunks.
template <int D>
static void tk_fwd_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                        const int* row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt, const int* tasks,
                        const int* task_info, int n_tasks, float* part_o, unsigned part_rows, float* part_lse) {
  using G = fwd_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2);
  bf16* dq = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
  bf16* dk = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
  bf16* dv = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
  bf16* d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
  float* dl = lse.data_ptr<float>();
  typename G::q_gl qg{dq, B, H, L, (unsigned)D};
  typename G::k_gl kg{dk, B, H, L, (unsigned)D};
  typename G::v_gl vg{dv, B, H, L, (unsigned)D};
  typename G::l_gl lg{dl, B, H, 1U, L};
  typename G::o_gl og{d_o, B, H, L, (unsigned)D};
  // Unsplit: the kernel never touches po; it still needs a well-formed descriptor (64 x D over o's address).
  typename G::p_gl pg{tasks ? part_o : reinterpret_cast<float*>(d_o), 1U, 1U, tasks ? part_rows : 64U,
                      (unsigned)D};
  const int NB = L / BLOCK_M;
  G g{qg, kg, vg, lg, og, (int)L, NB, row_ptr, col_idx.data_ptr<int>(), blk_cnt.data_ptr<int>(),
      pg, part_lse, (int)H, tasks, task_info};
  constexpr int mem_size = 54000;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (tasks) {
    cudaFuncSetAttribute(fwd_attend_ker<D, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (n_tasks > 0) fwd_attend_ker<D, true><<<n_tasks, 128, mem_size, stream>>>(g);
    return;
  }
  cudaFuncSetAttribute(fwd_attend_ker<D, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  dim3 grid(NB, H, B);
  fwd_attend_ker<D, false><<<grid, 128, mem_size, stream>>>(g);
}

template <int D>
static void tk_bwd_impl(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                        torch::Tensor delta, const int* row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                        torch::Tensor qg_acc, torch::Tensor kg_acc, torch::Tensor vg_acc, const int* tasks,
                        const int* task_info, int n_tasks, int opts) {
  using G = bwd_globals<D>;
  const unsigned B = q.size(0), H = q.size(1), L = q.size(2);
  typename G::q_gl a_q{reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::k_gl a_k{reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::v_gl a_v{reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::og_gl a_og{reinterpret_cast<bf16*>(dout_f.data_ptr<c10::BFloat16>()), B, H, L, (unsigned)D};
  typename G::qg_gl a_qg{qg_acc.data_ptr<float>(), B, H, L, (unsigned)D};
  typename G::kg_gl a_kg{kg_acc.data_ptr<float>(), B, H, L, (unsigned)D};
  typename G::vg_gl a_vg{vg_acc.data_ptr<float>(), B, H, L, (unsigned)D};
  typename G::l_gl a_l{lse.data_ptr<float>(), B, H, 1U, L};
  typename G::d_gl a_d{delta.data_ptr<float>(), B, H, 1U, L};
  const int NB = L / BLOCK_N;
  G g{a_q, a_k, a_v, a_og, a_qg, a_kg, a_vg, a_l, a_d, NB,
      row_ptr, col_idx.data_ptr<int>(), blk_cnt.data_ptr<int>(), (int)H, tasks, task_info, opts};
  constexpr int mem_size = (D == 64) ? 72000 : 113000;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (tasks) {
    cudaFuncSetAttribute(bwd_attend_ker<D, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    if (n_tasks > 0) bwd_attend_ker<D, true><<<n_tasks, 128, mem_size, stream>>>(g);
    return;
  }
  cudaFuncSetAttribute(bwd_attend_ker<D, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
  dim3 grid(NB, H, B);
  bwd_attend_ker<D, false><<<grid, 128, mem_size, stream>>>(g);
}

}  // namespace vsa2_tk

// q/k/v/o: [B, H, L, D] bf16; lse [B, H, L] fp32 (written)
void vsa2_tk_fine_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                      torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt) {
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int* rp = row_ptr.data_ptr<int>();
  if (q.size(3) == 64)
    vsa2_tk::tk_fwd_impl<64>(q, k, v, o, lse, rp, col_idx, blk_cnt, nullptr, nullptr, 0, nullptr, 0, nullptr);
  else
    vsa2_tk::tk_fwd_impl<128>(q, k, v, o, lse, rp, col_idx, blk_cnt, nullptr, nullptr, 0, nullptr, 0, nullptr);
}

// Split-KV task mode (tasks/info from split_plan over q2k). part_o [slots >= 1, 64, D] fp32, part_lse [slots, 64].
void vsa2_tk_fine_fwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                            torch::Tensor col_idx, torch::Tensor blk_cnt, torch::Tensor tasks, torch::Tensor info,
                            torch::Tensor part_o, torch::Tensor part_lse) {
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int* tp = tasks.data_ptr<int>();
  const int* ip = info.data_ptr<int>();
  const int n = (int)tasks.size(0);
  const unsigned rows = (unsigned)(part_o.size(0) * part_o.size(1));
  float* po = part_o.data_ptr<float>();
  float* pl = part_lse.data_ptr<float>();
  if (q.size(3) == 64)
    vsa2_tk::tk_fwd_impl<64>(q, k, v, o, lse, nullptr, col_idx, blk_cnt, tp, ip, n, po, rows, pl);
  else
    vsa2_tk::tk_fwd_impl<128>(q, k, v, o, lse, nullptr, col_idx, blk_cnt, tp, ip, n, po, rows, pl);
}

// qg/kg/vg: fp32 [B, H, L, D] zero-initialized accumulators (written with TMA reduce-add)
void vsa2_tk_fine_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                      torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                      torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, int opts) {
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int* rp = row_ptr.data_ptr<int>();
  if (q.size(3) == 64)
    vsa2_tk::tk_bwd_impl<64>(q, k, v, dout_f, lse, delta, rp, col_idx, blk_cnt, qg, kg, vg, nullptr, nullptr, 0, opts);
  else
    vsa2_tk::tk_bwd_impl<128>(q, k, v, dout_f, lse, delta, rp, col_idx, blk_cnt, qg, kg, vg, nullptr, nullptr, 0, opts);
}

// Split-KV task mode over k2q (tasks/info from split_plan); same zeroed fp32 accumulators as vsa2_tk_fine_bwd.
void vsa2_tk_fine_bwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f,
                            torch::Tensor lse, torch::Tensor delta, torch::Tensor col_idx, torch::Tensor blk_cnt,
                            torch::Tensor tasks, torch::Tensor info, torch::Tensor qg, torch::Tensor kg,
                            torch::Tensor vg, int opts) {
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int* tp = tasks.data_ptr<int>();
  const int* ip = info.data_ptr<int>();
  const int n = (int)tasks.size(0);
  if (q.size(3) == 64)
    vsa2_tk::tk_bwd_impl<64>(q, k, v, dout_f, lse, delta, nullptr, col_idx, blk_cnt, qg, kg, vg, tp, ip, n, opts);
  else
    vsa2_tk::tk_bwd_impl<128>(q, k, v, dout_f, lse, delta, nullptr, col_idx, blk_cnt, qg, kg, vg, tp, ip, n, opts);
}
