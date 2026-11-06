// Shared device helpers for the three-tier representative kernels: the gathered tile load and the
// column stream that walks tier A (CSR list), tier B (bits_ab & ~bits_a) and tier C (~bits_ab).
//
// Both attention kernels consume one 64-column tile per iteration whatever its tier, so the MMA and
// online-softmax bodies of VSA2's own kernels are reused with one change: every column carries an
// additive log2-domain bias -- 0 for a real token, log2(c) for a representative of c tokens, -inf for
// a masked or absent column.
//
// Producing a tile's column metadata is split so that tier A, which does the bulk of the work, pays
// almost nothing for the generalization:
//   * ColStream::next() runs on thread 0 and only *decides* the tile: its tier, and either the kv
//     block id (tier A) or the list of block ids the tile packs (tiers B and C). It writes a handful
//     of words, never 64 of them, and the barrier the pipeline already has publishes them.
//   * expand_tile() then runs on all threads, and only for a gathered tier, turning those block ids
//     into 64 table rows and 64 biases. Tier A skips it entirely and reads its mask straight from the
//     block's valid-token count, so it needs no index and no bias array at all.
#pragma once

#include "common.cuh"
#include "rep_kernels.h"

namespace vsa2rep {

using vsa2::cp_async_16;
using vsa2::neg_inf;
using vsa2::smem_u32;
using vsa2::swz;

// Tile tiers, as published in sMeta[0].
enum { kTileA = 0, kTileB = 1, kTileC = 2, kTileNone = -1 };
// Per-tile metadata words (stride kMetaStride): tier, tier-A kv block id, tier-A valid tokens or the
// number of block ids a gathered tile packs.
enum { kMetaTier = 0, kMetaBlk = 1, kMetaCnt = 2, kMetaStride = 4 };

// Asynchronously gather a [ROWS][COLS] 16-bit tile whose row r is row sIdx[r] of a row-major table
// (row stride COLS). sIdx[r] < 0 zero-fills without reading global memory.
// Chunk mapping matches VSA2's contiguous loader: consecutive threads copy consecutive 16-byte
// chunks, so a warp's accesses coalesce. Each chunk reads its row's table index from shared memory;
// a mapping that reads the index once per thread instead was measured and is slower, because it
// breaks that coalescing (forward A|B|C 1.347 -> 1.442 ms at L=16384, H=16, D=128).
template <int ROWS, int COLS, int NTHREADS, typename Elt>
__device__ __forceinline__ void load_tile_gather_async(Elt* smem, const Elt* table, const int* sIdx, int tid) {
  constexpr int CHUNKS_PER_ROW = COLS / 8;
  constexpr int TOTAL = ROWS * CHUNKS_PER_ROW;
  constexpr int PER = TOTAL / NTHREADS;
  static_assert(TOTAL % NTHREADS == 0, "tile chunks must divide threads");
  // The table indices are read first, into registers, and only then are the copies issued: cp.async
  // is inline `asm volatile`, so a read left between two copies cannot be hoisted above them and its
  // latency is exposed once per chunk. Measured worth 11% of the gathered pass.
  int src[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) src[i] = sIdx[(tid + i * NTHREADS) / CHUNKS_PER_ROW];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int id = tid + i * NTHREADS;
    const int row = id / CHUNKS_PER_ROW;
    const int ch = id % CHUNKS_PER_ROW;
    const bool ok = src[i] >= 0;
    const Elt* p = table + (size_t)(ok ? src[i] : 0) * COLS + ch * 8;
    cp_async_16(smem_u32(smem + swz<COLS>(row, ch * 8)), p, ok);
  }
}

// Collect up to `need` kv block ids, from `cursor` onward, whose derived selection bit is set.
// mode 0 = tier B (in A|B but not in A), mode 1 = tier C (in neither).
//
// Run by all 32 lanes of one warp, each taking one 32-block word, because this scan is the only work
// in the tile pipeline that is not already spread over the CTA: done by a single thread it adds up to
// 64 dependent __ffs iterations that every other thread then waits for at the next barrier. Every
// lane keeps identical cursor/phase state, so no broadcast is needed between calls.
//
// Returns the number written (same on every lane) and leaves `cursor` just past the last one.
__device__ __forceinline__ int collect_tier(const uint32_t* row_a, const uint32_t* row_ab, int mode, int NB,
                                            int& cursor, int need, int* out, int lane) {
  int cur = cursor, n = 0;
  while (n < need && cur < NB) {
    const int w0 = cur >> 5;
    const int w = w0 + lane;
    const int base = w << 5;
    uint32_t word = 0;
    if (base < NB) {
      word = (mode == 0) ? (row_ab[w] & ~row_a[w]) : ~row_ab[w];
      if (lane == 0) {
        const int lo = cur & 31;
        if (lo) word &= 0xffffffffu << lo;   // bits before the cursor
      }
      const int hi = NB - base;              // bits of this word that are real columns
      if (hi < 32) word &= (1u << hi) - 1u;
    }
    const int pc = __popc(word);
    int incl = pc;                           // inclusive prefix of popc over the lanes
#pragma unroll
    for (int s = 1; s < 32; s <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, incl, s);
      if (lane >= s) incl += y;
    }
    const int total = __shfl_sync(0xffffffffu, incl, 31);
    const int excl = incl - pc;
    const int room = need - n;
    int keep = room - excl;                  // how many of this lane's bits still fit
    keep = keep < 0 ? 0 : (keep > pc ? pc : keep);
    uint32_t m = word;
    for (int i = 0; i < keep; ++i) {
      const int b = __ffs(m) - 1;
      out[n + excl + i] = base + b;
      m &= m - 1;
    }
    if (total <= room) {                     // the whole 32-word window fit
      cur = (w0 + 32) << 5;
      n += total;
    } else {                                 // stop just past the room-th bit of the window
      int mycur = -1;
      if (excl < room && room <= excl + pc) {
        uint32_t m2 = word;
        for (int i = 0; i < room - excl - 1; ++i) m2 &= m2 - 1;
        mycur = base + (__ffs(m2) - 1) + 1;
      }
#pragma unroll
      for (int s = 16; s; s >>= 1) mycur = max(mycur, __shfl_xor_sync(0xffffffffu, mycur, s));
      cur = mycur;
      n += room;
    }
  }
  cursor = cur > NB ? NB : cur;
  return n;
}

// Per-(bh, query block) cursor over the three tiers. Advanced by warp 0.
struct ColStream {
  const int* a_idx;        // tier-A kv block list (CSR row)
  const uint32_t* row_a;   // bits_a  of this query block
  const uint32_t* row_ab;  // bits_ab of this query block
  const int* blk_cnt;
  int a_n, NB, RM, phases, diag;
  int a_i, b_blk, c_blk, phase;

  __device__ __forceinline__ void init(int tier_a_n, int nb, int rm, int ph, int dg = 0) {
    a_n = tier_a_n;
    NB = nb;
    RM = rm;
    phases = ph;
    diag = dg;
    a_i = b_blk = c_blk = 0;
    phase = 0;
  }

  // Decides the next tile. Writes sMeta (tier, block id, count) and, for a gathered tier, the block
  // ids it packs into sCol. Run by all 32 lanes of warp 0 (the bit scan is lane-parallel); the
  // caller's next __syncthreads() publishes the result to the rest of the CTA.
  __device__ void next(int* sMeta, int* sCol, int lane) {
    int tier = kTileNone, cnt = 0, blk = -1;
    while (phase < 3) {
      if (phase == 0) {
        if ((phases & kPhaseA) && a_i < a_n) {
          blk = a_idx[a_i++];
          tier = kTileA;
          break;
        }
        phase = 1;
      } else if (phase == 1) {
        if (phases & kPhaseB) {
          cnt = collect_tier(row_a, row_ab, 0, NB, b_blk, kBlock / RM, sCol, lane);
          if (cnt > 0) {
            tier = kTileB;
            break;
          }
        }
        phase = 2;
      } else {
        if (phases & kPhaseC) {
          cnt = collect_tier(row_a, row_ab, 1, NB, c_blk, kBlock, sCol, lane);
          if (cnt > 0) {
            tier = kTileC;
            break;
          }
        }
        phase = 3;
      }
    }
    if (lane == 0) {
      sMeta[kMetaTier] = tier;
      sMeta[kMetaBlk] = blk;
      sMeta[kMetaCnt] = (tier == kTileA) ? blk_cnt[blk] : cnt;
    }
  }
};

// Turns a gathered tile's block ids into its 64 table rows and 64 log2-count biases. Split in two so
// the log2-count load, which is a scattered 4-byte read every thread's store depends on, can be
// issued before a long MMA and published after it instead of stalling a barrier-to-barrier window:
//   expand_tile_fetch(...)   issues the load, keeps (row, bias) in registers
//   <long tensor-core stage>
//   expand_tile_publish(...) stores them; the stage's closing barrier makes them visible
// Only called when the tier is kTileB or kTileC; tier A needs neither, its mask is its token count.
__device__ __forceinline__ void expand_tile_fetch(int tier, int cnt, int RM, const int* sCol,
                                                  const float* lc_mid, const float* lc_tail, int tid, int& row,
                                                  float& bias) {
  row = -1;
  bias = neg_inf();
  if (tid >= kBlock) return;
  if (tier == kTileB) {
    const int slot = tid / RM, g = tid - slot * RM;
    if (slot < cnt) {
      row = sCol[slot] * RM + g;
      bias = lc_mid[row];
    }
  } else if (tid < cnt) {
    row = sCol[tid];
    bias = lc_tail[row];
  }
}

__device__ __forceinline__ void expand_tile_publish(int* sIdx, float* sBias, int tid, int row, float bias) {
  if (tid >= kBlock) return;
  sIdx[tid] = row;
  sBias[tid] = bias;
}

// The two columns a thread owns in n-tile `nt`. Tier A has no bias array: its mask comes straight
// from the block's valid-token count (kcA >= 0), which is exactly VSA2's own `col < blk_cnt` test.
__device__ __forceinline__ void tile_bias2(int kcA, const float* sBias, int col, float& b0, float& b1) {
  if (kcA >= 0) {
    b0 = (col < kcA) ? 0.f : neg_inf();
    b1 = (col + 1 < kcA) ? 0.f : neg_inf();
  } else {
    b0 = sBias[col];
    b1 = sBias[col + 1];
  }
}

}  // namespace vsa2rep
