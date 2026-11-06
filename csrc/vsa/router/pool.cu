// VSA2 pooling: one pass over the q/k/v tokens of a 64-token block produces
//   * router pools  qr/kr : mean of each R-token group (G = 64/R groups per block)
//   * coarse pools  qc/kc/vc : mean of the block's valid tokens
// Only the first blk_cnt[blk] tokens of a block are read (valid-first layout),
// so padding contents never matter. Empty router groups are written as 0 and
// masked by the router kernels. Router pooling is stop-gradient; coarse
// pooling is differentiated in the backward (unpool: grad/cnt per valid token).
#include <type_traits>

#include "../common.cuh"
#include "../vsa2_kernels.h"

namespace vsa2 {

template <typename Elt, int D>
__global__ void pool_kernel(const PoolParams p) {
  using T = EltTraits<Elt>;
  constexpr int CP = D / 2;  // column pairs
  const int blk = blockIdx.x;
  const int bh = blockIdx.y;
  const int tensor = threadIdx.x / CP;  // 0 q, 1 k, 2 v
  const int cp = threadIdx.x % CP;
  const int cnt = p.blk_cnt[blk];
  const Elt* base = reinterpret_cast<const Elt*>(tensor == 0 ? p.q : (tensor == 1 ? p.k : p.v));
  const uint32_t* src =
      reinterpret_cast<const uint32_t*>(base + ((size_t)bh * p.L + (size_t)blk * kBlock) * D) + cp;
  Elt* rdst = tensor == 0 ? reinterpret_cast<Elt*>(p.qr) : reinterpret_cast<Elt*>(p.kr);
  float tot0 = 0.f, tot1 = 0.f;
  for (int gi = 0; gi < p.G; ++gi) {
    const int rb = gi * p.R;
    const int re = min(rb + p.R, cnt);
    float a0 = 0.f, a1 = 0.f;
    for (int r = rb; r < re; ++r) {
      float2 x = T::unpack2(src[(size_t)r * CP]);
      a0 += x.x;
      a1 += x.y;
    }
    const float inv_g = re > rb ? 1.f / (float)(re - rb) : 0.f;
    if (tensor < 2) {
      size_t row = (size_t)bh * p.Lr_pad + (size_t)blk * p.G + gi;
      reinterpret_cast<uint32_t*>(rdst + row * D)[cp] = T::pack2(a0 * inv_g, a1 * inv_g);
    } else if (p.vr != nullptr) {  // V group means, same rows as kr: the tier-B V representatives
      size_t row = (size_t)bh * p.Lr_pad + (size_t)blk * p.G + gi;
      reinterpret_cast<uint32_t*>(reinterpret_cast<Elt*>(p.vr) + row * D)[cp] = T::pack2(a0 * inv_g, a1 * inv_g);
    }
    tot0 += a0;
    tot1 += a1;
  }
  Elt* cdst = reinterpret_cast<Elt*>(tensor == 0 ? p.qc : (tensor == 1 ? p.kc : p.vc));
  float inv = 1.f / (float)cnt;
  size_t row = (size_t)bh * p.NBc_pad + blk;
  reinterpret_cast<uint32_t*>(cdst + row * D)[cp] = T::pack2(tot0 * inv, tot1 * inv);
}

// pool_kernel with V consecutive column pairs per thread (V = 2: one 64-bit load / store per row, V = 4: 128
// bits), launch_pool's vec. Every column keeps its own accumulator and the same sequence of adds over the
// block's rows, and the same multiplies by 1/n, so all six outputs are byte-identical to pool_kernel's.
__device__ __forceinline__ uint32_t word_at(const uint2& w, int i) { return i == 0 ? w.x : w.y; }
__device__ __forceinline__ uint32_t word_at(const uint4& w, int i) {
  return i == 0 ? w.x : (i == 1 ? w.y : (i == 2 ? w.z : w.w));
}
__device__ __forceinline__ uint2 make_words(const uint32_t (&o)[2]) { return make_uint2(o[0], o[1]); }
__device__ __forceinline__ uint4 make_words(const uint32_t (&o)[4]) { return make_uint4(o[0], o[1], o[2], o[3]); }

template <typename Elt, int D, int V>
__global__ void pool_vec_kernel(const PoolParams p) {
  static_assert(V == 2 || V == 4, "two or four column pairs per thread");
  using T = EltTraits<Elt>;
  using Vec = std::conditional_t<V == 2, uint2, uint4>;
  constexpr int TPT = D / 2 / V;  // threads per tensor; also a row's length in Vec units
  const int blk = blockIdx.x;
  const int bh = blockIdx.y;
  const int tensor = threadIdx.x / TPT;  // 0 q, 1 k, 2 v
  const int c = threadIdx.x % TPT;       // column pairs c*V .. c*V + V - 1
  const int cnt = p.blk_cnt[blk];
  const Elt* base = reinterpret_cast<const Elt*>(tensor == 0 ? p.q : (tensor == 1 ? p.k : p.v));
  const Vec* src = reinterpret_cast<const Vec*>(base + ((size_t)bh * p.L + (size_t)blk * kBlock) * D) + c;
  Elt* rdst = tensor == 0 ? reinterpret_cast<Elt*>(p.qr) : reinterpret_cast<Elt*>(p.kr);
  float tot[2 * V];
#pragma unroll
  for (int u = 0; u < 2 * V; ++u) tot[u] = 0.f;
  for (int gi = 0; gi < p.G; ++gi) {
    const int rb = gi * p.R;
    const int re = min(rb + p.R, cnt);
    float a[2 * V];
#pragma unroll
    for (int u = 0; u < 2 * V; ++u) a[u] = 0.f;
    for (int r = rb; r < re; ++r) {
      const Vec w = src[(size_t)r * TPT];
#pragma unroll
      for (int u = 0; u < V; ++u) {
        float2 x = T::unpack2(word_at(w, u));
        a[2 * u] += x.x;
        a[2 * u + 1] += x.y;
      }
    }
    const float inv_g = re > rb ? 1.f / (float)(re - rb) : 0.f;
    uint32_t o[V];
#pragma unroll
    for (int u = 0; u < V; ++u) o[u] = T::pack2(a[2 * u] * inv_g, a[2 * u + 1] * inv_g);
    if (tensor < 2) {
      size_t row = (size_t)bh * p.Lr_pad + (size_t)blk * p.G + gi;
      reinterpret_cast<Vec*>(rdst + row * D)[c] = make_words(o);
    } else if (p.vr != nullptr) {  // V group means, same rows as kr: the tier-B V representatives
      size_t row = (size_t)bh * p.Lr_pad + (size_t)blk * p.G + gi;
      reinterpret_cast<Vec*>(reinterpret_cast<Elt*>(p.vr) + row * D)[c] = make_words(o);
    }
#pragma unroll
    for (int u = 0; u < 2 * V; ++u) tot[u] += a[u];
  }
  Elt* cdst = reinterpret_cast<Elt*>(tensor == 0 ? p.qc : (tensor == 1 ? p.kc : p.vc));
  float inv = 1.f / (float)cnt;
  size_t row = (size_t)bh * p.NBc_pad + blk;
  uint32_t o[V];
#pragma unroll
  for (int u = 0; u < V; ++u) o[u] = T::pack2(tot[2 * u] * inv, tot[2 * u + 1] * inv);
  reinterpret_cast<Vec*>(cdst + row * D)[c] = make_words(o);
}

template <typename Elt, int D>
static void run_pool(const PoolParams& p, int vec, cudaStream_t s) {
  dim3 grid(p.NB, p.BH);
  if (vec == 2) pool_vec_kernel<Elt, D, 2><<<grid, 3 * (D / 4), 0, s>>>(p);
  else pool_kernel<Elt, D><<<grid, 3 * (D / 2), 0, s>>>(p);
}

void launch_pool(const PoolParams& p, int D, DType dt, cudaStream_t s, int vec) {
  if (dt == DType::kBF16) {
    if (D == 64) run_pool<__nv_bfloat16, 64>(p, vec, s);
    else run_pool<__nv_bfloat16, 128>(p, vec, s);
  } else {
    if (D == 64) run_pool<__half, 64>(p, vec, s);
    else run_pool<__half, 128>(p, vec, s);
  }
}

}  // namespace vsa2
