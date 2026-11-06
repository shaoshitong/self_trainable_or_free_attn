// Torch bindings for VSA2 representative-block attention (vsa2_rep._C).
//
// Sequencing and autograd live in Python (vsa2_rep/ops.py). Everything here launches native CUDA:
// the representative tables, the three-tier forward and backward, the adjoint of the construction,
// and the tier assignment -- which reuses VSA2's own PSP router scores and its exact global top-k,
// run at two budgets over the same score matrix, and exports the two selection bitmaps.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rep_kernels.h"
#include "vsa2_kernels.h"

using namespace vsa2rep;

#ifdef VSA2_REP_TK  // csrc/rep_tk/rep_tk.cu (ThunderKittens + TMA, sm_90a)
std::vector<torch::Tensor> rep_tk_build(torch::Tensor k, torch::Tensor v, torch::Tensor blk_cnt, int64_t RM);
std::vector<torch::Tensor> rep_tk_build_pooled(torch::Tensor kr, torch::Tensor vr, torch::Tensor kc,
                                               torch::Tensor vc, torch::Tensor blk_cnt, int64_t RM);
torch::Tensor rep_tk_lcb(torch::Tensor lc_mid, torch::Tensor b_ptr, torch::Tensor b_idx, int64_t NB, int64_t RM);
std::vector<torch::Tensor> rep_tk_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, std::vector<torch::Tensor> tab,
                                      torch::Tensor a_ptr, torch::Tensor a_idx, torch::Tensor b_ptr,
                                      torch::Tensor b_idx, torch::Tensor bits_ab, torch::Tensor blk_cnt, int64_t RM,
                                      int64_t phases, int64_t diag);
void rep_tk_bwd_c(torch::Tensor q, torch::Tensor kt, torch::Tensor vt, torch::Tensor lc_tail, torch::Tensor dout,
                  torch::Tensor lse, torch::Tensor delta, torch::Tensor bits_ab, torch::Tensor qg, torch::Tensor kgt,
                  torch::Tensor vgt);
void rep_tk_bwd_b(torch::Tensor q, torch::Tensor k_sw, torch::Tensor v_sw, torch::Tensor lc_mid, torch::Tensor dout,
                  torch::Tensor lse, torch::Tensor delta, torch::Tensor b_ptr, torch::Tensor b_idx, torch::Tensor qg,
                  torch::Tensor dk_mid, torch::Tensor dv_mid, int64_t RM);
std::vector<torch::Tensor> tk_native_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor row_ptr,
                                         torch::Tensor col_idx, torch::Tensor blk_cnt);
bool tk_bwd_store_kv();
void tk_native_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                   torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                   torch::Tensor qg, torch::Tensor kg, torch::Tensor vg);
#endif

namespace {

#define CHECK_DEV(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be a contiguous CUDA tensor")
#define CHECK_I32(x)                                                                        \
  TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == torch::kInt32, \
              #x " must be a contiguous CUDA int32 tensor")
#define CHECK_F32(x)                                                                          \
  TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == torch::kFloat32, \
              #x " must be a contiguous CUDA fp32 tensor")

DType dtype_of(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kBFloat16) return DType::kBF16;
  if (t.scalar_type() == torch::kHalf) return DType::kFP16;
  TORCH_CHECK(false, "vsa2_rep: only bfloat16 and float16 token tensors are supported");
  return DType::kBF16;
}

void check_tokens(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.dim() == 4, name, " must be [B, H, L, D]");
  CHECK_DEV(x);
  TORCH_CHECK(x.size(3) == 64 || x.size(3) == 128, name, ": head dim must be 64 or 128");
  TORCH_CHECK(x.size(2) % kBlock == 0, name, ": L must be a multiple of 64 (block-padded layout)");
}

inline cudaStream_t cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }
inline void check_launch() {
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "vsa2_rep CUDA launch failed: ", cudaGetErrorString(e));
}
inline int ceil_to(long long x, int m) { return (int)((x + m - 1) / m * m); }
// REP_TK_FWD_CFG=48: the pool pass with two column pairs per thread (64-bit loads / stores), byte-identical.
inline int pool_vec() {
  static const int v = [] {
    const char* e = std::getenv("REP_TK_FWD_CFG");
    return (e != nullptr && std::strcmp(e, "48") == 0) ? 2 : 1;
  }();
  return v;
}

void check_rm(int64_t RM) {
  TORCH_CHECK(RM == 1 || RM == 2 || RM == 4 || RM == 8 || RM == 16,
              "vsa2_rep: mid representatives per block must be 1, 2, 4, 8 or 16, got ", RM);
}

// ---------------------------------------------------------------- representative tables
std::vector<torch::Tensor> rep_build(torch::Tensor k, torch::Tensor v, torch::Tensor blk_cnt, int64_t RM) {
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(blk_cnt);
  check_rm(RM);
  const c10::cuda::OptionalCUDAGuard guard(k.device());
  const int B = k.size(0), H = k.size(1), L = k.size(2), D = k.size(3), NB = L / kBlock;
  TORCH_CHECK(blk_cnt.numel() == NB, "blk_cnt must have L/64 entries");
  TORCH_CHECK(k.sizes() == v.sizes(), "k and v must share [B,H,L,D]");
  auto opt = k.options();
  auto f32 = opt.dtype(torch::kFloat32);
  auto k_mid = torch::empty({B, H, NB * (int)RM, D}, opt);
  auto v_mid = torch::empty({B, H, NB * (int)RM, D}, opt);
  auto lc_mid = torch::empty({B, H, NB * (int)RM}, f32);
  auto k_tail = torch::empty({B, H, NB, D}, opt);
  auto v_tail = torch::empty({B, H, NB, D}, opt);
  auto lc_tail = torch::empty({B, H, NB}, f32);
  RepBuildParams p{k.data_ptr(),      v.data_ptr(),      k_mid.data_ptr(),         v_mid.data_ptr(),
                   lc_mid.data_ptr<float>(), k_tail.data_ptr(), v_tail.data_ptr(), lc_tail.data_ptr<float>(),
                   blk_cnt.data_ptr<int>(), B * H, L, NB, (int)RM};
  launch_rep_build(p, D, dtype_of(k), cur_stream());
  check_launch();
  return {k_mid, v_mid, lc_mid, k_tail, v_tail, lc_tail};
}

std::vector<torch::Tensor> rep_unpool(c10::optional<torch::Tensor> acc_k, c10::optional<torch::Tensor> acc_v,
                                      torch::Tensor dk_mid, torch::Tensor dv_mid, torch::Tensor dk_tail,
                                      torch::Tensor dv_tail, torch::Tensor blk_cnt, torch::Tensor like, int64_t RM) {
  check_tokens(like, "like");
  CHECK_I32(blk_cnt);
  CHECK_F32(dk_mid);
  CHECK_F32(dv_mid);
  CHECK_F32(dk_tail);
  CHECK_F32(dv_tail);
  check_rm(RM);
  const c10::cuda::OptionalCUDAGuard guard(like.device());
  const int B = like.size(0), H = like.size(1), L = like.size(2), D = like.size(3), NB = L / kBlock;
  TORCH_CHECK(dk_mid.numel() == (long long)B * H * NB * RM * D, "dk_mid must be [B,H,NB*RM,D]");
  TORCH_CHECK(dk_tail.dim() == 4 && dk_tail.size(2) >= NB && dk_tail.size(3) == D,
              "dk_tail must be [B,H,NBt>=NB,D]");
  auto dk = torch::empty_like(like);
  auto dv = torch::empty_like(like);
  RepUnpoolParams p{};
  if (acc_k.has_value()) {
    CHECK_F32(*acc_k);
    CHECK_F32(*acc_v);
    TORCH_CHECK(acc_k->sizes() == like.sizes(), "acc_k must be fp32 like the token tensors");
    p.acc_k = acc_k->data_ptr<float>();
    p.acc_v = acc_v->data_ptr<float>();
  }
  p.dk_mid = dk_mid.data_ptr<float>();
  p.dv_mid = dv_mid.data_ptr<float>();
  p.dk_tail = dk_tail.data_ptr<float>();
  p.dv_tail = dv_tail.data_ptr<float>();
  p.dk = dk.data_ptr();
  p.dv = dv.data_ptr();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.RM = (int)RM;
  p.NBt = (int)dk_tail.size(2);
  launch_rep_unpool(p, D, dtype_of(like), cur_stream());
  check_launch();
  return {dk, dv};
}

torch::Tensor row_dot(torch::Tensor a, torch::Tensor b, torch::Tensor blk_cnt) {
  check_tokens(a, "a");
  check_tokens(b, "b");
  CHECK_I32(blk_cnt);
  TORCH_CHECK(a.sizes() == b.sizes(), "row_dot operands must share [B,H,L,D]");
  const c10::cuda::OptionalCUDAGuard guard(a.device());
  const int B = a.size(0), H = a.size(1), L = a.size(2), D = a.size(3);
  auto out = torch::empty({B, H, L}, a.options().dtype(torch::kFloat32));
  RowDotParams p{a.data_ptr(), b.data_ptr(), out.data_ptr<float>(), blk_cnt.data_ptr<int>(), B * H, L, L / kBlock};
  launch_row_dot(p, D, dtype_of(a), cur_stream());
  check_launch();
  return out;
}

// ---------------------------------------------------------------- tier assignment
// VSA2's own PSP router scores, then its exact global top-k at two budgets over the same score
// matrix: budget_a selects tier A (whose CSR lists are returned), budget_ab selects A | B. The two
// selection bitmaps are exported so that tier B = A|B minus A and tier C = the complement of A|B are
// read bit by bit in the attention kernels -- no NB*NB index array for the reduced tiers.
// Returns q2k_ptr, q2k_idx, k2q_ptr, k2q_idx (tier A), bits_a, bits_ab.
std::vector<torch::Tensor> route_tiers(torch::Tensor qr, torch::Tensor kr, torch::Tensor blk_cnt, int64_t R,
                                       int64_t budget_a, int64_t budget_ab, int64_t row_norm,
                                       int64_t max_score_bytes) {
  CHECK_DEV(qr);
  CHECK_DEV(kr);
  CHECK_I32(blk_cnt);
  TORCH_CHECK(budget_a >= 0 && budget_ab >= budget_a, "need 0 <= budget_a <= budget_ab");
  const c10::cuda::OptionalCUDAGuard guard(qr.device());
  const int B = qr.size(0), H = qr.size(1), D = qr.size(3), BH = B * H;
  const int G = kBlock / (int)R;
  const int NB = blk_cnt.numel();
  const int Lr = NB * G, Lr_pad = qr.size(2);
  const long long nn = (long long)NB * NB;
  const long long keff_a = std::max<long long>(0, std::min<long long>(budget_a, nn));
  const long long keff_ab = std::max<long long>(0, std::min<long long>(budget_ab, nn));
  TORCH_CHECK((long long)BH * keff_ab < (1LL << 31), "vsa2_rep: selected blocks exceed int32 CSR range");
  const int W = words_of(NB);

  auto i32 = qr.options().dtype(torch::kInt32);
  auto q2k_ptr = torch::empty({B, H, NB + 1}, i32);
  auto k2q_ptr = torch::empty({B, H, NB + 1}, i32);
  // One allocation per direction, sized for the larger (A|B) pass and reused by the tier-A pass.
  auto q2k_buf = torch::empty({std::max<long long>(1, (long long)BH * keff_ab)}, i32);
  auto k2q_buf = torch::empty({std::max<long long>(1, (long long)BH * keff_ab)}, i32);
  // The A|B q2k CSR is kept (tier-B lists are derived from it); its k2q side is scratch.
  auto ab_ptr = torch::empty({B, H, NB + 1}, i32);
  auto ab_idx = torch::empty({std::max<long long>(1, (long long)BH * keff_ab)}, i32);
  auto ab_k2q_ptr = torch::empty({B, H, NB + 1}, i32);
  auto u32 = qr.options().dtype(torch::kInt32);
  auto bits_a = torch::zeros({B, H, NB, W}, u32);
  auto bits_ab = torch::zeros({B, H, NB, W}, u32);

  vsa2::RouterParams p{};
  p.qr = qr.data_ptr();
  p.kr = kr.data_ptr();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = BH;
  p.NB = NB;
  p.R = (int)R;
  p.G = G;
  p.Lr = Lr;
  p.Lr_pad = Lr_pad;
  p.row_norm = (int)row_norm;
  p.scale_log2 = vsa2::kLog2eHost() / std::sqrt((float)D);

  auto lse2 = torch::empty({B, H, Lr_pad}, qr.options().dtype(torch::kFloat32));
  p.lse2 = lse2.data_ptr<float>();
  vsa2::launch_router_lse(p, D, dtype_of(qr), cur_stream());
  check_launch();

  const long long per_bh = nn * (long long)sizeof(float);
  int chunk = (int)std::max<long long>(1, std::min<long long>(BH, max_score_bytes / std::max<long long>(per_bh, 1)));
  chunk = std::min(chunk, 65535);
  auto scores = torch::empty({chunk, NB, NB}, qr.options().dtype(torch::kFloat32));
  auto ws = torch::empty({(long long)vsa2::topk_workspace_bytes(chunk, NB)}, qr.options().dtype(torch::kUInt8));
  vsa2::TopkWorkspace w;
  vsa2::topk_carve_workspace(ws.data_ptr(), chunk, NB, &w);
  auto* bits_a_p = reinterpret_cast<uint32_t*>(bits_a.data_ptr<int>());
  auto* bits_ab_p = reinterpret_cast<uint32_t*>(bits_ab.data_ptr<int>());

  // REP_TK_FWD_CFG=44: the two budget chains are independent once `scores` exists, and each is fourteen tiny
  // kernels, so run the A|B chain on a second stream to fill the other's launch gaps. That needs its own
  // workspace and its own kv->q scratch (both chains otherwise share `k2q_buf`), plus events to order it
  // against the score pass and to join before the next iteration rewrites `scores`.
  // Default since it measured -2% on F+B; CFG44 is the accepted alias and CFG45 restores the sequential order.
  const char* cfg_env = std::getenv("REP_TK_FWD_CFG");
  const bool parallel = (keff_ab != keff_a) && !(cfg_env != nullptr && std::strcmp(cfg_env, "45") == 0);
  // Each chain builds its kv->q CSR with one kernel instead of cols / colscan / cols (same integers, two
  // launches fewer per chain); the bitmaps come from earlier kernels and are untouched. Default since it
  // measured -1.3% on the full forward; CFG46 is the accepted alias and CFG47 restores the three-kernel form.
  const bool fused_cols = !(cfg_env != nullptr && std::strcmp(cfg_env, "47") == 0);
  torch::Tensor ws2, ab_k2q_buf2;
  vsa2::TopkWorkspace w2{};
  // Created once per process: creating and destroying a stream plus two events on every routing call cost more
  // than the launch gaps this arm is trying to fill (that is what made the first attempt look unstable).
  static cudaStream_t s2 = nullptr;
  static cudaEvent_t ev_scores = nullptr, ev_ab = nullptr;
  static int s2_device = -1;
  if (parallel) {
    ws2 = torch::empty({(long long)vsa2::topk_workspace_bytes(chunk, NB)}, qr.options().dtype(torch::kUInt8));
    vsa2::topk_carve_workspace(ws2.data_ptr(), chunk, NB, &w2);
    ab_k2q_buf2 = torch::empty_like(k2q_buf);
    int device = -1;
    cudaGetDevice(&device);
    if (s2 == nullptr || s2_device != device) {  // a stream belongs to the device that created it
      cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);  // the previous one is left alone: it may be in use
      cudaEventCreateWithFlags(&ev_scores, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&ev_ab, cudaEventDisableTiming);
      s2_device = device;
    }
    check_launch();
  }

  for (int bh0 = 0; bh0 < BH; bh0 += chunk) {
    const int cnt = std::min(chunk, BH - bh0);
    p.scores = scores.data_ptr<float>();
    p.bh0 = bh0;
    p.bh_count = cnt;
    vsa2::launch_router_scores(p, D, dtype_of(qr), cur_stream());
    const size_t bits_bytes = (size_t)cnt * NB * W * sizeof(uint32_t);
    // A|B first, when it differs from A: its CSR output is scratch, only the bitmap is kept. With an
    // empty mid tier the two budgets coincide, tier B is empty and A|B *is* A, so one pass suffices --
    // which also makes frac_mid = 0 cost exactly VSA2's own single routing pass.
    if (keff_ab != keff_a) {
      vsa2::TopkParams ab{scores.data_ptr<float>(), NB,  bh0,                      cnt,
                          keff_ab,                  ab_ptr.data_ptr<int>(), ab_idx.data_ptr<int>(),
                          ab_k2q_ptr.data_ptr<int>(),
                          (parallel ? ab_k2q_buf2 : k2q_buf).data_ptr<int>()};
      if (parallel) {  // the A|B chain waits for the scores it reads, then runs beside the tier-A chain
        cudaEventRecord(ev_scores, cur_stream());
        cudaStreamWaitEvent(s2, ev_scores, 0);
        vsa2::launch_global_topk(ab, w2, s2, fused_cols);
        cudaMemcpyAsync(bits_ab_p + (size_t)bh0 * NB * W, w2.selbits, bits_bytes, cudaMemcpyDeviceToDevice, s2);
      } else {
        vsa2::launch_global_topk(ab, w, cur_stream(), fused_cols);
        cudaMemcpyAsync(bits_ab_p + (size_t)bh0 * NB * W, w.selbits, bits_bytes, cudaMemcpyDeviceToDevice,
                        cur_stream());
      }
    }
    // Tier A second, so the CSR buffers end up holding the tier-A lists.
    vsa2::TopkParams a{scores.data_ptr<float>(), NB,  bh0,                      cnt,
                       keff_a,                   q2k_ptr.data_ptr<int>(), q2k_buf.data_ptr<int>(),
                       k2q_ptr.data_ptr<int>(),  k2q_buf.data_ptr<int>()};
    vsa2::launch_global_topk(a, w, cur_stream(), fused_cols);
    cudaMemcpyAsync(bits_a_p + (size_t)bh0 * NB * W, w.selbits, bits_bytes, cudaMemcpyDeviceToDevice, cur_stream());
    if (keff_ab == keff_a)
      cudaMemcpyAsync(bits_ab_p + (size_t)bh0 * NB * W, w.selbits, bits_bytes, cudaMemcpyDeviceToDevice,
                      cur_stream());
    if (parallel) {  // join before the next iteration rewrites `scores`, and before tier-B lists read A|B
      cudaEventRecord(ev_ab, s2);
      cudaStreamWaitEvent(cur_stream(), ev_ab, 0);
    }
    check_launch();
  }
  // The per-iteration join above already ordered the main stream behind every A|B kernel; the stream and the
  // events outlive the call on purpose.
  const long long used = std::max<long long>(1, (long long)BH * keff_a);
  auto b_ptr = torch::empty({B, H, NB + 1}, i32);
  auto b_idx = torch::empty({std::max<long long>(1, (long long)BH * (keff_ab - keff_a))}, i32);
  if (keff_ab != keff_a) {
    TierBListParams tb{ab_ptr.data_ptr<int>(), ab_idx.data_ptr<int>(), q2k_ptr.data_ptr<int>(),
                       reinterpret_cast<const uint32_t*>(bits_a.data_ptr<int>()), b_ptr.data_ptr<int>(),
                       b_idx.data_ptr<int>(), BH, NB, W};
    launch_tier_b_lists(tb, cur_stream());
    check_launch();
  } else {  // empty tier B: ptr_B = 0 everywhere
    b_ptr.zero_();
  }
  return {q2k_ptr, q2k_buf.narrow(0, 0, used), k2q_ptr, k2q_buf.narrow(0, 0, used), bits_a, bits_ab, b_ptr,
          b_idx};
}

// ---------------------------------------------------------------- attention
std::vector<torch::Tensor> rep_attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_mid,
                                        torch::Tensor v_mid, torch::Tensor lc_mid, torch::Tensor k_tail,
                                        torch::Tensor v_tail, torch::Tensor lc_tail, torch::Tensor row_ptr,
                                        torch::Tensor col_idx, torch::Tensor bits_a, torch::Tensor bits_ab,
                                        torch::Tensor blk_cnt, int64_t RM, int64_t phases, int64_t diag) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(bits_a);
  CHECK_I32(bits_ab);
  CHECK_I32(blk_cnt);
  CHECK_F32(lc_mid);
  CHECK_F32(lc_tail);
  check_rm(RM);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v must share [B,H,L,D]");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  const int W = words_of(NB);
  TORCH_CHECK(blk_cnt.numel() == NB && row_ptr.numel() == (long long)B * H * (NB + 1), "metadata shape mismatch");
  TORCH_CHECK(bits_a.numel() == (long long)B * H * NB * W && bits_ab.numel() == bits_a.numel(),
              "selection bitmaps must be [B,H,NB,ceil(NB/32)]");
  auto o = torch::empty_like(q);
  auto lse = torch::empty({B, H, L}, q.options().dtype(torch::kFloat32));
  RepAttnFwdParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.k_mid = k_mid.data_ptr();
  p.v_mid = v_mid.data_ptr();
  p.lc_mid = lc_mid.data_ptr<float>();
  p.k_tail = k_tail.data_ptr();
  p.v_tail = v_tail.data_ptr();
  p.lc_tail = lc_tail.data_ptr<float>();
  p.o = o.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.row_ptr = row_ptr.data_ptr<int>();
  p.col_idx = col_idx.data_ptr<int>();
  p.bits_a = reinterpret_cast<const uint32_t*>(bits_a.data_ptr<int>());
  p.bits_ab = reinterpret_cast<const uint32_t*>(bits_ab.data_ptr<int>());
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.RM = (int)RM;
  p.W = W;
  p.scale_log2 = vsa2::kLog2eHost() / std::sqrt((float)D);
  p.phases = (int)phases;
  p.diag = (int)diag;
  launch_rep_attn_fwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {o, lse};
}

// dQ (exact, deterministic) plus the fp32 representative dK/dV tables, accumulated with atomics.
// dk_mid/dv_mid/dk_tail/dv_tail must be zeroed by the caller.
torch::Tensor rep_attn_bwd_q(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor k_mid,
                             torch::Tensor v_mid, torch::Tensor lc_mid, torch::Tensor k_tail, torch::Tensor v_tail,
                             torch::Tensor lc_tail, torch::Tensor dout, torch::Tensor lse, torch::Tensor delta,
                             torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor bits_a,
                             torch::Tensor bits_ab, torch::Tensor blk_cnt, torch::Tensor dk_mid,
                             torch::Tensor dv_mid, torch::Tensor dk_tail, torch::Tensor dv_tail, int64_t RM,
                             int64_t phases, c10::optional<torch::Tensor> add_dq,
                             c10::optional<torch::Tensor> dq_acc_in, int64_t diag) {
  check_tokens(q, "q");
  check_tokens(dout, "dout");
  CHECK_F32(lse);
  CHECK_F32(delta);
  CHECK_F32(dk_mid);
  CHECK_F32(dv_mid);
  CHECK_F32(dk_tail);
  CHECK_F32(dv_tail);
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(bits_a);
  CHECK_I32(bits_ab);
  CHECK_I32(blk_cnt);
  check_rm(RM);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  const int W = words_of(NB);
  auto dq = torch::empty_like(q);
  RepAttnBwdQParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.k_mid = k_mid.data_ptr();
  p.v_mid = v_mid.data_ptr();
  p.lc_mid = lc_mid.data_ptr<float>();
  p.k_tail = k_tail.data_ptr();
  p.v_tail = v_tail.data_ptr();
  p.lc_tail = lc_tail.data_ptr<float>();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  p.dq = dq.data_ptr();
  if (add_dq.has_value()) {
    CHECK_F32(*add_dq);
    p.add_dq = add_dq->data_ptr<float>();
    p.add_nb = (int)add_dq->size(2);
  }
  if (dq_acc_in.has_value()) {
    CHECK_F32(*dq_acc_in);
    TORCH_CHECK(dq_acc_in->sizes() == q.sizes(), "dq_acc_in must be fp32 like q");
    p.dq_acc_in = dq_acc_in->data_ptr<float>();
  }
  p.dk_mid = dk_mid.data_ptr<float>();
  p.dv_mid = dv_mid.data_ptr<float>();
  p.dk_tail = dk_tail.data_ptr<float>();
  p.dv_tail = dv_tail.data_ptr<float>();
  p.row_ptr = row_ptr.data_ptr<int>();
  p.col_idx = col_idx.data_ptr<int>();
  p.bits_a = reinterpret_cast<const uint32_t*>(bits_a.data_ptr<int>());
  p.bits_ab = reinterpret_cast<const uint32_t*>(bits_ab.data_ptr<int>());
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.RM = (int)RM;
  p.W = W;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = vsa2::kLog2eHost() / std::sqrt((float)D);
  p.phases = (int)phases;
  p.diag = (int)diag;
  launch_rep_attn_bwd_q(p, D, dtype_of(q), cur_stream());
  check_launch();
  return dq;
}

// Exact tier-A token dK/dV from VSA2's own KV-owner backward over the tier-A k2q lists, into fp32
// accumulators. Every tier-A column has bias 0, so VSA2's `col < blk_cnt` masking is exactly right
// and this kernel is reused unmodified; the global (three-tier) lse and delta are what it consumes.
std::vector<torch::Tensor> tier_a_bwd_kv(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout,
                                         torch::Tensor lse, torch::Tensor delta, torch::Tensor k2q_ptr,
                                         torch::Tensor k2q_idx, torch::Tensor blk_cnt,
                                         c10::optional<torch::Tensor> dq_acc) {
  check_tokens(q, "q");
  CHECK_F32(lse);
  CHECK_F32(delta);
  CHECK_I32(k2q_ptr);
  CHECK_I32(k2q_idx);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
  auto f32 = q.options().dtype(torch::kFloat32);
  auto dk = torch::empty({B, H, L, D}, f32);
  auto dv = torch::empty({B, H, L, D}, f32);
  vsa2::AttnBwdParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  if (dq_acc.has_value()) {  // VSA2 variant 1: this kernel also accumulates tier-A dQ with atomics
    CHECK_F32(*dq_acc);
    TORCH_CHECK(dq_acc->sizes() == q.sizes(), "dq_acc must be fp32 like q");
    p.dq_acc = dq_acc->data_ptr<float>();
  }
  p.dk_f32 = dk.data_ptr<float>();
  p.dv_f32 = dv.data_ptr<float>();
  p.row_ptr = k2q_ptr.data_ptr<int>();
  p.col_idx = k2q_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = L / kBlock;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = vsa2::kLog2eHost() / std::sqrt((float)D);
  vsa2::launch_attn_bwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {dk, dv};
}

// ---------------------------------------------------------------- re-exported VSA2 stages
// The three-tier operator needs VSA2's pooling (router pools and block means), its dense coarse
// attention over the block means, and its gate fusion. They are compiled into this extension so the
// package is self-contained; the kernels are VSA2's own, unmodified.
std::vector<torch::Tensor> pool_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor blk_cnt,
                                    int64_t R) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(blk_cnt);
  TORCH_CHECK(R == 4 || R == 8 || R == 16 || R == 32, "router pool size R must be 4, 8, 16 or 32");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  TORCH_CHECK(blk_cnt.numel() == NB, "blk_cnt must have L/64 entries");
  const int G = kBlock / (int)R;
  const int Lr_pad = ceil_to((long long)NB * G, 64), NBc_pad = ceil_to(NB, 64);
  auto opt = q.options();
  auto qr = torch::zeros({B, H, Lr_pad, D}, opt);
  auto kr = torch::zeros({B, H, Lr_pad, D}, opt);
  auto qc = torch::zeros({B, H, NBc_pad, D}, opt);
  auto kc = torch::zeros({B, H, NBc_pad, D}, opt);
  auto vc = torch::zeros({B, H, NBc_pad, D}, opt);
  vsa2::PoolParams p{q.data_ptr(),  k.data_ptr(),  v.data_ptr(),  qr.data_ptr(), kr.data_ptr(),
                     qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), nullptr,       blk_cnt.data_ptr<int>(),
                     B * H,         L,             NB,            (int)R,        G,
                     Lr_pad,        NBc_pad};
  vsa2::launch_pool(p, D, dtype_of(q), cur_stream(), pool_vec());
  check_launch();
  return {qr, kr, qc, kc, vc};
}

// Same pass as pool_fwd plus the V group means (`vr`), which are exactly the tier-B V representatives when
// R equals the mid-tier group size (64 / mid_reps). A separate entry point keeps pool_fwd's arity intact.
std::vector<torch::Tensor> pool_fwd_vr(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor blk_cnt,
                                       int64_t R) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(blk_cnt);
  TORCH_CHECK(R == 4 || R == 8 || R == 16 || R == 32, "router pool size R must be 4, 8, 16 or 32");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  TORCH_CHECK(blk_cnt.numel() == NB, "blk_cnt must have L/64 entries");
  const int G = kBlock / (int)R;
  const int Lr_pad = ceil_to((long long)NB * G, 64), NBc_pad = ceil_to(NB, 64);
  auto opt = q.options();
  auto qr = torch::zeros({B, H, Lr_pad, D}, opt);
  auto kr = torch::zeros({B, H, Lr_pad, D}, opt);
  auto vrl = torch::zeros({B, H, Lr_pad, D}, opt);
  auto qc = torch::zeros({B, H, NBc_pad, D}, opt);
  auto kc = torch::zeros({B, H, NBc_pad, D}, opt);
  auto vc = torch::zeros({B, H, NBc_pad, D}, opt);
  vsa2::PoolParams p{q.data_ptr(),  k.data_ptr(),  v.data_ptr(),  qr.data_ptr(),  kr.data_ptr(),
                     qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), vrl.data_ptr(), blk_cnt.data_ptr<int>(),
                     B * H,         L,             NB,            (int)R,         G,
                     Lr_pad,        NBc_pad};
  vsa2::launch_pool(p, D, dtype_of(q), cur_stream(), pool_vec());
  check_launch();
  return {qr, kr, vrl, qc, kc, vc};
}

std::vector<torch::Tensor> attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor row_ptr,
                                    torch::Tensor col_idx, torch::Tensor blk_cnt) {
  check_tokens(q, "q");
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
  auto o = torch::empty_like(q);
  auto lse = torch::empty({B, H, L}, q.options().dtype(torch::kFloat32));
  vsa2::AttnFwdParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.o = o.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.row_ptr = row_ptr.data_ptr<int>();
  p.col_idx = col_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = L / kBlock;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = vsa2::kLog2eHost() / std::sqrt((float)D);
  vsa2::launch_attn_fwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {o, lse};
}

std::vector<torch::Tensor> attn_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout,
                                    torch::Tensor lse, torch::Tensor delta, torch::Tensor row_ptr,
                                    torch::Tensor col_idx, torch::Tensor blk_cnt,
                                    c10::optional<torch::Tensor> dq_acc, bool out_fp32) {
  check_tokens(q, "q");
  CHECK_F32(lse);
  CHECK_F32(delta);
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3);
  torch::Tensor dk, dv;
  if (out_fp32) {
    dk = torch::empty(q.sizes(), q.options().dtype(torch::kFloat32));
    dv = torch::empty(q.sizes(), q.options().dtype(torch::kFloat32));
  } else {
    dk = torch::empty_like(q);
    dv = torch::empty_like(q);
  }
  vsa2::AttnBwdParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  if (dq_acc.has_value()) {
    CHECK_F32(*dq_acc);
    p.dq_acc = dq_acc->data_ptr<float>();
  }
  if (out_fp32) {
    p.dk_f32 = dk.data_ptr<float>();
    p.dv_f32 = dv.data_ptr<float>();
  } else {
    p.dk = dk.data_ptr();
    p.dv = dv.data_ptr();
  }
  p.row_ptr = row_ptr.data_ptr<int>();
  p.col_idx = col_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = L / kBlock;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = vsa2::kLog2eHost() / std::sqrt((float)D);
  vsa2::launch_attn_bwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {dk, dv};
}

int gate_mode_of(const c10::optional<torch::Tensor>& gate) {
  if (!gate.has_value()) return 0;
  CHECK_DEV(*gate);
  return gate->dim() == 3 ? 1 : 2;
}

torch::Tensor fuse_fwd(torch::Tensor of, torch::Tensor oc, c10::optional<torch::Tensor> gate, torch::Tensor blk_cnt,
                       int64_t gate_act) {
  check_tokens(of, "o_f");
  CHECK_DEV(oc);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(of.device());
  const int B = of.size(0), H = of.size(1), L = of.size(2), D = of.size(3);
  auto o = torch::empty_like(of);
  vsa2::FuseFwdParams p{of.data_ptr(),           oc.data_ptr(), gate.has_value() ? gate->data_ptr() : nullptr,
                        o.data_ptr(),            blk_cnt.data_ptr<int>(), B * H,
                        L,                       L / kBlock,    (int)oc.size(2),
                        gate_mode_of(gate),      (int)gate_act};
  vsa2::launch_fuse_fwd(p, D, dtype_of(of), cur_stream());
  check_launch();
  return o;
}

std::vector<torch::Tensor> fuse_bwd(torch::Tensor dout, torch::Tensor of, torch::Tensor oc,
                                    c10::optional<torch::Tensor> gate, torch::Tensor blk_cnt, int64_t gate_act,
                                    bool write_dout_f) {
  check_tokens(of, "o_f");
  CHECK_DEV(dout);
  CHECK_DEV(oc);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(of.device());
  const int B = of.size(0), H = of.size(1), L = of.size(2), D = of.size(3), NBc_pad = oc.size(2);
  const int mode = gate_mode_of(gate);
  auto dout_f = write_dout_f ? torch::empty_like(of) : torch::empty({0}, of.options());
  auto delta_f = torch::empty({B, H, L}, of.options().dtype(torch::kFloat32));
  auto dout_c = torch::zeros({B, H, NBc_pad, D}, of.options());
  auto delta_c = torch::zeros({B, H, NBc_pad}, of.options().dtype(torch::kFloat32));
  torch::Tensor dgate = mode ? torch::empty_like(*gate) : torch::empty({0}, of.options());
  vsa2::FuseBwdParams p{dout.data_ptr(),
                        of.data_ptr(),
                        oc.data_ptr(),
                        mode ? gate->data_ptr() : nullptr,
                        write_dout_f ? dout_f.data_ptr() : nullptr,
                        delta_f.data_ptr<float>(),
                        dout_c.data_ptr(),
                        delta_c.data_ptr<float>(),
                        mode ? dgate.data_ptr() : nullptr,
                        blk_cnt.data_ptr<int>(),
                        B * H,
                        L,
                        L / kBlock,
                        NBc_pad,
                        mode,
                        (int)gate_act};
  vsa2::launch_fuse_bwd(p, D, dtype_of(of), cur_stream());
  check_launch();
  return {dout_f, delta_f, dout_c, delta_c, dgate};
}

torch::Tensor finalize(c10::optional<torch::Tensor> acc, c10::optional<torch::Tensor> add, torch::Tensor blk_cnt,
                       torch::Tensor like) {
  check_tokens(like, "like");
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(like.device());
  const int B = like.size(0), H = like.size(1), L = like.size(2), D = like.size(3);
  auto out = torch::empty_like(like);
  vsa2::FinalizeParams p{};
  if (acc.has_value()) {
    CHECK_F32(*acc);
    p.acc = acc->data_ptr<float>();
  }
  if (add.has_value()) {
    CHECK_F32(*add);
    p.add = add->data_ptr<float>();
    p.add_nb = add->size(2);
  }
  p.out = out.data_ptr();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = L / kBlock;
  vsa2::launch_finalize(p, D, dtype_of(like), cur_stream());
  check_launch();
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "VSA2 representative-block attention (three-tier KV budget), native CUDA";
  // No pybind default arguments anywhere: every caller in vsa2_rep/ops.py passes all of them
  // positionally, and a default for c10::optional<Tensor> has to be converted at registration time.
  m.def("pool_fwd", &pool_fwd);
  m.def("pool_fwd_vr", &pool_fwd_vr);
  m.def("attn_fwd", &attn_fwd);
  m.def("attn_bwd", &attn_bwd);
  m.def("fuse_fwd", &fuse_fwd);
  m.def("fuse_bwd", &fuse_bwd);
  m.def("finalize", &finalize);
  m.def("rep_build", &rep_build);
  m.def("rep_unpool", &rep_unpool);
  m.def("row_dot", &row_dot);
  m.def("route_tiers", &route_tiers);
  m.def("rep_attn_fwd", &rep_attn_fwd);
  m.def("rep_attn_bwd_q", &rep_attn_bwd_q);
  m.def("tier_a_bwd_kv", &tier_a_bwd_kv);
#ifdef VSA2_REP_TK
  m.def("rep_tk_build", &rep_tk_build);
  m.def("rep_tk_build_pooled", &rep_tk_build_pooled);
  m.def("rep_tk_lcb", &rep_tk_lcb);
  m.def("rep_tk_fwd", &rep_tk_fwd);
  m.def("rep_tk_bwd_c", &rep_tk_bwd_c);
  m.def("rep_tk_bwd_b", &rep_tk_bwd_b);
  m.def("tk_native_fwd", &tk_native_fwd);
  m.def("tk_native_bwd", &tk_native_bwd);
  m.def("tk_bwd_store_kv", &tk_bwd_store_kv);
  m.def("has_tk", []() { return true; });
#else
  m.def("has_tk", []() { return false; });
#endif
}
