// Native VSA / VSA2 extension (echosparse.vsa._C): torch bindings. Sequencing/autograd lives in
// Python (echosparse/vsa/ops.py); every production op here launches native CUDA kernels.
#include <torch/extension.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "vsa2_kernels.h"

#ifdef VSA2_WITH_TK
void vsa2_tk_fine_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                      torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt);
void vsa2_tk_fine_fwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor lse,
                            torch::Tensor col_idx, torch::Tensor blk_cnt, torch::Tensor tasks, torch::Tensor info,
                            torch::Tensor part_o, torch::Tensor part_lse);
void vsa2_tk_fine_bwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f,
                            torch::Tensor lse, torch::Tensor delta, torch::Tensor col_idx, torch::Tensor blk_cnt,
                            torch::Tensor tasks, torch::Tensor info, torch::Tensor qg, torch::Tensor kg,
                            torch::Tensor vg, int opts);
void vsa2_tk_fine_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                      torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                      torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, int opts);
#endif

using namespace vsa2;

namespace {

// ECHOSPARSE_VSA_FUSED_COLS=1 (off by default): the global top-k builds its kv->q CSR with one kernel instead of
// topk_cols / topk_colscan / topk_cols (launch_global_topk's fused_cols; integers only, bit-identical lists).
// Read per call so the two paths can be compared in one process.
bool fused_cols_enabled() {
  const char* e = std::getenv("ECHOSPARSE_VSA_FUSED_COLS");
  return e != nullptr && e[0] == '1' && e[1] == '\0';
}

#define CHECK_DEV(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous(), #x " must be a contiguous CUDA tensor")
#define CHECK_I32(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && (x).scalar_type() == torch::kInt32, \
                                 #x " must be a contiguous CUDA int32 tensor")

DType dtype_of(const torch::Tensor& t) {
  if (t.scalar_type() == torch::kBFloat16) return DType::kBF16;
  if (t.scalar_type() == torch::kHalf) return DType::kFP16;
  TORCH_CHECK(false, "VSA2: only bfloat16 and float16 token tensors are supported");
  return DType::kBF16;
}

void check_tokens(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.dim() == 4, name, " must be [B, H, L, D]");
  CHECK_DEV(x);
  TORCH_CHECK(x.size(3) == 64 || x.size(3) == 128, name, ": head dim must be 64 or 128");
  TORCH_CHECK(x.size(2) % kBlock == 0, name, ": L must be a multiple of 64 (block-padded layout)");
}

inline int ceil_to(long long x, int m) { return (int)((x + m - 1) / m * m); }
inline cudaStream_t cur_stream() { return at::cuda::getCurrentCUDAStream().stream(); }
inline void check_launch() {
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "VSA2 CUDA launch failed: ", cudaGetErrorString(e));
}

}  // namespace

// ------------------------------------------------------------------ pooling
std::vector<torch::Tensor> pool_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor blk_cnt,
                                    int64_t R) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(blk_cnt);
  TORCH_CHECK(R == 4 || R == 8 || R == 16 || R == 32, "VSA2: router pool size R must be 4, 8, 16 or 32");
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
  PoolParams p{q.data_ptr(), k.data_ptr(), v.data_ptr(), qr.data_ptr(), kr.data_ptr(), qc.data_ptr(),
               kc.data_ptr(), vc.data_ptr(), blk_cnt.data_ptr<int>(), B * H, L, NB, (int)R, G, Lr_pad, NBc_pad};
  launch_pool(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {qr, kr, qc, kc, vc};
}

// ------------------------------------------------------------------ router
static RouterParams router_params(const torch::Tensor& qr, const torch::Tensor& kr, const torch::Tensor& blk_cnt,
                                  int64_t R, int64_t row_norm) {
  const int B = qr.size(0), H = qr.size(1), Lr_pad = qr.size(2), D = qr.size(3);
  const int NB = blk_cnt.numel();
  const int G = kBlock / (int)R;
  RouterParams p{};
  p.qr = qr.data_ptr();
  p.kr = kr.data_ptr();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.NB = NB;
  p.R = (int)R;
  p.G = G;
  p.Lr = NB * G;
  p.Lr_pad = Lr_pad;
  p.row_norm = (int)row_norm;
  p.scale_log2 = kLog2eHost() * (1.f / std::sqrt((float)D));
  return p;
}

torch::Tensor router_lse(torch::Tensor qr, torch::Tensor kr, torch::Tensor blk_cnt, int64_t R) {
  CHECK_DEV(qr);
  CHECK_DEV(kr);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(qr.device());
  auto p = router_params(qr, kr, blk_cnt, R, 0);
  auto lse2 = torch::empty({qr.size(0), qr.size(1), qr.size(2)}, qr.options().dtype(torch::kFloat32));
  p.lse2 = lse2.data_ptr<float>();
  launch_router_lse(p, qr.size(3), dtype_of(qr), cur_stream());
  check_launch();
  return lse2;
}

torch::Tensor router_scores(torch::Tensor qr, torch::Tensor kr, torch::Tensor lse2, torch::Tensor blk_cnt, int64_t R,
                            int64_t row_norm, int64_t bh0, int64_t bh_count) {
  CHECK_DEV(qr);
  CHECK_DEV(kr);
  CHECK_DEV(lse2);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(qr.device());
  auto p = router_params(qr, kr, blk_cnt, R, row_norm);
  if (bh_count <= 0) bh_count = p.BH - bh0;
  auto scores = torch::empty({bh_count, p.NB, p.NB}, qr.options().dtype(torch::kFloat32));
  p.lse2 = lse2.data_ptr<float>();
  p.scores = scores.data_ptr<float>();
  p.bh0 = (int)bh0;
  p.bh_count = (int)bh_count;
  launch_router_scores(p, qr.size(3), dtype_of(qr), cur_stream());
  check_launch();
  return scores;
}

// scores [bh_count, NB, NB] fp32 -> writes CSR rows bh0..bh0+bh_count-1 of the outputs
void global_topk_into(torch::Tensor scores, int64_t budget, int64_t bh0, torch::Tensor q2k_ptr, torch::Tensor q2k_idx,
                      torch::Tensor k2q_ptr, torch::Tensor k2q_idx) {
  CHECK_DEV(scores);
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32 && scores.dim() == 3 && scores.size(1) == scores.size(2),
              "scores must be fp32 [BH, NB, NB]");
  CHECK_I32(q2k_ptr);
  CHECK_I32(q2k_idx);
  CHECK_I32(k2q_ptr);
  CHECK_I32(k2q_idx);
  const c10::cuda::OptionalCUDAGuard guard(scores.device());
  const int bhc = scores.size(0), NB = scores.size(1);
  TORCH_CHECK(bhc <= 65535, "too many (b,h) per top-k chunk");
  auto ws = torch::empty({(long long)topk_workspace_bytes(bhc, NB)}, scores.options().dtype(torch::kUInt8));
  TopkWorkspace w;
  topk_carve_workspace(ws.data_ptr(), bhc, NB, &w);
  TopkParams p{scores.data_ptr<float>(), NB, (int)bh0, bhc, budget, q2k_ptr.data_ptr<int>(),
               q2k_idx.data_ptr<int>(), k2q_ptr.data_ptr<int>(), k2q_idx.data_ptr<int>()};
  launch_global_topk(p, w, cur_stream(), fused_cols_enabled());
  check_launch();
}

// Full router: lse pass over all heads, then score + exact global top-k in
// chunks of (b,h) bounded by max_score_bytes. Returns
// (q2k_ptr [B,H,NB+1], q2k_idx [B*H*k_eff], k2q_ptr, k2q_idx).
std::vector<torch::Tensor> route(torch::Tensor qr, torch::Tensor kr, torch::Tensor blk_cnt, int64_t R, int64_t budget,
                                 int64_t row_norm, int64_t max_score_bytes) {
  CHECK_DEV(qr);
  CHECK_DEV(kr);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(qr.device());
  auto p = router_params(qr, kr, blk_cnt, R, row_norm);
  const int B = qr.size(0), H = qr.size(1), D = qr.size(3), NB = p.NB, BH = p.BH;
  const long long nn = (long long)NB * NB;
  const long long keff = std::max<long long>(0, std::min<long long>(budget, nn));
  TORCH_CHECK((long long)BH * keff < (1LL << 31), "VSA2: total selected blocks exceed int32 CSR range");
  auto i32 = qr.options().dtype(torch::kInt32);
  auto q2k_ptr = torch::empty({B, H, NB + 1}, i32);
  auto k2q_ptr = torch::empty({B, H, NB + 1}, i32);
  auto q2k_idx = torch::empty({(long long)BH * keff}, i32);
  auto k2q_idx = torch::empty({(long long)BH * keff}, i32);

  auto lse2 = torch::empty({B, H, qr.size(2)}, qr.options().dtype(torch::kFloat32));
  p.lse2 = lse2.data_ptr<float>();
  launch_router_lse(p, D, dtype_of(qr), cur_stream());
  check_launch();

  long long per_bh = nn * (long long)sizeof(float);
  int chunk = (int)std::max<long long>(1, std::min<long long>(BH, max_score_bytes / std::max<long long>(per_bh, 1)));
  chunk = std::min(chunk, 65535);
  auto scores = torch::empty({chunk, NB, NB}, qr.options().dtype(torch::kFloat32));
  auto ws = torch::empty({(long long)topk_workspace_bytes(chunk, NB)}, qr.options().dtype(torch::kUInt8));
  TopkWorkspace w;
  topk_carve_workspace(ws.data_ptr(), chunk, NB, &w);
  const bool fused_cols = fused_cols_enabled();
  for (int bh0 = 0; bh0 < BH; bh0 += chunk) {
    int cnt = std::min(chunk, BH - bh0);
    p.scores = scores.data_ptr<float>();
    p.bh0 = bh0;
    p.bh_count = cnt;
    launch_router_scores(p, D, dtype_of(qr), cur_stream());
    TopkParams tp{scores.data_ptr<float>(), NB, bh0, cnt, budget, q2k_ptr.data_ptr<int>(),
                  q2k_idx.data_ptr<int>(), k2q_ptr.data_ptr<int>(), k2q_idx.data_ptr<int>()};
    launch_global_topk(tp, w, cur_stream(), fused_cols);
    check_launch();
  }
  return {q2k_ptr, q2k_idx, k2q_ptr, k2q_idx};
}

// ------------------------------------------------------------------ original VSA router
static BlockScoreParams block_score_params(const torch::Tensor& qc, const torch::Tensor& kc, int64_t nb) {
  CHECK_DEV(qc);
  CHECK_DEV(kc);
  TORCH_CHECK(qc.dim() == 4 && qc.sizes() == kc.sizes(), "qc/kc must be [B, H, NBc_pad, D] (pool_fwd)");
  TORCH_CHECK(qc.size(3) == 64 || qc.size(3) == 128, "qc: head dim must be 64 or 128");
  TORCH_CHECK(nb >= 1 && qc.size(2) == ceil_to(nb, 64), "qc must hold ceil(nb/64)*64 pooled rows");
  BlockScoreParams p{};
  p.qc = qc.data_ptr();
  p.kc = kc.data_ptr();
  p.NB = (int)nb;
  p.NBc_pad = (int)qc.size(2);
  p.scale = 1.f / std::sqrt((float)qc.size(3));
  return p;
}

static TopkWorkspace row_topk_workspace(torch::Tensor& ws, const torch::TensorOptions& opt, int bhc, int NB) {
  ws = torch::empty({(long long)topk_workspace_bytes(bhc, NB)}, opt.dtype(torch::kUInt8));
  TopkWorkspace w;
  topk_carve_workspace(ws.data_ptr(), bhc, NB, &w);
  return w;
}

// Whole-block scores of (b,h) rows bh0..bh0+bh_count-1 -> [bh_count, NB, NB] in the token dtype.
torch::Tensor block_scores(torch::Tensor qc, torch::Tensor kc, int64_t nb, int64_t bh0, int64_t bh_count) {
  const c10::cuda::OptionalCUDAGuard guard(qc.device());
  auto p = block_score_params(qc, kc, nb);
  const int BH = qc.size(0) * qc.size(1);
  if (bh_count <= 0) bh_count = BH - bh0;
  TORCH_CHECK(bh0 >= 0 && bh_count <= 65535 && bh0 + bh_count <= BH, "bad (b,h) range");
  auto scores = torch::empty({bh_count, nb, nb}, qc.options());
  p.scores = scores.data_ptr();
  p.bh0 = (int)bh0;
  p.bh_count = (int)bh_count;
  launch_block_scores(p, qc.size(3), dtype_of(qc), cur_stream());
  check_launch();
  return scores;
}

// scores [bh_count, NB, NB] bf16/fp16 -> writes CSR rows bh0..bh0+bh_count-1 of the outputs, K per row
void row_topk_into(torch::Tensor scores, int64_t topk, int64_t bh0, torch::Tensor q2k_ptr, torch::Tensor q2k_idx,
                   torch::Tensor k2q_ptr, torch::Tensor k2q_idx) {
  CHECK_DEV(scores);
  dtype_of(scores);
  TORCH_CHECK(scores.dim() == 3 && scores.size(1) == scores.size(2), "scores must be [BH, NB, NB]");
  CHECK_I32(q2k_ptr);
  CHECK_I32(q2k_idx);
  CHECK_I32(k2q_ptr);
  CHECK_I32(k2q_idx);
  const c10::cuda::OptionalCUDAGuard guard(scores.device());
  const int bhc = scores.size(0), NB = scores.size(1);
  TORCH_CHECK(bhc <= 65535, "too many (b,h) per top-k chunk");
  const int K = (int)std::max<int64_t>(0, std::min<int64_t>(topk, NB));
  torch::Tensor ws;
  auto w = row_topk_workspace(ws, scores.options(), bhc, NB);
  RowTopkParams p{reinterpret_cast<const uint16_t*>(scores.data_ptr()), NB, (int)bh0, bhc, K,
                  q2k_ptr.data_ptr<int>(), q2k_idx.data_ptr<int>(), k2q_ptr.data_ptr<int>(), k2q_idx.data_ptr<int>()};
  launch_row_topk(p, w, cur_stream());
  check_launch();
}

// Original VSA router: whole-block scores + per-row top-K in chunks of (b,h) bounded by max_score_bytes.
// Returns (q2k_ptr [B,H,NB+1], q2k_idx [B*H*NB*K], k2q_ptr, k2q_idx), K = min(topk, NB).
std::vector<torch::Tensor> route_rows(torch::Tensor qc, torch::Tensor kc, int64_t nb, int64_t topk,
                                      int64_t max_score_bytes) {
  const c10::cuda::OptionalCUDAGuard guard(qc.device());
  auto p = block_score_params(qc, kc, nb);
  const int B = qc.size(0), H = qc.size(1), BH = B * H, NB = (int)nb;
  const int K = (int)std::max<int64_t>(0, std::min<int64_t>(topk, NB));
  TORCH_CHECK((long long)BH * NB * K < (1LL << 31), "VSA: total selected blocks exceed int32 CSR range");
  auto i32 = qc.options().dtype(torch::kInt32);
  auto q2k_ptr = torch::empty({B, H, NB + 1}, i32);
  auto k2q_ptr = torch::empty({B, H, NB + 1}, i32);
  auto q2k_idx = torch::empty({(long long)BH * NB * K}, i32);
  auto k2q_idx = torch::empty({(long long)BH * NB * K}, i32);
  const long long per_bh = (long long)NB * NB * qc.element_size();
  int chunk = (int)std::max<long long>(1, std::min<long long>(BH, max_score_bytes / per_bh));
  chunk = std::min(chunk, 65535);
  auto scores = torch::empty({chunk, NB, NB}, qc.options());
  torch::Tensor ws;
  auto w = row_topk_workspace(ws, qc.options(), chunk, NB);
  for (int bh0 = 0; bh0 < BH; bh0 += chunk) {
    const int cnt = std::min(chunk, BH - bh0);
    p.scores = scores.data_ptr();
    p.bh0 = bh0;
    p.bh_count = cnt;
    launch_block_scores(p, qc.size(3), dtype_of(qc), cur_stream());
    RowTopkParams tp{reinterpret_cast<const uint16_t*>(scores.data_ptr()), NB, bh0, cnt, K,
                     q2k_ptr.data_ptr<int>(), q2k_idx.data_ptr<int>(), k2q_ptr.data_ptr<int>(),
                     k2q_idx.data_ptr<int>()};
    launch_row_topk(tp, w, cur_stream());
    check_launch();
  }
  return {q2k_ptr, q2k_idx, k2q_ptr, k2q_idx};
}

// ------------------------------------------------------------------ attention
std::vector<torch::Tensor> attn_fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor row_ptr,
                                    torch::Tensor col_idx, torch::Tensor blk_cnt, bool use_tk) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v must share [B,H,L,D] (MHA self-attention)");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  TORCH_CHECK(blk_cnt.numel() == NB && row_ptr.numel() == (long long)B * H * (NB + 1), "metadata shape mismatch");
  auto o = torch::empty_like(q);
  auto lse = torch::empty({B, H, L}, q.options().dtype(torch::kFloat32));
  if (use_tk) {
#ifdef VSA2_WITH_TK
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "TK backend is bf16 only");
    vsa2_tk_fine_fwd(q, k, v, o, lse, row_ptr, col_idx, blk_cnt);
    check_launch();
    return {o, lse};
#else
    TORCH_CHECK(false, "VSA2 was built without the SM90 ThunderKittens backend (VSA2_WITH_TK=0)");
#endif
  }
  AttnFwdParams p{q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), lse.data_ptr<float>(),
                  row_ptr.data_ptr<int>(), col_idx.data_ptr<int>(), blk_cnt.data_ptr<int>(), B * H, L, NB,
                  1.f / std::sqrt((float)D), kLog2eHost() / std::sqrt((float)D)};
  launch_attn_fwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {o, lse};
}

// Generic backward. dq_acc (fp32, zeroed by caller) is accumulated in place; without it dQ is
// not computed (see attn_bwd_q). Returns (dk, dv) as fp32 if out_fp32 else in q's dtype.
// add_dk/add_dv are optional per-block [B,H,add_nb,D] fp32 gradients distributed as add/cnt.
std::vector<torch::Tensor> attn_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout,
                                    torch::Tensor lse, torch::Tensor delta, torch::Tensor row_ptr,
                                    torch::Tensor col_idx, torch::Tensor blk_cnt, c10::optional<torch::Tensor> dq_acc,
                                    c10::optional<torch::Tensor> add_dk, c10::optional<torch::Tensor> add_dv,
                                    bool out_fp32, bool dq_word) {
  check_tokens(q, "q");
  CHECK_DEV(dout);
  CHECK_DEV(lse);
  CHECK_DEV(delta);
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  if (dq_acc.has_value()) {
    CHECK_DEV(*dq_acc);
    TORCH_CHECK(dq_acc->scalar_type() == torch::kFloat32 && dq_acc->sizes() == q.sizes(), "dq_acc must be fp32 like q");
  }
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  torch::Tensor dk, dv;
  if (out_fp32) {
    dk = torch::empty(q.sizes(), q.options().dtype(torch::kFloat32));
    dv = torch::empty(q.sizes(), q.options().dtype(torch::kFloat32));
  } else {
    dk = torch::empty_like(q);
    dv = torch::empty_like(q);
  }
  AttnBwdParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  p.dq_acc = dq_acc.has_value() ? dq_acc->data_ptr<float>() : nullptr;
  p.dq_word = dq_word;
  if (out_fp32) {
    p.dk_f32 = dk.data_ptr<float>();
    p.dv_f32 = dv.data_ptr<float>();
  } else {
    p.dk = dk.data_ptr();
    p.dv = dv.data_ptr();
  }
  if (add_dk.has_value()) {
    CHECK_DEV(*add_dk);
    p.add_dk = add_dk->data_ptr<float>();
    p.add_nb = add_dk->size(2);
  }
  if (add_dv.has_value()) {
    CHECK_DEV(*add_dv);
    p.add_dv = add_dv->data_ptr<float>();
    p.add_nb = add_dv->size(2);
  }
  p.row_ptr = row_ptr.data_ptr<int>();
  p.col_idx = col_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = kLog2eHost() / std::sqrt((float)D);
  launch_attn_bwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {dk, dv};
}

// dQ from the q->kv lists with no atomics; add_dq is an optional per-block fp32 gradient.
torch::Tensor attn_bwd_q(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout, torch::Tensor lse,
                         torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                         c10::optional<torch::Tensor> add_dq) {
  check_tokens(q, "q");
  CHECK_DEV(dout);
  CHECK_DEV(lse);
  CHECK_DEV(delta);
  CHECK_I32(row_ptr);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  auto dq = torch::empty_like(q);
  AttnBwdQParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  p.dq = dq.data_ptr();
  if (add_dq.has_value()) {
    CHECK_DEV(*add_dq);
    TORCH_CHECK(add_dq->scalar_type() == torch::kFloat32, "add_dq must be fp32");
    p.add_dq = add_dq->data_ptr<float>();
    p.add_nb = add_dq->size(2);
  }
  p.row_ptr = row_ptr.data_ptr<int>();
  p.col_idx = col_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = kLog2eHost() / std::sqrt((float)D);
  launch_attn_bwd_q(p, D, dtype_of(q), cur_stream());
  check_launch();
  return dq;
}

// ------------------------------------------------------------------ split-KV (experimental)
// Plan for the CSR list `ptr` [B,H,NB+1]: returns (tasks [NR+max_slots, 4], merges
// [max(1, min(NR, max_slots/2)), 4], info [8]: total tasks, split rows, split slots, NR, eligible
// rows, within budget, global threshold, edges), all on the device (no synchronization). Policy
// arguments (vsa2_kernels.h, SplitPlanParams) default to the manual rule: every row with n > chunk.
std::vector<torch::Tensor> split_plan(torch::Tensor ptr, int64_t chunk, int64_t max_slots, int64_t min_len,
                                      double rel_mean, double tail_frac, int64_t resident, int64_t budget) {
  CHECK_I32(ptr);
  TORCH_CHECK(ptr.dim() == 3 && ptr.size(2) >= 2, "ptr must be [B, H, NB+1]");
  TORCH_CHECK(chunk >= 1 && chunk <= (1 << 30), "split chunk must be in [1, 2^30]");
  TORCH_CHECK(max_slots >= 0 && max_slots < (1LL << 30), "bad max_slots");
  TORCH_CHECK(rel_mean >= 0.0 && tail_frac >= 0.0 && resident >= 1, "bad split policy");
  const c10::cuda::OptionalCUDAGuard guard(ptr.device());
  const int BH = ptr.size(0) * ptr.size(1), NB = ptr.size(2) - 1;
  const int64_t NR = (int64_t)BH * NB;
  TORCH_CHECK(NR + max_slots < (1LL << 31), "too many split-KV tasks");
  auto i32 = ptr.options().dtype(torch::kInt32);
  auto cnt = torch::empty({2, NR}, i32);
  auto flags = torch::empty({3, NR}, i32);
  auto tasks = torch::empty({NR + max_slots, 4}, i32);
  auto merges = torch::empty({std::max<int64_t>(1, std::min<int64_t>(NR, max_slots / 2)), 4}, i32);
  auto info = torch::empty({8}, i32);
  SplitPlanParams p{};
  p.ptr = ptr.data_ptr<int>();
  p.BH = BH;
  p.NB = NB;
  p.chunk = (int)chunk;
  p.max_slots = (int)max_slots;
  p.min_len = (int)std::max<int64_t>(chunk + 1, std::min<int64_t>(min_len, (1LL << 31) - 1));
  p.rel_mean = rel_mean;
  p.tail_frac = tail_frac;
  p.resident = (int)std::min<int64_t>(resident, 1 << 30);
  p.budget = budget < 0 ? INT_MAX : (int)std::min<int64_t>(budget, INT_MAX);
  p.cnt = cnt.data_ptr<int>();
  p.flags = flags.data_ptr<int>();
  p.tasks = tasks.data_ptr<int>();
  p.merges = merges.data_ptr<int>();
  p.info = info.data_ptr<int>();
  launch_split_count(p, cur_stream());
  check_launch();
  auto cnt_incl = torch::cumsum(cnt, 1, torch::kInt32);
  p.cnt_incl = cnt_incl.data_ptr<int>();
  launch_split_mark(p, cur_stream());
  check_launch();
  auto flags_incl = torch::cumsum(flags, 1, torch::kInt32);
  p.flags_incl = flags_incl.data_ptr<int>();
  launch_split_fill(p, cur_stream());
  check_launch();
  return {tasks, merges, info};
}

static void check_plan(const torch::Tensor& tasks, const torch::Tensor& info) {
  CHECK_I32(tasks);
  CHECK_I32(info);
  TORCH_CHECK(tasks.dim() == 2 && tasks.size(1) == 4 && info.numel() >= 4, "tasks/info must come from split_plan");
}

static void check_part(const torch::Tensor& part, int64_t D, const char* name) {
  CHECK_DEV(part);
  TORCH_CHECK(part.scalar_type() == torch::kFloat32, name, " must be fp32");
  TORCH_CHECK(part.numel() == 0 || (part.dim() == 3 && part.size(1) == kBlock && part.size(2) == D), name,
              " must be [slots, 64, D]");
}

// Fine forward over planned tasks. Direct rows are final in (o, lse); split rows hold partials in
// part_o [slots,64,D] / part_lse [slots,64] until split_merge_fwd.
std::vector<torch::Tensor> attn_fwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor col_idx,
                                          torch::Tensor blk_cnt, torch::Tensor tasks, torch::Tensor info,
                                          torch::Tensor part_o, torch::Tensor part_lse) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  check_plan(tasks, info);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v must share [B,H,L,D] (MHA self-attention)");
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  check_part(part_o, D, "part_o");
  CHECK_DEV(part_lse);
  TORCH_CHECK(part_lse.numel() == part_o.numel() / D, "part_lse must be [slots, 64]");
  TORCH_CHECK(blk_cnt.numel() == NB, "metadata shape mismatch");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto lse = torch::empty({B, H, L}, q.options().dtype(torch::kFloat32));
  AttnFwdParams p{q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), lse.data_ptr<float>(), nullptr,
                  col_idx.data_ptr<int>(), blk_cnt.data_ptr<int>(), B * H, L, NB,
                  1.f / std::sqrt((float)D), kLog2eHost() / std::sqrt((float)D)};
  p.tasks = tasks.data_ptr<int>();
  p.task_info = info.data_ptr<int>();
  p.n_tasks_cap = (int)tasks.size(0);
  p.part_o = part_o.numel() ? part_o.data_ptr<float>() : nullptr;
  p.part_lse = part_lse.numel() ? part_lse.data_ptr<float>() : nullptr;
  launch_attn_fwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {o, lse};
}

// Writes the split rows of (o, lse) in place from the partials.
void split_merge_fwd(torch::Tensor part_o, torch::Tensor part_lse, torch::Tensor merges, torch::Tensor info,
                     torch::Tensor blk_cnt, torch::Tensor o, torch::Tensor lse, bool mask_rows) {
  check_tokens(o, "o");
  CHECK_DEV(lse);
  CHECK_I32(merges);
  CHECK_I32(info);
  CHECK_I32(blk_cnt);
  const int L = o.size(2), D = o.size(3);
  check_part(part_o, D, "part_o");
  CHECK_DEV(part_lse);
  if (part_o.numel() == 0) return;  // no slots: nothing can be split
  const c10::cuda::OptionalCUDAGuard guard(o.device());
  SplitMergeFwdParams p{};
  p.merges = merges.data_ptr<int>();
  p.info = info.data_ptr<int>();
  p.n_merge_cap = (int)merges.size(0);
  p.part_o = part_o.data_ptr<float>();
  p.part_lse = part_lse.data_ptr<float>();
  p.o = o.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.L = L;
  p.NB = L / kBlock;
  p.mask_rows = mask_rows ? 1 : 0;
  launch_split_merge_fwd(p, D, dtype_of(o), cur_stream());
  check_launch();
}

// Writes the split rows/columns of `out` (Elt or fp32 [B,H,L,D]) in place: sum of partials + add/cnt.
void split_merge_sum(torch::Tensor part, torch::Tensor merges, torch::Tensor info, c10::optional<torch::Tensor> add,
                     torch::Tensor blk_cnt, torch::Tensor out) {
  CHECK_DEV(out);
  TORCH_CHECK(out.dim() == 4 && out.size(2) % kBlock == 0, "out must be [B,H,L,D]");
  CHECK_I32(merges);
  CHECK_I32(info);
  CHECK_I32(blk_cnt);
  const int L = out.size(2), D = out.size(3);
  check_part(part, D, "part");
  if (part.numel() == 0) return;
  const c10::cuda::OptionalCUDAGuard guard(out.device());
  SplitMergeSumParams p{};
  p.merges = merges.data_ptr<int>();
  p.info = info.data_ptr<int>();
  p.n_merge_cap = (int)merges.size(0);
  p.part = part.data_ptr<float>();
  if (add.has_value()) {
    CHECK_DEV(*add);
    TORCH_CHECK(add->scalar_type() == torch::kFloat32, "add must be fp32");
    p.add = add->data_ptr<float>();
    p.add_nb = add->size(2);
  }
  DType dt = DType::kBF16;
  if (out.scalar_type() == torch::kFloat32) {
    p.out_f32 = out.data_ptr<float>();
  } else {
    dt = dtype_of(out);
    p.out = out.data_ptr();
  }
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.L = L;
  p.NB = L / kBlock;
  launch_split_merge_sum(p, D, dt, cur_stream());
  check_launch();
}

// KV-owner fine backward over planned k2q tasks (attn_bwd semantics; dq_acc as there). Direct
// columns are final in (dk, dv); split columns hold partials until split_merge_sum.
std::vector<torch::Tensor> attn_bwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout,
                                          torch::Tensor lse, torch::Tensor delta, torch::Tensor col_idx,
                                          torch::Tensor blk_cnt, torch::Tensor tasks, torch::Tensor info,
                                          c10::optional<torch::Tensor> dq_acc, c10::optional<torch::Tensor> add_dk,
                                          c10::optional<torch::Tensor> add_dv, torch::Tensor part_dk,
                                          torch::Tensor part_dv, bool dq_word) {
  check_tokens(q, "q");
  CHECK_DEV(dout);
  CHECK_DEV(lse);
  CHECK_DEV(delta);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  check_plan(tasks, info);
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  check_part(part_dk, D, "part_dk");
  check_part(part_dv, D, "part_dv");
  TORCH_CHECK(part_dk.numel() == part_dv.numel(), "part_dk/part_dv mismatch");
  if (dq_acc.has_value()) {
    CHECK_DEV(*dq_acc);
    TORCH_CHECK(dq_acc->scalar_type() == torch::kFloat32 && dq_acc->sizes() == q.sizes(), "dq_acc must be fp32 like q");
  }
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  auto dk = torch::empty_like(q);
  auto dv = torch::empty_like(q);
  AttnBwdParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  p.dq_acc = dq_acc.has_value() ? dq_acc->data_ptr<float>() : nullptr;
  p.dq_word = dq_word;
  p.dk = dk.data_ptr();
  p.dv = dv.data_ptr();
  if (add_dk.has_value()) {
    CHECK_DEV(*add_dk);
    p.add_dk = add_dk->data_ptr<float>();
    p.add_nb = add_dk->size(2);
  }
  if (add_dv.has_value()) {
    CHECK_DEV(*add_dv);
    p.add_dv = add_dv->data_ptr<float>();
    p.add_nb = add_dv->size(2);
  }
  p.col_idx = col_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = kLog2eHost() / std::sqrt((float)D);
  p.tasks = tasks.data_ptr<int>();
  p.task_info = info.data_ptr<int>();
  p.n_tasks_cap = (int)tasks.size(0);
  p.part_dk = part_dk.numel() ? part_dk.data_ptr<float>() : nullptr;
  p.part_dv = part_dv.numel() ? part_dv.data_ptr<float>() : nullptr;
  launch_attn_bwd(p, D, dtype_of(q), cur_stream());
  check_launch();
  return {dk, dv};
}

// Q-owner dQ over planned q2k tasks (attn_bwd_q semantics). Split rows hold partials in part_dq.
torch::Tensor attn_bwd_q_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout,
                               torch::Tensor lse, torch::Tensor delta, torch::Tensor col_idx, torch::Tensor blk_cnt,
                               torch::Tensor tasks, torch::Tensor info, c10::optional<torch::Tensor> add_dq,
                               torch::Tensor part_dq) {
  check_tokens(q, "q");
  CHECK_DEV(dout);
  CHECK_DEV(lse);
  CHECK_DEV(delta);
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  check_plan(tasks, info);
  const int B = q.size(0), H = q.size(1), L = q.size(2), D = q.size(3), NB = L / kBlock;
  check_part(part_dq, D, "part_dq");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  auto dq = torch::empty_like(q);
  AttnBwdQParams p{};
  p.q = q.data_ptr();
  p.k = k.data_ptr();
  p.v = v.data_ptr();
  p.dout = dout.data_ptr();
  p.lse = lse.data_ptr<float>();
  p.delta = delta.data_ptr<float>();
  p.dq = dq.data_ptr();
  if (add_dq.has_value()) {
    CHECK_DEV(*add_dq);
    TORCH_CHECK(add_dq->scalar_type() == torch::kFloat32, "add_dq must be fp32");
    p.add_dq = add_dq->data_ptr<float>();
    p.add_nb = add_dq->size(2);
  }
  p.col_idx = col_idx.data_ptr<int>();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = NB;
  p.scale = 1.f / std::sqrt((float)D);
  p.scale_log2 = kLog2eHost() / std::sqrt((float)D);
  p.tasks = tasks.data_ptr<int>();
  p.task_info = info.data_ptr<int>();
  p.n_tasks_cap = (int)tasks.size(0);
  p.part_dq = part_dq.numel() ? part_dq.data_ptr<float>() : nullptr;
  launch_attn_bwd_q(p, D, dtype_of(q), cur_stream());
  check_launch();
  return dq;
}

// Limits that decide CTAs per SM for the fixed shared-memory tiles (benchmark metadata).
std::map<std::string, int64_t> device_limits() {
  cudaDeviceProp d{};
  cudaError_t e = cudaGetDeviceProperties(&d, (int)c10::cuda::current_device());
  TORCH_CHECK(e == cudaSuccess, "cudaGetDeviceProperties failed: ", cudaGetErrorString(e));
  return {{"sharedMemPerBlock", (int64_t)d.sharedMemPerBlock},
          {"sharedMemPerBlockOptin", (int64_t)d.sharedMemPerBlockOptin},
          {"sharedMemPerMultiprocessor", (int64_t)d.sharedMemPerMultiprocessor},
          {"maxThreadsPerMultiProcessor", (int64_t)d.maxThreadsPerMultiProcessor},
          {"multiProcessorCount", (int64_t)d.multiProcessorCount}};
}

#ifdef VSA2_WITH_TK
// TK fine forward over planned q2k tasks. Direct rows are final in (o, lse) (TK conventions); split rows hold
// partials in part_o [slots >= 1, 64, D] / part_lse [slots, 64] until split_merge_fwd(..., mask_rows=False).
std::vector<torch::Tensor> tk_fine_fwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                             torch::Tensor col_idx, torch::Tensor blk_cnt, torch::Tensor tasks,
                                             torch::Tensor info, torch::Tensor part_o, torch::Tensor part_lse) {
  check_tokens(q, "q");
  check_tokens(k, "k");
  check_tokens(v, "v");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "TK backend is bf16 only");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q/k/v must share [B,H,L,D] (MHA self-attention)");
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  check_plan(tasks, info);
  const int D = q.size(3);
  check_part(part_o, D, "part_o");
  TORCH_CHECK(part_o.size(0) >= 1, "TK split needs part_o with at least one slot (a TMA descriptor over it)");
  CHECK_DEV(part_lse);
  TORCH_CHECK(part_lse.scalar_type() == torch::kFloat32 && part_lse.numel() == part_o.numel() / D,
              "part_lse must be fp32 [slots, 64]");
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto lse = torch::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(torch::kFloat32));
  vsa2_tk_fine_fwd_tasks(q, k, v, o, lse, col_idx, blk_cnt, tasks, info, part_o, part_lse);
  check_launch();
  return {o, lse};
}

static void check_tk_bwd_opts(int64_t opts) {
  // Bit 8 (kMinb3, a launch-bound probe of the representative experiment) is not exposed here.
  TORCH_CHECK(opts >= 0 && (opts & ~int64_t(1 | 2 | 4 | 16)) == 0,
              "TK backward opts are bits 1 (dQ read-wait), 2 (rotate), 4 (skip dQ, diagnostic), "
              "16 (single-writer dK/dV store; unsplit launch only)");
}

// TK fine backward over planned k2q tasks; split chunks reduce-add into the same fp32 accumulators. The kernel
// ignores bit 16 here (several tasks can own one kv block), so kg / vg must be zeroed for every option.
void tk_fine_bwd_tasks(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                       torch::Tensor delta, torch::Tensor col_idx, torch::Tensor blk_cnt, torch::Tensor tasks,
                       torch::Tensor info, torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, int64_t opts) {
  check_tk_bwd_opts(opts);
  check_tokens(q, "q");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "TK backend is bf16 only");
  CHECK_I32(col_idx);
  CHECK_I32(blk_cnt);
  check_plan(tasks, info);
  for (const auto* t : {&qg, &kg, &vg})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == torch::kFloat32 && t->sizes() == q.sizes(),
                "TK accumulators must be zeroed fp32 tensors like q");
  vsa2_tk_fine_bwd_tasks(q, k, v, dout_f, lse, delta, col_idx, blk_cnt, tasks, info, qg, kg, vg, (int)opts);
  check_launch();
}

void tk_fine_bwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor dout_f, torch::Tensor lse,
                 torch::Tensor delta, torch::Tensor row_ptr, torch::Tensor col_idx, torch::Tensor blk_cnt,
                 torch::Tensor qg, torch::Tensor kg, torch::Tensor vg, int64_t opts) {
  check_tokens(q, "q");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "TK backend is bf16 only");
  check_tk_bwd_opts(opts);
  vsa2_tk_fine_bwd(q, k, v, dout_f, lse, delta, row_ptr, col_idx, blk_cnt, qg, kg, vg, (int)opts);
  check_launch();
}
#endif

bool has_tk() {
#ifdef VSA2_WITH_TK
  return true;
#else
  return false;
#endif
}

// ------------------------------------------------------------------ fusion
static int gate_mode_of(const c10::optional<torch::Tensor>& gate, const torch::Tensor& like) {
  if (!gate.has_value()) return 0;
  CHECK_DEV(*gate);
  TORCH_CHECK(gate->scalar_type() == like.scalar_type(), "gate must have the token dtype");
  if (gate->dim() == 3) {
    TORCH_CHECK(gate->size(0) == like.size(0) && gate->size(1) == like.size(1) && gate->size(2) == like.size(2),
                "per-token gate must be [B,H,L]");
    return 1;
  }
  TORCH_CHECK(gate->sizes() == like.sizes(), "per-element gate must be [B,H,L,D]");
  return 2;
}

torch::Tensor fuse_fwd(torch::Tensor of, torch::Tensor oc, c10::optional<torch::Tensor> gate, torch::Tensor blk_cnt,
                       int64_t gate_act) {
  check_tokens(of, "o_f");
  CHECK_DEV(oc);
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(of.device());
  const int B = of.size(0), H = of.size(1), L = of.size(2), D = of.size(3);
  auto o = torch::empty_like(of);
  FuseFwdParams p{of.data_ptr(), oc.data_ptr(), gate.has_value() ? gate->data_ptr() : nullptr, o.data_ptr(),
                  blk_cnt.data_ptr<int>(), B * H, L, L / kBlock, (int)oc.size(2), gate_mode_of(gate, of),
                  (int)gate_act};
  launch_fuse_fwd(p, D, dtype_of(of), cur_stream());
  check_launch();
  return o;
}

// returns (dout_f, delta_f [B,H,L], dout_c [B,H,NBc_pad,D], delta_c [B,H,NBc_pad], dgate or empty)
// write_dout_f=false: the generic fine backward reads dO directly (it zeroes P for padded
// and empty rows), saving one [B,H,L,D] write+read; the TK backward needs the masked copy.
std::vector<torch::Tensor> fuse_bwd(torch::Tensor dout, torch::Tensor of, torch::Tensor oc,
                                    c10::optional<torch::Tensor> gate, torch::Tensor blk_cnt, int64_t gate_act,
                                    bool write_dout_f) {
  check_tokens(of, "o_f");
  CHECK_DEV(dout);
  CHECK_DEV(oc);
  CHECK_I32(blk_cnt);
  TORCH_CHECK(dout.sizes() == of.sizes() && dout.scalar_type() == of.scalar_type(), "dout must match o");
  const c10::cuda::OptionalCUDAGuard guard(of.device());
  const int B = of.size(0), H = of.size(1), L = of.size(2), D = of.size(3), NBc_pad = oc.size(2);
  const int mode = gate_mode_of(gate, of);
  auto dout_f = write_dout_f ? torch::empty_like(of) : torch::empty({0}, of.options());
  auto delta_f = torch::empty({B, H, L}, of.options().dtype(torch::kFloat32));
  auto dout_c = torch::zeros({B, H, NBc_pad, D}, of.options());
  auto delta_c = torch::zeros({B, H, NBc_pad}, of.options().dtype(torch::kFloat32));
  torch::Tensor dgate = mode ? torch::empty_like(*gate) : torch::empty({0}, of.options());
  FuseBwdParams p{dout.data_ptr(), of.data_ptr(), oc.data_ptr(), mode ? gate->data_ptr() : nullptr,
                  write_dout_f ? dout_f.data_ptr() : nullptr, delta_f.data_ptr<float>(), dout_c.data_ptr(), delta_c.data_ptr<float>(),
                  mode ? dgate.data_ptr() : nullptr, blk_cnt.data_ptr<int>(), B * H, L, L / kBlock, NBc_pad, mode,
                  (int)gate_act};
  launch_fuse_bwd(p, D, dtype_of(of), cur_stream());
  check_launch();
  return {dout_f, delta_f, dout_c, delta_c, dgate};
}

// out[t] = valid ? acc[t] + add[blk]/cnt : 0 in `like`'s dtype
torch::Tensor finalize(c10::optional<torch::Tensor> acc, c10::optional<torch::Tensor> add, torch::Tensor blk_cnt,
                       torch::Tensor like) {
  check_tokens(like, "like");
  CHECK_I32(blk_cnt);
  const c10::cuda::OptionalCUDAGuard guard(like.device());
  const int B = like.size(0), H = like.size(1), L = like.size(2), D = like.size(3);
  auto out = torch::empty_like(like);
  FinalizeParams p{};
  if (acc.has_value()) {
    CHECK_DEV(*acc);
    TORCH_CHECK(acc->scalar_type() == torch::kFloat32 && acc->sizes() == like.sizes(), "acc must be fp32 like out");
    p.acc = acc->data_ptr<float>();
  }
  if (add.has_value()) {
    CHECK_DEV(*add);
    TORCH_CHECK(add->scalar_type() == torch::kFloat32, "add must be fp32");
    p.add = add->data_ptr<float>();
    p.add_nb = add->size(2);
  }
  p.out = out.data_ptr();
  p.blk_cnt = blk_cnt.data_ptr<int>();
  p.BH = B * H;
  p.L = L;
  p.NB = L / kBlock;
  launch_finalize(p, D, dtype_of(like), cur_stream());
  check_launch();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Native VSA / VSA2 CUDA kernels (SM90 / SM120)";
  m.def("pool_fwd", &pool_fwd);
  m.def("router_lse", &router_lse);
  m.def("router_scores", &router_scores);
  m.def("global_topk_into", &global_topk_into);
  m.def("route", &route);
  m.def("block_scores", &block_scores);
  m.def("row_topk_into", &row_topk_into);
  m.def("route_rows", &route_rows);
  m.def("attn_fwd", &attn_fwd);
  m.def("attn_bwd", &attn_bwd);
  m.def("attn_bwd_q", &attn_bwd_q);
  m.def("device_limits", &device_limits);
  m.def("fuse_fwd", &fuse_fwd);
  m.def("fuse_bwd", &fuse_bwd, py::arg("dout"), py::arg("of"), py::arg("oc"), py::arg("gate"), py::arg("blk_cnt"),
        py::arg("gate_act"), py::arg("write_dout_f") = true);
  m.def("finalize", &finalize);
  m.def("has_tk", &has_tk);
  m.def("split_plan", &split_plan, py::arg("ptr"), py::arg("chunk"), py::arg("max_slots"), py::arg("min_len") = 0,
        py::arg("rel_mean") = 0.0, py::arg("tail_frac") = 0.0, py::arg("resident") = 1, py::arg("budget") = -1);
  m.def("attn_fwd_tasks", &attn_fwd_tasks);
  m.def("split_merge_fwd", &split_merge_fwd, py::arg("part_o"), py::arg("part_lse"), py::arg("merges"),
        py::arg("info"), py::arg("blk_cnt"), py::arg("o"), py::arg("lse"), py::arg("mask_rows") = true);
  m.def("split_merge_sum", &split_merge_sum);
  m.def("attn_bwd_tasks", &attn_bwd_tasks);
  m.def("attn_bwd_q_tasks", &attn_bwd_q_tasks);
  m.def("set_fwd_variant", &vsa2::set_fwd_variant);
  m.def("get_fwd_variant", &vsa2::get_fwd_variant);
#ifdef VSA2_WITH_TK
  m.def("tk_fine_bwd", &tk_fine_bwd, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dout_f"), py::arg("lse"),
        py::arg("delta"), py::arg("row_ptr"), py::arg("col_idx"), py::arg("blk_cnt"), py::arg("qg"), py::arg("kg"),
        py::arg("vg"), py::arg("opts") = 0);
  m.def("tk_fine_fwd_tasks", &tk_fine_fwd_tasks);
  m.def("tk_fine_bwd_tasks", &tk_fine_bwd_tasks, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dout_f"),
        py::arg("lse"), py::arg("delta"), py::arg("col_idx"), py::arg("blk_cnt"), py::arg("tasks"), py::arg("info"),
        py::arg("qg"), py::arg("kg"), py::arg("vg"), py::arg("opts") = 0);
#endif
}
