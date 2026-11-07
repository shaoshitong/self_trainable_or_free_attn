# VSA2 representative TK/TMA on H100 — 优化总结

面向 `vsa2-representative`（64/4/1 三层近似 attention，ThunderKittens + TMA，sm90a）。
所有数字均为 H100 80GB HBM3 上同 build 的 A/B，A/A 对照，计时前必须通过 CPU / generic / TK 三套件。
ncu 无权限（`ERR_NVGPUCTRPERM`），因此没有 MFU / 带宽 / 占用率实测，凡涉及这些的推断都已标注。

---

## 1. 采纳了两处改动

### A. tier-B 的 gather 换搬运引擎（R4 采纳，R5 升为默认）

原来每个 B tile 发 32 次 512 B 的 `cp.async.bulk`（16 slot × 2 panel，一个 warp 发、一个 mbarrier 收）。
改成 **128 线程各发 16 B 的 `cp.async.cg`**，字节、预 swizzle 记录、smem 布局、算术全部不变。

| 行 | 旧 | 新 | 比值 |
|---|---:|---:|---:|
| `kernel.tk.fwd.B` | 0.2154 | 0.1654 | **0.768** |
| `kernel.tk.fwd.A\|B\|C` | 0.8500 | 0.7788 | **0.916** |
| `fine_op` 前向 | 0.9063 | 0.8067 | **0.890** |
| `full_op` 前向 | 1.3704 | 1.2738 | **0.930** |
| `full_op` 前向+反向 | 4.1789 | 4.0843 | **0.977** |

32K 上前向收益更大（`A|B|C` 0.899、full 前向 0.933），f+b 只有 0.990，因为 32K 时反向占 9.65 / 13.0 ms。
副作用：ptxas 从 168 regs + 36/42 B spill 变成 **166 regs、0 spill**。TK 套件 349 项全过。

### B. tier-A dK/dV 改成单写者 store（R11 采纳，R12 升为默认）

routed 反向里 `qg`/`kg`/`vg` 三张表被 memset（384 MiB）。在 row_ptr 路径上每个 kv 块只有一个 owner CTA，
所以 tier-A kernel 可以直接 **store** 自己的 dK/dV 行，`kg`/`vg` 的两个 memset 随之取消。

| 行 | 旧 | 新 | 比值 |
|---|---:|---:|---:|
| `fine_op ... b_direct` | 3.0290 | 2.9294 | **0.9671** |
| `full_op ... b_direct` | 2.9389 | 2.8597 | **0.9731** |
| `fine_op ... f_plus_b` | 3.7521 | 3.6857 | **0.9823** |
| `full_op ... f_plus_b` | 4.1363 | 4.0495 | **0.9790** |

升为默认后再验一次：旧路径（CFG37）在 `b_direct` 上是 1.0370、`f+b` 1.0254，即旧路径现在是明确的回退。

---

## 1b. 后续两处采纳（都不在 kernel 算术里）

**C. coarse CSR 元数据按形状缓存（R16，默认）**：`_coarse_meta` 每次都从 `(B,H,nb,nbc_pad)` 重建三个小 int32
张量，与数据无关。缓存后 `full_op f` 1.2878 → 1.2542（0.9739）、`f+b` 0.9904；升默认后再验，旧路径（CFG40）
在 `full_op f` 上是 1.0316、`f+b` 1.0115。

**D. 代表表改用 router 的池化输出（R19，默认）**：`rep_tk_build` 实测 0.061 ms，读 K+V 共 128 MiB 去重算
`pool_kernel` 已经算过的均值。`pool_fwd_vr` 在同一次 pass 里补出 16-token 的 V 组均值（4 MiB），
`build_kernel_pooled` 只做重排、不再读 K/V。`full_op f` 1.2494 → 1.2035（0.9633）、`f+b` 0.9905；
升默认后再验，旧路径（CFG43）`full_op f` 1.0245、`f+b` 1.0058。**容差等价、非逐位等价**（组均值先舍入到
bf16 再合成块均值），验收走套件 + 对 generic 后端的输出交叉校验。

**E. 两条 top-k 预算链并行（R23，默认）**：`route` 里 A|B 与 A 两条链在拿到 `scores` 之后互不依赖，各给一份
workspace 与 k2q scratch，一发主 stream、一发副 stream。`full_op f` 0.9323/0.9463（两组独立 A/B）、
`f+b` 0.9777/0.9797；升默认后再验，顺序路径（CFG45）是 1.0739/1.0175。曾经因为**每次调用都 create/destroy
stream 与 event** 而测不出收益，改成进程内 static 后才稳定。

**F. router 列阶段三 kernel 合一（R30，默认）**：`topk_cols` / `topk_colscan` / `topk_cols` 合成
`topk_cols_fused_kernel`，一条链少两次 launch。列阶段只产整数，kv→q CSR 与 CFG1 逐位相同。
`full_op f` 0.9867、`f+b` 0.9967；升默认后再验，旧路径（CFG47）1.0104/1.0032。

## 2. 当前账单（16K/H16/D128/RM4，最新 build）

| 阶段 | ms |
|---|---:|
| full 前向 | 1.293 |
| ├ `fine`（kernel 0.78 + tables 0.07） | ~0.84 |
| ├ `route`（含两次 top-k 与 tier-B 列表） | 0.262 |
| ├ `pool` | 0.098 |
| ├ `coarse`（meta + attn） | 0.106 |
| └ `fuse` | 0.051 |
| full 反向 | 2.850 |
| ├ host/autograd 到第一个 kernel 的间隙 | 0.118 |
| ├ `fuse_bwd` | 0.099 |
| ├ token 表 memset（只剩 `qg`） | 0.045 |
| ├ tier-A `bwd_attend_ker` | 1.531 |
| ├ tier-B | 0.494 |
| ├ tier-C | 0.311 |
| ├ `finalize`（dQ fp32→bf16） | 0.073 |
| └ `rep_unpool`（dK/dV fp32→bf16） | 0.172 |
| **full f+b** | **~4.03** |

反向那 0.62 ms 的"记账"已经被逐项拆开：大部分是必要带宽（fuse_bwd、finalize、unpool 都是 fp32→bf16 转换
或 gate 梯度），真正能拿掉的只有 kg/vg 的 memset（已被 B 拿走）和 0.118 ms 的 host 延迟。

---

## 2b. 这批判定是怎么被验证的（以及一个必须知道的坑）

- **只用同一次运行内的 A/B 比值**。这台机器跨运行有约 2% 漂移，两组 A/A 之间都能差到 2%。
- **这个算子的反向本身不可复现**：同进程、同输入跑两次，`o` 与选择位图逐位相同，但 `dq`/`dk`/`dv` 各有
  300–1200 个元素（共 3355 万）不同——跨 CTA 的 fp32 归约顺序不定。所以任何以梯度为证据的改动只能比容差，
  不能比逐位。第 13 轮否掉整算子 CUDA graph 时用的"梯度差 1 ulp"这条理由，因此作废（性能理由仍成立）。
- **harness 曾有六处"看起来在检查、其实没检查"**，最严重的一处是：输出交叉校验从未覆盖 routed 算子，只测了
  fine 路径。也就是说 pooled 代表表与融合列阶段最初"通过"的其实是位图校验，不是输出校验。修好之后三个已
  采纳臂（CFG42/44/46）在真正覆盖 routed 的检查下**复验通过**。
- 每个采纳的改动都配一个"恢复旧行为"的 CFG（20/37/40/43/45/47），可在同一份源码上重跑 A/B。

## 3. 十条被证伪的路线（都带 A/A 地板，别重复）

| # | 手段 | 结果 | 原因 |
|---|---|---|---|
| 1 | 向量化 softmax（R2） | 1.0018 | 指令数不是瓶颈 |
| 2 | MINB=2 提寄存器上限（R3） | 前向 +22.8%，f+b +5.0% | 掉了 CTA，spill 不是主因 |
| 3 | MINB=4（早期） | 0.723 vs 0.521 | 大量 spill |
| 4 | K 单边双缓冲（R5B） | 1.0003 | 3 CTA 已隐藏 K 延迟 |
| 5 | tier-C split 策略（R6） | wave 1.0002 / 强制 4 1.0013 | 16K 下三种策略等价 |
| 6 | tier-A `launch_bounds(128,3)`（R6） | f+b 1.350，`bwd.A` 1.874 | 113 KB smem 本就放不下 3 CTA，只测到 spill |
| 7 | 两段式 softmax（R7） | kernel 1.292，f+b 1.063 | 保持 3 CTA 也输，额外状态与重排的等待 |
| 8 | 2 CTA + 2/3 深 K/V ring（R8） | 1.076 / 1.110 | 用占用率换深度必然净亏 |
| 9 | softmax 指令削减：FFMA 融合 / 跳过 rescale（R9） | 0.995 / 0.996，落在噪声 | 3 CTA 交错吸收 epilogue |
| 10 | CUDA graph（R13） | 复用张量 3.940，**带新输入拷贝 4.118**（比 eager 4.045 还慢） | 输入拷贝吃掉 launch 收益 |

另外 **router 单直方图**经分析后放弃：一次 `launch_global_topk` 是 9 个小 kernel，两次调用 18 个，
而每个直方图 pass 只读 4 MiB —— 成本是 launch 延迟，不是冗余带宽，融合省下的三个 pass 没有意义。

---

## 4. 为什么到此为止（三条硬约束）

1. **176 寄存器悬崖**：128 线程 × 168 regs × 3 CTA = 64,512 / 65,536。多 8 个寄存器就掉到 2 CTA。
   所有"加 32 寄存器做流水线"的方案（#2、#7）都死在这里。
2. **3 CTA / 单缓冲就是 (占用率 × 深度) 网格的最优点**：完整网格为 3×1 → 0.78、3×2 → 0.78、
   2×2 → 0.86、2×3 → 0.89 ms。
3. **tier-A 反向是结构性的**：smem 里 64 KB 是 K/V/Q/dO 四个 bf16 tile，32 KB 是 **fp32 dQ 暂存累加器**，
   8 KB 是注意力 tile。3 CTA 要求 ≤75.6 KB，必须砍掉 35 KB，唯一够大的就是 dQ 暂存块——它之所以存在，
   是因为这个 kernel 是 KV-owner、dQ 要跨 CTA 累加。
   把它改成寄存器 A 操作数（RS wgmma）也不行：Q+dO 的 fragment 要 **+64 regs/thread**，而当前已经是 245，
   任何一档都会越过 255 的硬件上限。

---

## 5. 复现

- 源码：`vsa2-representative-opt-20261106` 工作树，最终 commit `6d38ccb`；分支
  `codex/vsa2-rep-tk-softmax-vecops-20261106`，标签 `vsa2-rep-accepted-20261106`。
- `REP_TK_FWD_CFG`：1/10（默认，gather + store）、20（R4 前的前向）、37（R11 前的反向）、
  11/24/32/33/34/35/36（负结果对照）、21/22/23（反向 split 与 launch-bound 臂）。
- 作业：`h100-r4 … h100-r13`（每个都带源清单逐文件 SHA-256、fresh build、套件门禁），观测在各自
  `private/latest.json`。
- 计时脚本：`bench_compare.py`（主）、`bench_stages.py`（前向分项）、`bench_bwd_stages.py`（反向分项）、
  `bench_graph.py`（graph 探针）。

---

## 6. 如果要继续，只有三条路（都是新项目，不是调参）

| 路线 | 预期 | 代价 |
|---|---|---|
| tier-A 反向改 query-owner | 可能 5–7% 的 f+b | 重写 VSA2 原生 kernel，dK/dV 改为跨 CTA 归约，需重验确定性与 PSNR |
| 32 行 query tile | 未知 | 每个 tile 算术强度减半，换 smem 余量；未见得划算 |
| 算法层：让多个 query tile 共享 KV 选择 | 最大 | 改近似本身，必须重验 PSNR |

六处采纳改动把 full 前向从 1.3704 降到约 1.128 ms（−17.7%），full f+b 从 4.1789 降到约 3.873 ms（−7.3%）；
每一处都带同 build 的 A/A 对照与套件门禁，后五处还各自做了"升默认后再验一次"（即旧路径对照）。再往前走需要上面三条之一，而不是继续在 kernel 指令层找。
