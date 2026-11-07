# 现在能做什么 —— VSA2 representative TK/TMA 优化行动清单

基线：`vsa2-rep-accepted-20261106`（commit `d23d31f`，已推送，默认 CFG1）。
所有数字来自 H100 同 build A/B，A/A 地板 ≤1.4%，套件门禁（CPU/generic/TK）在计时前全过。

---

## 0. 已完成、可复现的收益

| 16K | R4 前 | 现在 | 变化 |
|---|---:|---:|---:|
| tier-B kernel | 0.2154 | 0.1654 | **−23.2%** |
| `A\|B\|C` kernel | 0.8500 | 0.7788 | **−8.4%** |
| fine 前向 | 0.9063 | 0.8067 | **−11.0%** |
| full 前向 | 1.3704 | 1.2738 | **−7.0%** |
| full 前向+反向 | 4.1789 | 4.0843 | **−2.3%** |

32K：`A|B|C` 3.3072 → 2.9721（−10.1%）、full 前向 4.1134 → 3.8392（−6.7%）；f+b 只有 −1.0%，因为 32K 时反向占 9.65/13.0 ms。

**不要重试**（已实测证伪）：向量化 softmax、MINB=2、MINB=4、K 单边双缓冲、split 策略调参、
tier-A `launch_bounds(128,3)`、两段式 softmax。

---

## 1. P0 — 先花一轮把瓶颈钉死（不改算法、不占寄存器）

| 做什么 | 为什么 | 产出 | 成本 |
|---|---|---|---|
| **A29** `clock64()` 暴露等待直方图 | ncu 无权限，现在分不清"等 stage / 等 softmax / 等内存" | 每个 mbarrier 等待的周期分布 | 1 个 CFG + 1 次作业 |
| **A27** depth×occupancy 四点扫描 | 3 CTA/SM 下每 CTA 只有 76 KB，K+V 一个 stage 就 32 KB ⇒ **最多 2 stage**；这可能是流水线一直藏不住延迟的真因 | (3×2)/(3×4 半箱)/(2×3)/(2×4) 四条曲线 | 同上 |

**决策规则**：等待直方图里 mbarrier 占比 >40% → 走 A27 的深 stage 变体；softmax/ALU 占比 >40% → 走 B08+C 组；
两者都不高 → 说明已经在靠 L2 撑，走 P2。

---

## 2. P1 — 零新增寄存器的 kernel 改动（保数学，预期 5–15%）

| # | 改动 | 机制 | 预期 |
|---|---|---|---|
| **B08** | n32 半块 softmax + 双 S 缓冲 | S 从 32 regs 降 16，双缓冲合计 48 = 现在 32+16，TC 周期不变 | fwd 10%+ |
| **C01** | exp2 折叠（`log2e*scale` 并进 FFMA） | 每元素少一次乘 | 2–3% |
| **C13+C14** | 条件 rescale + max lag | max 没变就不乘 O；每 T 个 tile 刷新一次 | 5–8% |
| **B07** | wgmma commit-group 切分 + `wait_group 1` | 只等 QK，不等 PV 排空 | 2–5% |
| **A20** | P 双缓冲（smem，不用寄存器） | PV 与下一 QK 重叠 | 与 B07 叠加 |

前置：P0 的结果。若 P0 说 softmax 串行，则 B08 优先；若说 stage 不足，B07/A20 优先。

---

## 3. P2 — 调度与内存策略（不动 kernel，保数学，预期 5–15%）

| # | 改动 | 机制 | 成本 |
|---|---|---|---|
| **B14** | 选择感知的 CTA swizzle | 让同 SM 的 3 个 CTA 选择尽量重叠，KV 走 L2 | 要把选择元数据接进调度器 |
| **A28** | `accessPolicyWindow` 钉住表/Q + KV `discard.global.L2` | 只加 hint | 低 |
| **A06** | `cp.async.bulk.prefetch.tensor` 预取 i+2 | 把 HBM 延迟变 L2 延迟 | 低 |
| **B13** | 持久化 kernel + tile scheduler | 摊薄 prologue、削尾 | 中（mbarrier 相位要重置） |

依据：38% 利用率下 KV 需求约 5.8 TB/s > HBM 3.35 TB/s，说明已经在吃 L2 命中 —— 这条线是"不碰 kernel 的唯一大杠杆"。

---

## 4. P3 — 反向的记账开销（保数学，确定性最高）

反向 2.98 ms 里约 **0.6 ms 是纯记账**（建表/unpool/zero-init/delta/finalize）。目标吃掉一半：

| # | 改动 | 预期 |
|---|---|---|
| **D02** | mean-pool 伴随 `dK_j += (1/c)dS` 融进 epilogue | 省一次 global 往返 |
| **D03** | unpool 融进同一 kernel（smem 转置 scatter） | 少一个 kernel |
| **D04** | 去 zero-init，改 first-writer 谓词 + `red.add` | 省 memset（估算 ~115 µs） |
| **D05** | delta 融进首个持有 dO 的消费者 | 少一个 kernel |

合计目标：f+b 再降 5–8%。

---

## 5. P4 — 反向 tier-A 的 co-design（保数学，高风险）

**D01**：把 113 KB 动态 smem 压到 ≤76 KB（K/V 单缓冲、dK/dV 只留寄存器、向量化 st.global），
再用 `setmaxnreg` 让算术 warp 232–240、load warp 24–32，才可能真拿到 3 CTA/SM。
目标：tier A 1.55 → ~1.2 ms（f+b −8%）。风险：动的是 VSA2 共用 kernel，回归面大。

**前提**：P3 做完，否则先动它收益被记账开销吃掉。

---

## 5b. P6 — 让代表表构建与 routing 重叠（保数学，新发现）

实测 `fine.tables` 是 **0.070 ms**（full 前向 5.4%），而它在 `_rep_fwd_tk` 里紧挨着 kernel 执行、与
`route_tiers`（0.262 ms）完全串行。但两者**互不依赖**：`tk_tables(C, k, v, blk_cnt, sel, mid_reps)` 只需要
K/V、blk_cnt 和选择结果里的 tier-B 列表，而 route 的 0.262 ms 是一串小 kernel（launch 受限，GPU 并不饱和）。

做法：在 routed 前向里，`pool` 之后立刻把表构建发到**第二条 stream**，与 `route_tiers` 并行；fine kernel 之前
用 event 汇合。收益上限 = 0.070 ms（full 前向 −5.4%，f+b −1.7%），数学完全不变。

风险：多流同步正确性、分配器的 stream 语义（需要 `record_stream` 或显式同步）、以及 route 若其实占满 GPU
则没有重叠空间。**验收**：`fine.tables` 单独计时 + `full f` 的 A/B；如果 f 没变或变慢，说明 route 已占满。

## 5c. P7 — 用 pool 的输出来构代表表（容差等价，唯一有"读放大"证据的结构性改动）

实测：`fine.tables` 0.0692 ms 里 **`rep_tk_build` 占 0.0610 ms**，它读 K+V 共 128 MiB（≈2.1 TB/s，读受限）；
`rep_tk_lcb` 只有 0.0058 ms。

而 `pool_kernel` 已经算出 `kr`（16-token K 组均值 = tier-B 的 K 代表）、`kc`/`vc`（64-token 块均值 = tier-C 的
K/V 代表）。**只缺 tier-B 的 V 16-token 组均值。**

步骤：
1. `csrc/vsa/router/pool.cu`：在 `tensor == 2`（V）分支顺带写出 `vr`，**同一次 pass 内**，不要新起读 V 的 kernel；
   为不改 `pool_fwd` 的返回元数，新增 binding（如 `pool_fwd_vr`）返回 6 个张量。
2. `csrc/rep_tk/rep_tk.cu`：新增 `build_kernel_pooled<D>`，照抄 `build_kernel` 的 swizzle 写法
   （`sw_off(pi,b,g)`、`phi=(pi*RM)&7` 的 chunk XOR、`NB` 零记录、`Bt` 尾部行），只把累加 token 换成读组均值；
   块均值 = `Σ mean_g · n_g / cnt`。
3. 新 binding `rep_tk_build_pooled(kr, vr, kc, vc, blk_cnt, mid_reps)`，输出与原 7 个表同形。
4. `vsa2_rep/ops.py` 用 **CFG42** 门控，默认不动。
5. 预期省 **0.04–0.05 ms**（full 前向 3–4%）。

**注意**：这是容差等价，不是逐位等价——组均值先舍入到 bf16 再合成块均值，与现在"fp32 累加整块再舍入"不同。
验收用输出/PSNR 差异表，不能放 `FORWARD_EQUAL`；TK 套件必须在 CFG42 下跑。

## 5d. P8 — 让两个预算的 top-k 链并行（保数学，待 CC 实现）

实测：`route.one_budget` 0.162 ms（一条 top-k 链），`route` 0.262 ms（两条）。差值 0.100 ms ≈ 一条链的
14 次 launch × ~7 µs —— **两条链几乎全在 launch 上**（`launch_global_topk` 每次 14 个 kernel：memset、init、
3×(hist+find)、mask、cut、emit、cols、colscan、cols）。

两条链在给定 `scores` 后**互不依赖**：A|B 那条写 `ab_ptr/ab_idx` + `bits_ab`，A 那条写 `q2k_ptr/q2k_buf` +
`bits_a`。所以可以：

1. 给 A|B 那条单独的 `k2q` scratch（现在两条链共用 `k2q_buf`，并行会冲突）；
2. 给第二条链单独的 `TopkWorkspace`（现在是同一个 `ws`）；
3. 在 `route_tiers` 里把 A|B 发到主 stream、A 发到副 stream，join 后再做 bits 拷贝与 tier_b_lists。

预期：两条 14-kernel 链的 launch 间隙互相填充，**省 0.03–0.05 ms（full 前向 2.5–4%）**，数学不变。
风险：流同步与 scratch 复用，必须保证 `bits_ab`/`bits_a` 的 memcpy 在各自链之后、且 tier_b_lists 在 A|B
完成之后才跑。

## 6. P5 — router（保数学，需位精确证明）

**E01** 单直方图双阈值：一次统计出 top-10% 与 top-30% 两个切点，两张位图逐位相同。
收益 ~0.10 ms（full 前向 8%、f+b 2.5%）。验收条件：新旧位图在全部测试形状上逐位相等。

**E03/E04**（融合 pool+score、融合 tier_b_lists）作为后续小项。

---

## 7. P6 — 结构性 / 算法层（会改数学，属于新项目）

| # | 方向 | 为什么值得 | 代价 |
|---|---|---|---|
| **G01/G05/A32** | 让 2 个 q-tile 共享一份 KV 流 | 现在 64 FLOP/byte，H100 脊点是 295 —— **唯一能突破带宽天花板的杠杆** | 改选择粒度 → 改近似 → 重验 PSNR |
| **B19** | 128 行 query tile | FLOP/byte 翻倍 | O 累加器 128 regs，必须 2 CTA 或 2 WG |
| **B05** | ping-pong 双 consumer | FA3 的核心手段 | 需共享 K/V（同上） |

这三条同源：**只要选择是逐 query 的，ping-pong/多 q-tile 摊薄都用不上**。要做就得先决定是否接受近似变化。

---

## 8. 明确不建议

- **CLC**（sm100 才有）、**gather4**（sm100a）、**redux.sync**（f32 是 sm100a，且它是 32-lane）
- **真 warp specialization**（160 线程 → 2 CTA，正是已证伪的 MINB=2 区间）
- **TMA multicast**（Hopper 上常输给 redundant unicast，且 cluster 调度会吃掉 CTA 槽位）
- **fp8 P 检查点**（在 P3 之前不碰：会往 dS 注入 ~2⁻³ 误差）
- **把 router 的 512 B 块预聚成大块**（A01 已经用 16 B `cp.async` 修好，再预聚要多付 ~104 MB/次流量）

---

## 9. 验收标准（每一条都适用）

1. 每轮一个变量，同 build、A/A 对照、ABBA、warmup 10 / 30 iters / 3 batches。
2. 计时前必须过 CPU + generic + TK 三套件；新增配置要有自己的套件实例。
3. **保数学**的改动：输出/位图必须逐位相等（或 ≤1e-6 相对误差），否则按"改数学"处理。
4. **改数学**的改动：必须附 PSNR 表（float64 MSE），并区分算法误差与实现回归。
5. 失败的配置留在源码里当负对照，不删。
