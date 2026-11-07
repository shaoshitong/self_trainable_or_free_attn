# H100 (sm90a) CUDA / TMA / ThunderKittens 优化候选清单 — 约 100 条

面向的 kernel：VSA2 representative 64/4/1，bf16，D=128，64 行 query tile，每 CTA 34 个 KV tile，
单 warpgroup，166–168 寄存器，50 KB smem，3 CTA/SM，实测张量核利用率约 38%。
每个 query block 有**自己的** A/B 选择（`a_ptr/a_idx`、`b_ptr/b_idx`、`bits_ab` 逐行），这是所有设计的硬约束。

字段说明：
- **适用**：F 前向 / B 反向 / R router / O 全算子
- **保数学**：✅ 输出等价（至多浮点重结合）；❌ 改变近似或精度
- **状态**：已试（结果）/ 未试 / 不适用
- 预期收益都是估计值，不是测量值；已试项以本仓库实测为准。

---

## A. 数据搬运 / TMA（24 条）

| # | 手段 | 机制 | 适用 | 保数学 | 预期 | 风险 | 状态 |
|---|---|---|---|---|---|---|---|
| A01 | 512 B bulk → 128 线程 16 B `cp.async` | 换搬运引擎，字节与布局不变 | F | ✅ | −23% B tier | 低 | **已试，采纳** |
| A02 | tensor-TMA 直接 load B 记录 | 一次 box 取 4×64 | F | ✅ | 中 | 中 | 未试；预 swizzle 表会双重 swizzle，须改非 swizzle 布局 |
| A03 | 预打包 per-row B 表 | 让一个 tile 一次 TMA | F | ✅ | 负 | 低 | 已算过：104 MB/次流量，不划算 |
| A04 | 5D tensor map 多 box 复用 | 一个描述符覆盖 phase/block/panel | F | ✅ | 小 | 低 | 未试 |
| A05 | `cuTensorMapReplaceAddress` / `tensormap.replace` | 换基址不重建描述符 | F | ✅ | 小 | 低 | 未试 |
| A06 | TMA prefetch（L2 预热） | `cp.async.bulk.prefetch.tensor` | F | ✅ | 小-中 | 低 | 未试 |
| A07 | `tile::gather4` | 4 行任意索引一次取 | F | ✅ | — | — | **不适用**：sm_100a 起才有 |
| A08 | im2col TMA 模式 | 卷积式布局 | F | ✅ | — | — | 不适用（无重叠窗口） |
| A09 | interleave 模式 | 交错布局 | F | ✅ | — | — | 不适用 |
| A10 | OOB fill 语义 | 越界补零，省 mask | F | ✅ | 小 | 低 | 未试 |
| A11 | L2 promotion / cache hint | `evict_last` 给 K/V，`evict_first` 给 Q/dO | F/B | ✅ | 中-大 | 低 | 未试；我们已需 5.8 TB/s，L2 命中是隐形承重 |
| A12 | TMA multicast | 一个 load 广播给 cluster | F | ✅ | — | — | 未试；Hopper 上常输给 redundant unicast |
| A13 | cluster + DSMEM 共享 K/V | 对门 CTA 读对门 smem | F | ✅ | 中 | 高 | 未试；需选择交集，会砍掉可调度 CTA 数 |
| A14 | `cp.async` 的 .ca/.cg 变体 | 走/不走 L1 | F | ✅ | 小 | 低 | 未试 |
| A15 | 增大单次 bulk 到 ≥16 KB | 低于 16 KB 时 TMA 有 3.1× 税 | F | ✅ | 中 | 中 | 部分已试（A01） |
| A16 | 多 warp 分摊 gather 发射 | 4 warps 各发 1/4 | F | ✅ | 小 | 低 | 已隐含在 A01（128 线程） |
| A17 | `elect_one` 单线程发射 | 省掉整块 barrier | F | ✅ | 1-3% | 低 | 未试 |
| A18 | expect_tx 精确匹配 | 防挂死 | F | ✅ | — | — | 已遵守（正确性） |
| A19 | 双/三缓冲 stage | 隐藏 TMA 延迟 | F | ✅ | 0 | 低 | **已试 K 双缓冲：无收益** |
| A20 | 独立 P smem buffer + `wait_group 1` | 让 PV 与下一个 QK 重叠 | F | ✅ | 10-25% | 中 | 未试（本轮最佳之一） |
| A21 | epilogue 复用 store_async | O 用 TMA 写回 | F | ✅ | 小 | 低 | 未试 |
| A22 | `cp.reduce.async.bulk` 归约 | 一次指令替代 512 次原子 | B | ✅ | 中 | 中 | 部分已用（tier B dK/dV） |
| A23 | `ld.global.v4` + `stmatrix` 手写 gather | 绕开 TMA 引擎 | F | ✅ | 中 | 中 | A01 的更激进版本，暂不需要 |
| A24 | Q/K/V 布局（k-major vs MN-major） | 配合 SS/RS 与 swizzle | F/B | ✅ | 中 | 高 | 未试；影响 16 KB smem 与 34 次 fence |

| A25 | 描述符放进 kernel 参数常量区 | `const __grid_constant__ CUtensorMap`（64 B 对齐），TMA 单元直接读，不进 local | F/B/R | ✅ | 中 | 低 | 未试 |
| A26 | 分裂 K/V 屏障（各一个 mbarrier） | K 落地即开 QK，不等 V | F | ✅ | 中 | 中 | **已做**（`k_bar`/`v_bar` 分开） |
| A27 | 深度 vs 占用率四点扫描 | (3 CTA×2 stage)、(3 CTA×4 半箱)、(2 CTA×3)、(2 CTA×4) | F | ✅ | 中 | 中 | 部分（只测了 3 CTA×2） |
| A28 | `accessPolicyWindow` + `discard.global.L2` | 表/Q 常驻 L2，KV 用完即弃 | F/B | ✅ | 中 | 低 | 未试 |
| A29 | `clock64()` 暴露等待直方图 | 无 ncu 时用时间戳替代计数器 | 全部 | ✅ | 高（定位用） | 低 | 未试 |
| A30 | 多 lane 各自发射独立 TMA | K/V/四分块并行入队 | F/B | ✅ | 小-中 | 低 | 未试 |
| A31 | K/V 交错布局、一次 TMA 取两者 | 布局 (seq,2,128)，一个 rank-3 box | F | ✅ | 中 | 中 | 未试（改布局，影响其他 kernel） |
| A32 | 2 个 q-tile 共享一份 KV 流 | cluster 多播或 2 warpgroups | F/B | ✅ | 大 | 高 | 与 G01 同源：需改选择粒度 |

## B. 计算调度 / 占用率 / WGMMA（26 条）

| # | 手段 | 机制 | 适用 | 保数学 | 预期 | 风险 | 状态 |
|---|---|---|---|---|---|---|---|
| B01 | 认清 176 寄存器悬崖 | 128×168×3=64,512；176 就掉到 2 CTA | F/B | ✅ | — | — | 已验证：解释所有负结果 |
| B02 | 4 CTA（≤128 regs） | 多 33% 并发 | F | ✅ | — | — | 不可行：需 SS + 4×66 KB > 228 KB |
| B03 | 2 CTA（≤256 regs） | 允许 +88 regs 做 pipeline | F | ✅ | 负 | — | **已试 MINB=2：+5~23% 更慢** |
| B04 | 生产/消费 warp specialization | 5th warp 发 TMA，`setmaxnreg` | F/B | ✅ | 大 | 高 | 不适用（160 线程 → 2 CTA，正是 B03 的坑） |
| B05 | ping-pong 双 consumer WG | 两 WG 交替 MMA/softmax | F | ✅ | 大 | 高 | **不适用**：需共享 K/V，逐 query 选择不允许 |
| B06 | 零寄存器专业化（elect_one） | 不加 warp 的"伪 WS" | F | ✅ | 1-3% | 低 | 未试（同 A17） |
| B07 | wgmma commit-group 拆分 | QK/PV 分组，`wait_group 1` | F/B | ✅ | 中 | 低 | 未试 |
| B08 | **n32 半块 softmax + 双 S 缓冲** | S 16 regs，两块 48 = 现在 32+16，TC 周期不变 | F | ✅ | **最大未试** | 中 | 未试 |
| B09 | n128 KV 块 | 每 128 key 一次 rescale | F | ✅ | +5-10% | 中 | 未试；+32 regs 会掉 CTA |
| B10 | RS wgmma（A 在寄存器） | 省 smem A 读与 fence | F | ✅ | 中 | 中 | 未试；仅 n≤32 有优势 |
| B11 | `mma.sync` 处理瘦尾块 | 避免 wgmma 的 64 行粒度浪费 | F/R | ✅ | 5-10% | 低 | 未试 |
| B12 | 循环内去掉 `__syncthreads` | mbarrier parity + named barrier | F | ✅ | 中 | 高 | 部分（cp.async 路径有 barrier） |
| B13 | 持久化 kernel + tile scheduler | 摊薄 prologue、削尾 | F/B | ✅ | 3-8% | 中 | 未试 |
| B14 | **选择感知的 CTA swizzle / L2 驻留** | 让同 SM 的 CTA 选择重叠 | F/B | ✅ | 大 | 中 | 未试 |
| B15 | stream-K / split-K（前向） | 拆 KV 区间再合并 | F | ✅ | 中 | 中 | 未试；不平衡 <20% 时通常输 |
| B16 | 2-SM wgmma（`cta_group::2`） | 128 行累加器摊到两 CTA | F | ✅ | — | — | 不适用（需共享 B） |
| B17 | PDL / `griddepcontrol` | router 尾与 attention 头重叠 | O | ✅ | 3-6% | 中 | 未试 |
| B18 | CUDA graph 整链 | 去 launch 开销 | O | ✅ | 3-8% | 低 | 未试 |
| B19 | 128 行 query tile | FLOP/byte 翻倍 | F | ✅ | 大 | 高 | 未试；需 2 CTA 或 B05 |
| B20 | 融合 epilogue | 最后一次 rescale 并进 PV 等待 | F | ✅ | 2-3% | 低 | 未试 |
| B21 | 尾量化均衡 | 34 tile 的 wave 余数 | F | ✅ | 小 | 低 | 未试（B13 的一部分） |
| B22 | 编译期特化 A/B/C 三路 | 去分支与无用寄存器 | F | ✅ | 小-中 | 低 | 部分（模板已分 A_ONLY） |
| B23 | bias/mask 预进寄存器或常量 | 省每 tile 加载 | F | ✅ | 小 | 低 | 已做（lcb 扁平化） |
| B24 | 去每 tile 的 descriptor 重建 | 描述符常驻 | F | ✅ | 小 | 低 | 已做 |
| B25 | 3D tensor map 一次多 stage | 一次 TMA 取多 tile | F | ✅ | 小 | 中 | 未试 |
| B26 | CLC（cluster launch control） | 硬件派 tile | F | ✅ | — | — | 不适用：sm100 |

## C. Softmax / 数值 / 寄存器（20 条）

| # | 手段 | 机制 | 适用 | 保数学 | 预期 | 风险 | 状态 |
|---|---|---|---|---|---|---|---|
| C01 | exp2 折叠 | `log2e*scale` 并进 FFMA | F | ✅ | 2-3% | 低 | 未试 |
| C02 | 只用 `ex2.approx` | 别走 `expf` 的范围归约 | F | ✅ | 中 | 低 | 部分（log2 域已是） |
| C03 | 不用 `__expf` 做底变换 | 误差随 \|x\| 增长 | F | ❌ | — | — | 已避免 |
| C04 | **fp16 P + `ex2.approx.f16x2`** | 一条 MUFU 出两个 exp，且免 cvt | F | ❌（P 精度变） | 大 | 中 | 未试 |
| C05 | ALU 多项式 exp2 | 把 MUFU 压力挪到 ALU | F | ❌ | 负 | 中 | 不推荐（本 kernel ALU 也忙） |
| C06 | bias 常驻寄存器 | 每线程固定 16 列 | F | ✅ | 小 | 低 | 已做（TileBias 预取） |
| C07 | 有限哨兵替代 -inf | 避免 `-inf - -inf` = NaN | F | ✅ | 小 | 低 | 部分（已有 ninf 守卫） |
| C08 | 清零 P 而非污染 logit | mask 不进 row max | F | ✅ | 小-中 | 中 | 未试 |
| C09 | tile 级 mask 分类 | skip / 无 mask / 全 mask 三路 | F | ✅ | 中 | 低 | 部分（cmask 已跳空 tile） |
| C10 | quad reduce（2 次 shfl） | 行归约只需 4 lane | F | ✅ | 中 | 低 | 未试 |
| C11 | `redux.sync` | 单指令归约 | F | ✅ | — | — | 陷阱：f32 redux 是 sm100a；且是 32 lane |
| C12 | 归约保持 warp 内 | 不落 smem | F | ✅ | 中 | 低 | 已做（TK col_vec） |
| C13 | 条件 rescale（`__any_sync`） | max 没变就不乘 O | F | ✅ | 中 | 低 | **已试 deferred rescale：无变化**（实现方式不同，可重试） |
| C14 | max lag（每 T tile 刷新） | 摊薄 rescale | F | ✅ | 中 | 中 | 未试 |
| C15 | max elision（范围证明） | 彻底去掉 running max | F | ❌ | 大 | 高 | 未试；需 |q·k| 上界 |
| C16 | 行和从 fp32 P 累加 | 避免 bf16 求和 | F | ✅ | 小 | 低 | 已做 |
| C17 | `cvt.rn.bf16x2` | 打包转换、就近舍入 | F | ✅ | 小 | 低 | 未试（可能已是） |
| C18 | P 不进 smem（`movmatrix`） | 省 stmatrix + barrier | F | ✅ | 中 | 中 | 未试 |
| C19 | O 留 fp32 寄存器 | vs fp16 / smem | F | ✅ | — | — | 已是 fp32 寄存器 |
| C20 | `rcp` 代替 div、1/sqrt(D) 常量化 | epilogue 归一化 | F | ✅ | 小 | 低 | 部分 |

## D. 反向传播（26 条）

| # | 手段 | 机制 | 适用 | 保数学 | 预期 | 风险 | 状态 |
|---|---|---|---|---|---|---|---|
| D01 | 砍 smem 113→≤76 KB + setmaxnreg | 唯一能真拿到 3 CTA 的路 | B | ✅ | **tier A 1.55→~1.2 ms** | 高 | 未试 |
| D02 | 融合 mean-pool 伴随 | `dK_j += (1/c)dS` 进 epilogue | B | ✅ | 中 | 中 | 未试 |
| D03 | 融合 unpool | 与 D02 同 kernel | B | ✅ | 中 | 中 | 未试 |
| D04 | 去 zero-init | first-writer 谓词 + `red.add` | B | ✅ | 中 | 低 | 未试 |
| D05 | 融合 delta | 与首个消费者同 kernel | B | ✅ | 中 | 低 | 未试 |
| D06 | KV-owner / query-owner 混合 | 按 tier 选主 | B | ✅ | 中 | 中 | 已是混合（A KV、B/C query） |
| D07 | split-K / stream-K（反向） | 拆区间再合并 | B | ✅ | 小 | 中 | 部分（tier C split） |
| D08 | 确定性 fp32 partials | 可复现 | B | ✅ | 负 | 低 | 未用（现用原子） |
| D09 | `cp.reduce.async.bulk` | 批量归约 | B | ✅ | 小-中 | 中 | 已用（tier B） |
| D10 | `red.global.add.v4` 替代 atomicAdd | 向量化原子 | B | ✅ | 小-中 | 低 | 未试 |
| D11 | P 重算 vs 存储 | 省带宽/寄存器 | B | ✅ | 中 | 中 | 已是重算 |
| D12 | 两 kernel 无原子分解 | 彻底去原子 | B | ✅ | 负 | 中 | 不推荐（重复读 Q/K/V） |
| D13 | fp8 P 检查点 | 省带宽 | B | ❌ | 大 | 高 | 未试；会注入 ~2^-3 误差 |
| D14 | 反向 PDL / graph | 去 launch 开销 | B | ✅ | 小 | 低 | 未试；FA3 对 bwd 全部关闭 PDL |
| D15 | 专用 dQ writer warpgroup | 隔离原子争用 | B | ✅ | 中 | 高 | 未试 |
| D16 | 反向 tile 调度 / L2 | 同 B14 | B | ✅ | 中 | 中 | 未试 |
| D17 | dK/dV 按块局部归约 | 降原子冲突 | B | ✅ | 小 | 低 | 已是（fp32 表） |
| D18 | 反向 warp specialization | 生产/消费 | B | ✅ | — | — | 受同一 K/V 不共享限制 |
| D19 | 反向 epilogue 向量化 | 合并 store | B | ✅ | 小 | 低 | 未试 |
| D20 | 流的顺序（fwd/bwd 同/异流） | 重叠 | B | ✅ | 小 | 中 | 未试 |
| D21 | delta 与 O 同精度 | 避免系统性漂移 | B | ✅ | — | — | 已注意 |
| D22 | split 策略 | tier C 的切分 | B | ✅ | 0 | 低 | **已试：16K 下三种策略等价** |
| D23 | tier-A launch_bounds 探针 | 试 3 CTA | B | ✅ | 负 | — | **已试：1.35× 更慢** |
| D24 | 245/250 寄存器微调 | 同占空比点 | B | ✅ | 0 | — | 已论证无意义 |
| D25 | 合并 tier-B/C kernel | 少一次启动与表读 | B | ✅ | 小 | 中 | 未试 |
| D26 | 反向 kernel 重排（C 先于 A） | 重叠尾部 | B | ✅ | 小 | 中 | 未试 |

## E. Router / 全算子（10 条）

| # | 手段 | 机制 | 适用 | 保数学 | 预期 | 风险 | 状态 |
|---|---|---|---|---|---|---|---|
| E01 | 单直方图双阈值 | 一次统计出两个切点 | R | ✅ | ~0.1 ms（full fwd 8%） | 中 | 未试（需位图逐位证明） |
| E02 | 从 A\|B CSR 派 A 位图 | 少一次 top-k | R | ✅ | 中 | 中 | 未试（依赖 rank 顺序） |
| E03 | 融合 pool + score | 少一次读写 | R | ✅ | 中 | 中 | 未试 |
| E04 | 融合 tier_b_lists 进 topk | 少一个 kernel | R | ✅ | 小 | 中 | 未试 |
| E05 | router 小 GEMM 用 `mma.sync` | 避免 wgmma 64 行粒度 | R | ✅ | 5-10% | 低 | 未试 |
| E06 | router 融进 attention prologue | 去 launch 与中间矩阵 | R+F | ✅ | 中 | 高 | 未试 |
| E07 | PDL 连接 router→attn | 重叠 | O | ✅ | 小 | 中 | 未试 |
| E08 | CUDA graph 整链 | 去 launch | O | ✅ | 小-中 | 低 | 未试 |
| E09 | 减少 score 矩阵物化 | 64×2176 fp32/block | R | ✅ | 中 | 高 | 未试 |
| E10 | 位图运算优化 | popcount/掩码合并 | R | ✅ | 小 | 低 | 未试 |

## F. 测量与工程（7 条）

| # | 手段 | 说明 | 状态 |
|---|---|---|---|
| F01 | SASS 检查（`cuobjdump -sass`） | 确认 MUFU.EX2、无 CAL、无 spill | 部分（ptxas 有） |
| F02 | `clock64()` 分区计时 | 无 ncu 时定位区间 | 未试 |
| F03 | nsys / CUPTI | 时间线，不依赖计数器 | 未试（权限未知） |
| F04 | CFG 变体受控消融 | 本仓库主方法 | 已用 |
| F05 | A/A 地板与漂移 | 每轮必带 | 已用（最高见 1.4%） |
| F06 | 固定时钟 / 多 seed | 降噪 | 未用（调度器不可控） |
| F07 | 多形状稳健性（D=64/RM=2/8/partial） | 防过拟合单点 | 部分（套件覆盖） |

## G. 算法层（会改数学，单独标注）（5 条）

| # | 手段 | 机制 | 保数学 | 预期 | 风险 |
|---|---|---|---|---|---|
| G01 | 按 KV 块聚合 query | 让多个 query tile 共享 K/V 选择 | ❌ | 大 | 高 |
| G02 | 全局预算自由分配（VSA2 原意） | 所有 Q 共 10%，非每 Q 10% | ❌ | 大 | 中 |
| G03 | 层次化选择（粗到细） | 先块级后 token | ❌ | 大 | 中 |
| G04 | 低精度重算 P | 反向省带宽 | ❌ | 中 | 高 |
| G05 | 共享 superset 选择 | 为 ping-pong 铺路 | ❌ | 大 | 高 |

---

## 已证伪（不要重复）

1. 向量化 softmax（R2，1.0018）
2. MINB=2 提寄存器上限（R3，f+b +5.0%，fwd +22.8%）
3. MINB=4（早期，0.723 vs 0.521）
4. K 单边双缓冲（R5B，1.0003）
5. tier-C split 策略（R6，16K 下等价）
6. tier-A `launch_bounds(128,3)`（R6，1.350）
7. 两段式 softmax（R7，kernel 1.292）

## 跨目录的三条硬结论

1. **176 寄存器悬崖**：128×168×3 = 64,512 / 65,536，再多 8 个寄存器就掉到 2 CTA。所有"加 32 寄存器做 pipeline"的方案都会死在这里——这解释了 R3 和 R7。
2. **设计空间只有 {3 CTA, 2 CTA}**，而 2 CTA 已被两次实测否定。所以任何优化必须**零新增寄存器**才可能赢。
3. **L2 是隐形承重**：38% 利用率下 KV 需求约 5.8 TB/s，高于 HBM 3.35 TB/s，说明已经靠 L2 命中在撑。选择感知的调度（B14）是唯一"不碰 kernel 就能提速"的杠杆。

## 补充（TMA 目录回收后新增）

- **A27 是最便宜的信息增量**：现在只测过「3 CTA × 2 stage」，而 76 KB/CTA 在数学上刚好卡住 ~2 个 KV stage。
  四点扫描 (3×2)、(3×4 半箱)、(2×3)、(2×4) 比任何 copy-engine 重写都更可能解释"为什么流水线一直没藏住 HBM 延迟"。
- **A29 `clock64()` 直方图**：ncu 没权限，这是唯一能区分"等待受限 / 发射受限 / 带宽受限"的手段。
- **A28 L2 驻留 + discard**：只加 hint，无正确性风险；12 warps/SM 藏不住 HBM 延迟，pin + prefetch（A06）是最便宜的替代。
- **A26 已做**（K/V 各自 barrier），不用再动。
- **"把 router 的 16×512 B 预聚成 ≥2 KB 一次搬"这条被我们的实测覆盖了**：A01 已经用 128 线程 16 B `cp.async`
  把这一层修好（B tier −23%），再预聚要多付 ~104 MB/次的流量，不划算。

## 若只做三件

1. **B08**（n32 半块 + 双 S 缓冲）：唯一能在保住 3 CTA 的前提下把 softmax 藏到 MMA 后面。
2. **C13+C14+C01**（条件 rescale + max lag + exp2 折叠）：零寄存器，直接砍串行区里的 ALU。
3. **B14**（选择感知 CTA swizzle）：不动 kernel、不加寄存器，冲着 L2 命中率去。

反向则优先 **D02+D03+D04+D05**（那 0.6 ms 的记账开销），而不是继续碰 tier-A 的占用率。
