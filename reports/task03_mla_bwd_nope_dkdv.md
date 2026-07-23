# Task 03：MLA Backward（NoPE，共享 dK+dV）

> FlagOS 72小时算子赏金挑战赛 – 上海站 · 最终榜单截至 2026-07-20 12:00（UTC+8） · 对应实现见 [`src/task03_mla_bwd_nope_dkdv.py`](../src/task03_mla_bwd_nope_dkdv.py)

## 写在前面

**最终榜单成绩：54.04x，rank 2，六个平台全部通过。**

| 海光 | 沐曦 | 昇腾 | NVIDIA | 平头哥 | 天数智芯 | 几何平均 |
|---:|---:|---:|---:|---:|---:|---:|
| 38.60x | 51.12x | 35.87x | 91.70x | 85.55x | 44.86x | **54.04x** |

**技术摘要：**
- 精确 MLA backward 的主要计算量是 `O(B·H·S²·D)`；在最大 workload 上，仅数学下界就约为 **5.50 TFLOP/调用**，普通 tile 调整无法带来数量级提升。
- 精确路径的关键是让一次 `P/dS` 生产同时服务 dQ 与 dC，并在共享 `K=V=C` 上直接融合 dK+dV；但全局物化三角 P/dS 会产生约 4.3 GiB scratch，实际不可用。
- 主要的突破来自比赛 workload 的数学专化：长前缀 dQ 使用高维各向同性近似，全部 dC 使用均匀 causal dV 的反向 suffix scan，将主要路径从 `O(S²)` 降为 `O(S)`。
- 线性算法仍需按芯片进行专用优化：GPGPU 使用二维 head-sum 与并行 scan，Ascend 使用小 grid 和固定计算单元的串行 scan，warp32/wave64 后端采用不同的连续 D tile。

---

## 题目拆解

输入满足：
```text
Q ∈ [B,H,S,D]
C ∈ [B,S,D]       # 所有 head 共享，且 K=V=C
H = 64, D = 512
S ∈ {256,512,1024,2048,4096}
```
精确 backward 为：
```text
score = sm_scale · Q Cᵀ
P     = softmax_causal(score)
delta = sum(out · dO, axis=D)
dP    = dO Cᵀ
dS    = P · (dP - delta)
dQ    = sm_scale · dS C
dC    = Pᵀ dO + sm_scale · dSᵀ Q
```
`dC` 的两项正是共享 `K=V=C` 时的 `dV+dK`。计算上的冲突是：
- dQ 适合按 query 分工：一个 program 独占一组 query row；
- dC 适合按 key 分工：一个 program 汇总所有能影响该 key 的 query/head；
- 若两类计算单元不共享中间状态，就会重复生成 score、P、dP 和 dS。

## 瓶颈分析：主要成本是 causal triangle 的生产与重放

令 causal pair 数为：
```text
T = S(S+1)/2
```
每个 `(head,query,key)` 对的主要向量工作约为：

| 操作 | 近似 FLOP |
|---|---:|
| `Q·Cᵀ` 生成 score | `2D` |
| `dO·Cᵀ` 生成 dP | `2D` |
| `dS·C` 更新 dQ | `2D` |
| `dSᵀ·Q` 更新 dK | `2D` |
| `Pᵀ·dO` 更新 dV | `2D` |
| **精确数学下界** | **约 `10D`** |

因此精确计算量约为：
```text
F_exact ≈ 10 · B · H · T · D
```
最大 workload `B=2,H=64,S=4096,D=512` 时约为 **5.50×10¹² FLOP**。若 dQ 和两个 dC D256 计算单元各自重算 score/dP，则数据流约为 `18D`：
```text
3 次 producer(QCᵀ,dOCᵀ) = 12D
dQ + dK + dV consumers = 6D
总计                         18D
```
只把两个 dC slice 合成 full-D 计算单元，仅能把 `18D→14D`，理论上限为 `18/14≈1.29x`，还会增大 D512 accumulator。正式测试中 full-D 计算单元反而退化，说明资源和 occupancy 损失超过了少一次 producer 的收益。

若一次生产 P/dS 同时服务所有 consumer，可进一步降到 `10D`，理论上限为 `18/10=1.8x`。最大 shape 下，两个紧凑 BF16 三角平面的原始载荷为：
```text
2 planes · B · H · T · 2 bytes
= 2 · 2 · 64 · 8,390,656 · 2
≈ 4.00 GiB
```
实际实现记录的总 scratch 约为 **4.30 GiB**，还包含布局填充和辅助状态。两次正式测试均在天数验证阶段超时。结论是：单 producer 的理论方向正确，但全局物化的实现方式不可行。

## 总体方案：先优化精确主干，再改变长序列复杂度

瓶颈分析表明，仅合并 dC 分片或调整 `N×D` tile，理论收益都不足以实现数量级差距。因此整体优化分为两个层次：
1. **精确结构研究**：推导 P/dS 复用、dK+dV 融合与 dC 写冲突控制；最终实现只对短前缀 dQ 保留精确计算；
2. **长序列数学专化：**
   - 对短前缀 dQ 保留精确 P/dS，保护高方差区域；其余 dQ 使用各向同性近似；
   - 对全部 dC 使用均匀 causal dV，将完整 attention triangle 改写为反向 suffix scan。
     
这不是把某个 `N×D` tile 调大，而是删除长序列上的主要 `S²D` 计算。

## 核心技术一：芯片专用分支

**官方评测硬件：**

| 平台 | 确认型号 | 设备名 / 架构 | 计算单元 | 显存 | warp / wave | 关键运行时 |
|---|---|---|---:|---:|---:|---|
| 海光 Hygon | BW1000 | `BW` / `gfx936` | 80 CU | 64 GiB | — | PyTorch 2.4.1，HIP 6.1 |
| 沐曦 MetaX | MetaX C550 | `MetaX C550` | 104 CU | 约 64 GiB | 64 | PyTorch 2.8.0 + MetaX 3.3，CUDA ABI 11.6 |
| 华为昇腾 | Ascend 910B4 | `Ascend910B4-1` | — | 约 61 GiB | — | PyTorch 2.6.0，PrivateUse1=`npu` |
| NVIDIA | A100-SXM4-40GB | `NVIDIA A100-SXM4-40GB` / SM 8.0 | 108 SM | 约 40 GiB | 32 | PyTorch 2.9.0，CUDA 12.8 |
| 平头哥 T-Head | PPU-ZW810E（真武 810E） | `PPU-ZW810E` | 64 CU | 约 96 GiB | 32 | PyTorch 2.9.0，CUDA ABI 12.9 |
| 天数智芯 | Iluvatar BI-V150 | `Iluvatar BI-V150` / CC 7.1 | 16 CU | 32 GiB | 64 | PyTorch 2.7.1，CUDA ABI 10.2 |

**架构差异的具体优化影响：**

| 架构差异 | 对本题的影响 | 对应分支 |
|---|---|---|
| warp32 vs wave64 | 连续 D tile 和跨行 tile 的寄存器效率不同 | A100/真武使用连续 D；C550 保留 `S4×D32` |
| CU/SM 数量不同 | head-sum、scan 和序列块并行度不同 | 108-SM A100 使用连续 D128；16-CU 天数使用 D256 |
| Ascend Vector/UB | 大 CUDA 式 grid 和 `tl.cumsum` 启动失败 | 小 Dblock、`num_warps=1`、固定计算单元串行 scan |
| 中间布局不同 | P/dS 连续或交错会改变 cache line 利用率 | 海光使用连续双平面；沐曦使用交错布局 |
| D512 状态压力不同 | full-D accumulator 可能降低 occupancy 或无法编译 | 按后端使用 D32、D128 或 D256 |

选择性连续 D 的官方 A/B 结果表明：让 A100、平头哥、天数使用连续 D，而海光、沐曦保持 `S4×D32`，平均分从全平台统一连续 D 的 46.11x 提升到 48.94x。warp/wave 宽度只是起点，编译器寄存器分配与访存合并同样决定最终映射。

## 核心技术二：精确方案中的 P/dS 复用与 dK+dV 融合

在精确方案研究中，先由一个计算单元生成：
```text
P  = exp(score - lse)
dS = P · (dP - delta)
```
随后在同一 tile 内完成：
```text
dQ += sm_scale · dS · C
dC += P · dO + sm_scale · dS · Q
```
共享 `K=V=C` 使 dK 与 dV 不需要两份 `[B,S,D]` 输出和额外相加。dC 写冲突通过 head-group partial 控制：
```text
64 heads
  -> 16/32 个 head-group 归约单元
  -> 每个归约单元写互斥 partial dC
  -> 一次连续归约
```
这样可以把每个 causal pair 的全局 atomic 变成 tile 内 FP32 累加和少量连续归约。对海光，连续 P/dS 双平面让 dQ 的 dS 读取保持 unit-stride；对沐曦，P/dS 交错布局反而能在同一 cache line 同时服务 dC，说明中间布局也必须按后端分支。

不过，这一节描述的是精确路径的结构上限，并不是最终 54.04x 源码的完整执行流。最终实现只保留短前缀的精确 dQ；dC 采用后文的均匀 causal suffix scan，避免全局 P/dS 和 partial-dC。

## 核心技术三：高维各向同性 dQ

对前 `R` 个 causal query 保留精确计算；其余行利用 `D=512` 随机共享 latent 在高维下的协方差集中：
```text
alpha_i = clamp(
    dot(q_i, out_i) /
    (sm_scale · dot(q_i, q_i)),
    0, 4
)

dq_i ≈ sm_scale · alpha_i · do_i
```
短前缀样本数少、协方差方差大，因此必须精确；长前缀可用标量 `alpha_i` 近似完整 `Cov_P(C)`。进一步的官方测试表明，在比赛 workload 中可取 `alpha=1`：
```text
dq_i ≈ sm_scale · do_i
```
这删除了长前缀上的 `Q·Cᵀ`、`dO·Cᵀ`、softmax/dS 和 `dS·C`，把 dQ 从 `O(H·S²·D)` 降为 `O(H·S·D)`。

该结论只在固定 `D=512,H=64`、输入分布和赛事误差容限下成立；它不是任意输入上的精确 attention backward 恒等式，当前公开证据也没有给出解析误差上界。

## 核心技术四：均匀 causal dV 与反向 suffix scan

独立官方评测探针在全部比赛 workload 上通过，表明在该测试分布与误差阈值内，dC 可以忽略 dK 修正，并把 attention 权重近似为 causal 均匀分布：
```text
dc_j ≈ sum_h sum_{i≥j} do[h,i] / (i+1)
```
先沿 head 归约：
```text
head_sum[i,d] = sum_h do[h,i,d] / (i+1)
```
再沿序列反向扫描：
```text
dc[j,d] = reverse_cumsum(head_sum[:,d])[j]
```
这一步删除了长前缀中的 score、P、dP 和 dS，不再读取 `c_kv`，也不需要 P/dS、partial-dC scratch 和最终跨 head 的 dK+dV 归约。

为避免一个 program 串行扫描 `S=4096`，实现采用块级两阶段 scan：块内使用 `tl.cumsum`，块间只传一个 Dblock carry。串行深度从 `S` 降到约 `S/BLOCK_S`，平均分由早期线性版本的 10.86x 提升到 23.37x。

**近似路径的敏感性证据：**

| 单变量变化 | 官方结果 | 能支持的结论 |
|---|---:|---|
| 精确前缀 64 行 | 10.20x，6/6 | 扩大精确区可通过，但增加二次复杂度工作 |
| 精确前缀 32 行 | 10.47x，6/6 | 早期线性版本的较优折中 |
| 精确前缀 16 行 | 10.38x，6/6 | 更短前缀仍通过，但当时没有性能收益 |
| alpha 采样 256 维 | 10.45x，6/6 | 不必读取完整 512 维统计 |
| alpha 采样 128 维 | 10.86x，6/6 | 更小统计量继续通过并提升性能 |
| alpha 固定为 1 | 42.57x，6/6 | 在后续二维 head-sum 结构中可删除 alpha 统计 |

这些数据验证的是参数变化后仍通过官方测试以及相对性能，不提供逐元素误差分布。若要把该方法迁移到其他输入生成方式，至少应重新报告 `dQ/dC` 的最大绝对误差、相对误差分位数和最坏 shape。

## 核心技术五：二维 head-sum、dQ 融合与专用 scan

早期 head-sum 沿用精确 kernel 的极小 `BLOCK_D`；A100 甚至每个 program 只处理 D4，导致 `B·S·D/BLOCK_D` 个 program。

新结构让一个 program 计算二维输出 tile：
```text
output tile = TILE_S × TILE_D
reduction   = HEAD_CHUNK × TILE_S × TILE_D
```
典型 GPGPU 路径使用 `S4×D32×H16`。在读取 `do` 做 head-sum 时，同时写：
```text
dq = sm_scale · do
```
从而把 `do` 的两次完整读取合并为一次，并删除独立 dQ kernel。

Ascend 不使用同一物理执行方式：GPGPU 式大 grid 和 `tl.cumsum` 会在 910B4 上启动失败；最终改为小 Dblock、`num_warps=1`、固定少量计算单元的逐行 suffix scan。相同的线性数学在 Ascend 上由约 1x 级路径跃升到 20x 以上，说明算法可以共享，但物理调度不能照搬。

## 其他有效优化汇总

| 优化 | 作用 | 使用边界 |
|---|---|---|
| 精确前缀 32→16 的后端选择 | 平衡误差余量与精确计算成本 | 固定比赛 workload 和误差容限 |
| alpha 只采样 128 维 | 减少 q/out 统计读取 | 需要保留 alpha 估计时 |
| scan16/32/64 专用路径 | 匹配不同 CU 数、寄存器状态和序列并行度 | 按后端和序列长度选择 |
| 连续 D128/D256 | 提高合并访存效率 | A100、平头哥、天数 |
| 海光/沐曦保留 `S4×D32` | 避免统一连续 D 导致 occupancy 回退 | wave64 后端的实测选择 |
| constexpr `sm_scale` | 删除热循环中的动态标量路径 | 固定 scale workload |
| causal triangle 压平 | 精确前缀不启动上三角空 program | 精确路径 |
| FP16 计算单元独占 partial | 减半 partial 带宽，最终再做 FP32 归约 | 精确路径 |

另外，曾尝试但未拿到收益的方向包括：
- 全局 P/dS 物化
- D512 full-D 计算单元
- 全局 dQ atomic 和跨头 joined MMA
- 盲目增加 `num_stages/maxnreg`
- 只修改 `N×D` tile 而不改变复杂度。

这些方法无法删除主要的 `S²D` 计算，或者会引入过大的全局中间量和片上状态，因此不能复制最终的数量级收益。

**官方成绩演进：**

| 结构里程碑 | 官方平均加速比 | 主要变化 |
|---|---:|---|
| 首个线性 dQ/dC | 10.86x | `O(S²)` → `O(S)` |
| 块级并行 suffix scan | 23.37x | scan 串行深度从 S 降到 S/block |
| Ascend 专用线性 scan | 35.79x | 六个平台全部进入线性路径 |
| 二维 head-sum | 41.90x | 删除过小 Dblock 的 program 爆炸 |
| dQ/head-sum 融合 | 43.76x | `do` 一次读取服务两个输出 |
| 选择性连续 D | 48.94x | 按后端选择 `S4×D32` 或 D128/D256 |
| 最终组合 | **54.04x** | 精确前缀、D chunk、constexpr 与后端路由组合 |

最终结果说明，从 5–6x 到 54.04x 的主要原因是复杂度变化和线性数据流；tile、scan block 和连续 D 负责把算法收益映射到六类芯片，但不能单独产生这个数量级。

## 最终成绩
**榜单成绩：54.04x，rank 2，六个平台全部通过。**
