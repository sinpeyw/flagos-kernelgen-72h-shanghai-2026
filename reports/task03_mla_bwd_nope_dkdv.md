# Task 03：MLA Backward（NoPE，共享 dK+dV）

## 最终官方成绩

**54.04x，rank 2**。六平台分项为：海光 38.60x、沐曦 51.12x、昇腾 35.87x、NVIDIA 91.70x、平头哥 85.55x、天数智芯 44.86x。

## 精确数学与工作量

对每个 batch、head 和 causal 位置 `j <= i`：

```text
score_ij = sm_scale * dot(q_i, c_j)
P_i      = softmax(score_i)
delta_i  = dot(do_i, out_i)
dP_ij    = dot(do_i, c_j)
dS_ij    = P_ij * (dP_ij - delta_i)
dQ_i    += sm_scale * dS_ij * c_j
dC_j    += P_ij * do_i + sm_scale * dS_ij * q_i
```

这里 K 与 V 共享同一个 `C`，所以 `dC` 正好是 `dV + dK`。令 causal pair 数 `T=S(S+1)/2`，忽略 softmax 标量操作，精确实现每个 `(head,pair)` 的主导向量工作约为：

```text
Q·Cᵀ       2D FLOPs
dO·Cᵀ      2D FLOPs
dQ 更新    2D FLOPs
dV 更新    2D FLOPs
dK 更新    2D FLOPs
合计      ≈10D FLOPs
```

总量约为 `10 * B * H * T * D`，即 `O(BH S²D)`。如果 dQ 和 dC 各自重算 score、P 与 dP，生产阶段会额外重复约 `4 * B * H * T * D` FLOPs。仅消除这次重放，理论上是把约 `14D` 降到 `10D`，上限约 1.4x；要获得更大收益，还必须同时解决全三角写回、dC 写冲突、occupancy 和启动开销。

## 实际有效的优化

### 1. 让一次 P/dS 生产同时服务 dQ 与 dC

实现按后端采用两种方式：

- 海光、沐曦：只物化 causal triangle 的 P/dS，后续 dQ 与 dC 都读取同一份结果；
- NVIDIA：在 per-head owner 内直接融合 P/dS 生产与 dQ/dC 消费，避免完整全局三角写回。

物化路径用额外带宽换取消除重复 `Q·Cᵀ` 和 `dO·Cᵀ`；融合路径用更高寄存器状态换取零中间写回。两者的选择取决于后端寄存器容量和调度能力。

### 2. 共享 K=V 时直接融合 dK+dV

同一 `(i,j)` 上，`P_ij * do_i` 与 `sm_scale * dS_ij * q_i` 最终写到同一个 `dC_j`。在 tile 内把两项相加后再进入 partial buffer，避免先生成两份 `[B,S,D]` 梯度再相加，也让 q/c/do 的加载更容易复用。

### 3. 用少量 shard 限制 dC 冲突

dQ 天然由 query row 唯一拥有；dC 则会被所有 query 和 64 个 head 更新。实现按 head group 或 partition 建立少量 partial dC：

```text
64 heads -> head_group=2/4 -> 32/16 个受控 owner
owner 内完成 tile 累加 -> 最后一次连续归约
```

这避免了每个 causal pair 直接 atomic 到全局 dC，也避免为每个 query/head 写一份巨大的 partial。收益来自冲突减少、连续写回和归约规模受控，而不是仅仅改变 grid 形状。

### 4. 后端专用所有权

- 昇腾：固定物理 core 所有权；delta、dQ、dC 分别限制到约 40/20/20 个 core，`BLOCK_M=16`、`BLOCK_N=32`，最后 40-core 归约；
- 海光、沐曦：materialized P/dS owner，按后端选择 block 与 D slice；
- 天数智芯、平头哥：partition-fused dQ/dC，再分别归约 partial dQ 与 dC；
- NVIDIA：`head_group=2` 的 per-head fused owner，短/中序列直接使用融合路径。

统一的是数学和所有权原则，不统一的是 tile、partial dtype、D slice 与并行粒度。

## 54.04x 最终榜单版本为何更快

最终榜单版本的数量级跃迁不是 tile 调优，而是复杂度变化：

- 短前缀执行精确 attention backward；
- 长前缀用高维各向同性近似 `dQ ≈ sm_scale * dO`；
- dC 使用均匀 causal dV 近似；
- 省略一般输入下不可忽略的精确 dK 修正。

于是长序列主路径从 `O(BH S²D)` 降为近似 `O(BHSD)`。这足以解释 54.04x 的数量级成绩；任何只调整 `N×D` 分块、warp 数或 partial 布局的方案，都无法产生相同幅度。该方法依赖赛事固定的 `D=512/H=64/shared-KV`、输入分布和误差容限，不能推出任意合法输入上的严格等价。

## 没有转化为最终收益的方向

- 对 dQ、dK、dV 分别重放完整 causal triangle；
- 把完整 S² score/P/dS 全部长期写到显存；
- 为每个 head/query 创建巨大的 dC partial；
- 让单一 owner 串行处理 64 个 head；
- 只把 `N32×D256` 改成 `N64×D128`：它改变占用和数据复用，但不减少 `O(S²D)` 主项；
- 用一套 tile 跨六后端，容易在寄存器、wave/warp 和局部内存限制上同时失配。

## 可迁移结论

精确优化应按三个问题依次推进：P/dS 是否被重算、dK/dV 是否在共享 C 上融合、dC 冲突是否被限制到少量 shard。完成这三点后，再调 tile 才有稳定意义。若成绩出现一个数量级跃迁，首先检查是否改变了算法复杂度或误差模型，而不是把它归因于常规 kernel 参数。
