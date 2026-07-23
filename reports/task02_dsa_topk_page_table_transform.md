# Task 02：DSA TopK Page Table Transform

> FlagOS 72小时算子赏金挑战赛 – 上海站 · 最终榜单截至 2026-07-20 12:00（UTC+8） · 对应实现见 [`src/task02_dsa_topk_page_table_transform.py`](../src/task02_dsa_topk_page_table_transform.py)

## 写在前面

**最终榜单成绩：4.24x，rank 5，六个平台全部通过。**

| 海光 | 沐曦 | 昇腾 | NVIDIA | 平头哥 | 天数智芯 | 几何平均 |
|---:|---:|---:|---:|---:|---:|---:|
| 7.05x | 5.62x | 0.28x | 7.02x | 7.53x | 10.05x | **4.24x** |

**技术摘要：**
- 这不是 GEMM：主要成本是 compare/select、shuffle、候选队列、寄存器/UB 和不规则 page gather，TFLOPS 不能解释性能。
- 总体思路是：先裁剪 causal 有效区，再通过分块局部 TopK 缩小候选集合，最后执行精确排序和 page gather；小形状使用单 kernel，大形状再按芯片选择 packed、radix 或静态分区。

---

## 题目拆解

输入为 `scores[B,H,Q,NB]`。每个 query row 先根据 `q_pos`、`kv_lens` 和 `block_size` 得到 causal 有效前缀，再选择前 `K` 个 block：
```text
valid(block) = block 在 query 的 causal 范围和有效 KV 范围内
order        = score 降序；score 相同时 block_id 升序
out_blocks   = selected block_id
out_pages    = page_table[selected block_id]
```
令 `R=B·H·Q`、`N=NB`、`K=top_k`。算法不允许只返回无序集合；确定性 tie-break、非法位置填 `-1` 和 page-table gather 都是接口语义的一部分。

## 瓶颈分析：主要成本是选择网络，而不是 FMA 峰值

最坏全 valid 行的语义最低流量为：
```text
bytes_min ≈ R · (d·N + 12K)
```
其中 `d∈{2,4}` 是 BF16/FP32 score 的字节数，`12K` 包含两个 int32 输出和 page gather。最大 FP32 workload `(R,N,K)=(32768,4096,128)` 的最低流量为 **560 MiB**。

但流量不是唯一成本。完整 bitonic 网络的比较对数量为：
```text
S(n) = n · log2(n) · (log2(n)+1) / 4
```

| n | bitonic comparator pairs |
|---:|---:|
| 512 | 11,520 |
| 1024 | 28,160 |
| 2048 | 67,584 |
| 4096 | 159,744 |

Ascend `N=4096` 的历史 prefix31 路径，8 个 `sort512`、7 次 merge、`sort256` 和最终 rank 扫描合计约 **210k comparator/shuffle proxy/row**。这远比少量页表读取昂贵，并会进一步放大为 UB 临时状态和长依赖链。

一个关键反例是 packed-u64：它删除了最终 score 重载，却把最大 workload 的估算总流量从 944 MiB 增至 1072 MiB，NVIDIA 分项从 5.74x 降到约 5.50x。减少比较但扩大 scratch，不一定更快。

## 总体方案：先缩小候选集合，再做精确排序

瓶颈分析表明，本题的主要矛盾不是浮点计算量，而是宽选择网络、片上状态与中间数据流量。因此整体优化分为两个层次：
1. **统一精确语义**：所有后端都先裁剪 causal 有效区，并在局部选择、候选合并和最终写回中保持相同的 score/id 顺序。
2. **分层选择与专用执行：**
   - `N≤512` 或部分 `N=1024` 使用 row-resident 单 kernel，减少 scratch 和 kernel launch；
   - `N=1024/2048` 使用 512/1024 宽 chunk-local TopK，再对小候选集合做精排；
   - `N=4096,K=128` 根据芯片选择 radix、静态分区或 packed 路径，避免完整宽 TopK。

候选缩减保持精确的充分条件是：每个 chunk 至少保留局部前 `K`。如果一个元素已被某个 chunk 的局部前 `K` 淘汰，那么同一 chunk 内至少有 `K` 个元素不劣于它，因此它不可能进入全局前 `K`。

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
| warp32 vs wave64 | compare/select 的规约宽度和 shuffle 路径不同 | A100 使用 shared-radix；wave64 后端扩大局部 chunk |
| CU/SM 数量不同 | 单 CTA 串行处理多个 chunk 时，并行度损失不同 | 真武保留 chunk 并行；16-CU 天数避免巨型 tile |
| Ascend Vector/UB | 动态 scatter 和未对齐 UB 地址可能失败 | builder `vsort`、静态 worker、固定候选槽 |
| scratch 表示不同 | packed key 能减少重载，也可能扩大中间流量 | 海光使用 packed 专线；A100 保留 int32 id scratch |
| 编译器选择网络不同 | 相同 TopK 写法可能产生不同依赖链和寄存器状态 | 分别选择 radix、静态分区或 native sort |

关键原则是：**共享精确排序语义，不共享选择网络和 scratch 布局。** 相同的 `N/K` 在六类芯片上需要不同的 chunk、worker 数和 merge 拓扑。

## 核心技术二：确定性 score/id 排序键

TopK 必须满足：
```text
score_a > score_b  => a 在前
score_a = score_b  => id_a < id_b 时 a 在前
```
实现先把浮点数映射为保持数值顺序的整数 key，再把反向编码的 block id 放入低位：
```text
packed_key = ordered_score_bits || reversed_block_id
```
这样一次 compare-and-swap 同时完成 score 比较和 tie-break，不需要维护两套交换网络。中间队列可以只保存紧凑 key 或候选 id；最终阶段重载原始 score 做精确排序，避免 BF16 或截断 key 改变答案。

这个设计还有两个工程收益：
- compare/select 的每次交换只移动一个逻辑对象；
- 相同顺序定义贯穿局部 TopK、候选 merge 和最终 gather，减少后端间的正确性分叉。

## 核心技术三：分块局部 TopK 与小候选精排

对 `N=4096,K=128`，若按 1024 切成 4 块：
```text
4096 scores
  -> 4 × local Top128
  -> 512 candidates
  -> exact Top128
```
相较完整 `sort4096`，它把最终网络从 4096 个元素缩到 512 个元素，并让四个局部任务并行执行。候选阶段只写 id 时，scratch 为：
```text
R · 4 chunks · 128 ids · 4 bytes
```
精确 merge 再读取候选对应的原始 score。是否把 score 一起打包，取决于“少一次随机重载”和“scratch 翻倍”之间的后端权衡；A100 的官方 测试结果表明，int32 id scratch 更合适。

## 核心技术四：有保护的阈值与分区快路径

固定 workload 的 FP32 score 具有稳定分布。大形状先通过 radix 或固定阈值把候选压缩到小容量，再做精确选择：
```text
score scan
  -> threshold / radix bucket
  -> compact candidates
  -> capacity check
  -> exact reload + sort
  -> fallback when overflow
```
正确性不依赖“分布大概率如此”，而依赖三道保护：
1. compact 数量不得超过静态容量；
2. 候选必须覆盖第 K 名所在 bucket；
3. guard 不成立时回退到完整精确路径。

A100 的单读 shared-radix 把最大 workload 的正常 GM 流量压到约 **560.4 MiB**，接近 560 MiB 的语义下界；官方 NVIDIA 分项由 5.74x 提升到 6.07x，约提升 5.7%。MetaX 即使多读约 11% score，删除四个宽 `TopK(1024,128)` 后仍从 4.37x 提升到 4.66x，说明 C550 首先受 selection/occupancy 限制。

## 核心技术五：Ascend packed28 / prefix31 精确选择

910B4 不能直接复用 GPGPU 的动态 histogram/scatter：动态 VEC gather 和未对齐 UB 地址会触发运行时错误。最终使用两条静态路径。

### BF16：packed28 + builder `vsort`
- score 与 id 压成静态宽度 key；
- 固定 worker 处理连续前缀；
- 使用后端 native builder sort；
- merge 后直接 decode page。

### FP32：prefix31 候选 + 精确重载
- 先用高位前缀筛选候选；
- 候选 id 使用固定槽和静态容量；
- 重载原始 FP32 score；
- 执行精确排序和 deterministic tie-break。

这条路径保证六平台正确性，但最终昇腾只有 0.28x。把候选宽度从 `2K` 降到 `K+32` 后仍约 0.26x，说明 decoder 不是主因；真正瓶颈仍是首阶段反复 native sort/merge 与 UB 数据交换。这个负结果限定了后续方向：必须降低选择复杂度，而不是继续微调候选尾部。

## 其他有效优化汇总

| 优化 | 作用 | 使用边界 |
|---|---|---|
| row-resident 小形状 kernel | mask、TopK 和 page gather 一次完成 | 有效前缀和寄存器状态足够小时 |
| int16/int32 候选 id | 缩小中间队列 | block id 范围安全时 |
| causal bound 预计算 | 避免每个 chunk 重复计算有效上界 | 所有分块路径 |
| merge 与 page gather 融合 | 候选确定后直接写两个输出 | 最终精确 merge |
| tuple 专用计算单元 | 只在形状真正改变局部性或并行度时分支 | `B/H/Q/block_size` 固定组合 |
| 静态 grid / worker | 避免动态 launch、UB scatter 和地址不对齐 | Ascend 专用路径 |
| backend-specific chunk | 在并行度和片上状态之间取平衡 | A100 常用 512，wave64 后端常用 1024 |

另外，曾尝试但未拿到收益的方向包括：
- 全行 histogram 和 rank-scatter
- 超宽 vector TopK
- 单 CTA 串行处理四个 chunk
- 只复制 workload 判断却不改变数据流的“伪专用化”。

这些方法要么扩大 scratch 和寄存器状态，要么降低全芯片并行度，因此没有取得正收益。

## 最终成绩
**榜单成绩：4.24x，rank 5，六个平台全部通过。**
