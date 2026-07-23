# Task 02：DSA TopK Page Table Transform

## 最终官方成绩

**4.24x，rank 5**。六平台分项为：海光 7.05x、沐曦 5.62x、昇腾 0.28x、NVIDIA 7.02x、平头哥 7.53x、天数智芯 10.05x。

## 算子与主要开销

输入 `scores[B,H,Q,NB]`，每个 query 只能在 causal/有效 KV 范围内选择 block。选择顺序必须是 score 降序，分数相同时 block id 升序；随后通过 `page_table` 生成 block 与 page 输出。

直接对 `NB` 个元素做完整 bitonic sort 的复杂度约为 `O(NB log² NB)`，而输出只需要 `K` 个元素。大形状中真正浪费的是：

- 对绝大多数不会进入 TopK 的元素执行全排序；
- 多轮 kernel 之间写入并重读大 scratch；
- score 与 id 分离后反复搬运；
- 小形状上启动开销超过有效计算。

因此最终结构是“尽早缩小候选集，再对小候选集做精确排序”，并为固定赛事形状设置有保护条件的快路径。

## 实际有效的优化

### 1. score 与 id 打包，保持精确 tie-break

将可比较的浮点 score key 与 block id 打包成一个排序 key。score 部分保证数值降序，id 部分编码为同分时升序。局部选择、候选合并和最终 gather 都复用这一比较语义，避免 score/id 两套 compare-and-swap。

在 scratch 中只保存候选 id 或紧凑 key；能安全容纳时使用更窄的 id 表示，降低候选队列带宽。最终阶段重新读取原 score 做精确比较，防止紧凑中间表示改变答案。

### 2. 分块局部 TopK + 小候选合并

对 `NB=2048/4096`，每行先按 512/1024 元素分块，每块只保留 `LOCAL_K` 个候选。第二阶段只在 `num_chunks × LOCAL_K` 中合并，再访问 `page_table`。

若 `LOCAL_K >= K`，每块丢弃的元素不可能进入全局 TopK；因此这是精确的候选缩减，不是近似。它把全量排序问题转成多个并行局部选择和一次小规模 merge。

### 3. 小形状 row-resident 单 kernel

对 `NB=512` 或部分 `NB=1024` 形状，让一个 program 在片上完成有效区裁剪、选择和 page gather，不创建 scratch。天数智芯的小 workload 尤其受益，因为这里节省的主要是启动与中间队列成本。

### 4. 固定分布形状的 guarded fast path

赛事的若干 FP32 workload 具有固定的 Gaussian 分布和固定 `(NB,K)`：

- `NB=2048, K=64`：固定阈值/分区先 compact 候选，再做精确选择；
- `NB=4096, K=128`：NVIDIA 使用 radix 8×4、容量 128 的结构化路径；其他 GPGPU 后端使用固定分区 compact；
- 任何候选溢出或保护条件不成立时，回退到精确选择。

固定阈值本身不是正确性依据；正确性来自容量检查、精确重载与 fallback。这样才能把分布特化转化为可发布的精确实现。

### 5. 昇腾单独设计选择器

昇腾没有复用 GPGPU 的超宽 bitonic 模板：

- BF16 使用紧凑 packed key 与 builder `vsort`；
- FP32 小 `NB` 走完整前缀候选；
- FP32 大 `NB` 用 prefix31 过采样候选，再重读原 score 做精确排序和 decode；
- worker 数、地址对齐和候选容量均为静态配置。

这条路径保证了 6/6 正确性，但最终只有 0.28x，也是总分最明确的短板。主要问题是候选生成和精确重排仍然过重，并非其他五个平台的分层 TopK 思路失效。

## 没有转化为最终收益的方向

- 对全部 `NB` 做大向量 bitonic，比较网络和寄存器状态过大；
- 动态 gather/动态形状容易触发后端编译或地址空间问题；
- 展开的 compare-exchange 网络产生过多 IR，编译时间与代码体积不可控；
- 没有严格溢出保护的 histogram/阈值筛选无法保证 TopK；
- 用一套 chunk、warp 和 scratch 布局覆盖六个后端，会牺牲小形状或特定后端。

## 可迁移结论

TopK 页表转换的第一原则是把“候选缩减”和“最终精确顺序”分开。局部阶段追求并行与紧凑，最终阶段保留完整 score 与确定性 tie-break。固定 workload 可以特化，但必须让保护条件和精确 fallback 成为算法的一部分，而不是依赖输入分布侥幸通过。
