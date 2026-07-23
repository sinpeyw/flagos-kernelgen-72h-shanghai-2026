# Task 01：Fused Add + RMSNorm + Group Quant

## 最终官方成绩

**4.38x，rank 1**。六平台分项为：海光 4.39x、沐曦 2.90x、昇腾 5.98x、NVIDIA 3.99x、平头哥 4.71x、天数智芯 4.92x。

## 算子与瓶颈

每行依次执行：

1. `r = x + residual`，并写出 `residual_out`；
2. `inv_rms = rsqrt(mean(r²) + eps)`；
3. `y = r * inv_rms * gamma`，并写出 `norm_out`；
4. 每 `group_size` 个元素求绝对值最大值，生成 scale 与 int8 量化结果。

在 bf16 输入/输出、int8 量化下，忽略缓存复用和对齐填充，单行不可避免的数据量约为：

```text
读取 x + residual                 4D bytes
写 residual_out + norm_out       4D bytes
写 x_q                           1D bytes
读 gamma                         2D bytes（跨行可缓存）
写 scale                         4D/G bytes
```

主体算术只是逐元素加乘、平方、绝对值与规约，算术强度很低；大形状主要受显存流量、规约代价和 kernel 启动开销约束。优化重点不是增加计算峰值利用率，而是避免中间张量和重复读取。

## 实际有效的优化

### 1. 单次读取并保留 preweighted 数据

融合内核在读取 `x` 与 `residual` 后立即得到 `r`，一边累加平方和、一边写 `residual_out`，同时计算 `r * gamma` 并保留在寄存器/局部值中。得到最终 `inv_rms` 后，直接完成 `norm_out`、group max、scale 与 int8 写回。

这避免了拆分实现中对 `r` 或 `norm_out` 的额外全局内存往返。收益最稳定，也构成所有后端路径的共同基础。

### 2. 按行形状选择 full-row 与 segmented-row

- 小 `M` 使用 full-row，避免多 kernel、partial buffer 和额外归约；
- 大 `D` 使用 2048/4096 宽 segment，降低单 program 的寄存器压力；
- 海光、沐曦、NVIDIA、平头哥根据后端调整 `num_warps`、segment 宽度和 direct-int8 路径；
- NVIDIA 在 `M >= 512, D <= 6144` 时用两行一组，复用同一段 `gamma`，并保持行间写入完全不重叠。

这里的关键不是固定某个 tile，而是让“行并行度、寄存器容量、gamma 复用”同时处于可接受区间。

### 3. 昇腾双 AIV 行映射

对 `M >= 8` 的昇腾形状，入口使用：

```text
row = 2 * program_id + sub_vec_id
```

两个 vector 子核各自拥有一整行，`unit_flag=True`，不存在跨子核写冲突。相比让两个子核协作同一行，这种映射省去了行内同步和 partial 归约；相比只启用一个子核，则恢复了向量侧并行度。该路径仍保持完整 RMSNorm 与逐组量化数学。

### 4. 天数智芯 wave64 分层 group max

天数后端对大 `M` 使用 wave64 分层最大值归约。先在 wave 内规约，再形成 group max，避免用超宽扁平向量承载全部 group 状态。`D=4096/7168` 与其他形状分别选择 stage 数，以控制流水和资源占用。

## 没有转化为最终收益的方向

- 单纯改成 block pointer，没有消除主导的 compulsory I/O；
- 把 RMS 放到 Cube/矩阵路径，搬运与调度成本超过简单向量规约；
- 多阶段实现重复读取完整行，通常不如融合路径；
- 更激进的 multibuffer、跨子核协作和实验性 TLE pipeline 没有形成稳定收益；
- 只调 `num_warps` 或 `num_stages`，无法替代正确的行所有权与 segment 结构。

## 可迁移结论

对于“逐元素 + 行规约 + 分组规约 + 多输出”的算子，应先计算不可避免的字节数，再围绕一次读取、寄存器保留和无冲突所有权设计内核。跨后端移植时，保留数学融合不变，只把行映射、segment 宽度、wave/warp 数和转换路径做后端特化，通常比维护一套统一 tile 更可靠。
