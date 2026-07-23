# Task 01：Fused Add + RMSNorm + Group Quant

> FlagOS 72小时算子赏金挑战赛 – 上海站 · 最终榜单截至 2026-07-20 12:00（UTC+8） · 对应实现见 [`src/task01_fused_add_rmsnorm_group_quant.py`](../src/task01_fused_add_rmsnorm_group_quant.py)

## 写在前面

**最终榜单成绩：4.38x，rank 1，六个平台全部通过。**

| 海光 | 沐曦 | 昇腾 | NVIDIA | 平头哥 | 天数智芯 | 几何平均 |
|---:|---:|---:|---:|---:|---:|---:|
| 4.39x | 2.90x | 5.98x | 3.99x | 4.71x | 4.92x | **4.38x** |

**技术摘要：**
- 该算子的主体算术强度只有约 **0.666 FLOP/Byte**，大 workload 首先受 HBM、寄存器 live range 和规约依赖限制，而不是 Tensor/Cube 算力限制。
- 总体思路是：减少HBM的读写，维持单 kernel、单遍输入、四输出直接写回；再按`M`维度大小、warp/wave 宽度和昇腾双 AIV 能力进行专用的分支优化。
- 主要的收益来自：preweighted 单遍融合、单 program 分段流式、昇腾双 AIV 独立行映射，以及 warp/wave 原生规约与跨行 `gamma` 复用等。

---

## 题目拆解

设输入为 `x,residual ∈ BF16[M,D]`，每 `G` 个元素构成一个量化组：
```text
r       = x + residual
inv_rms = rsqrt(mean(r²) + eps)
y       = r * inv_rms * gamma
scale_g = max(abs(y_g)) / 127
q_g     = clamp(round(y_g / scale_g), -127, 127)
```
必须同时写出：

- `residual_out = r`；
- `norm_out = y`；
- `x_q = q`；
- `x_scale = scale`。

困难在于 RMS 需要整行规约，而量化又需要每组规约。如果先写中间张量再启动第二个 kernel，整行数据会被重复搬运；如果把整行全部常驻，`D=4096–8192` 又会产生过宽的寄存器状态。

## 瓶颈分析：大形状是带宽问题，小形状是延迟问题

令 `N=M·D`，忽略 `abs/max/round/clamp` 等非 FLOP 指令，常规浮点工作量约为：
```text
F ≈ 6N + 3M + 2N/G
```
API 无法消除的最低主存流量为：
```text
B_min = 9N + 4N/G + s_gamma·D
```
其中 `9N` 来自两路 BF16 输入、两路 BF16 输出和一路 INT8 输出。若 `gamma` 被 cache，`G=128` 时：
```text
AI ≈ 6.016 / 9.031 = 0.666 FLOP/Byte
```
A100 的 FP32 ridge point 为 `19.5 TFLOPS / 1555 GB/s = 12.54 FLOP/Byte`，比该算子高约 **18.8 倍**。因此从算法级 Roofline 模型看，大 workload 不会首先受 FP32 峰值限制；增加第二遍读取或全局 scratch 必然抬高带宽下界。

| 规模 | 主导因素 | 直接含义 |
|---|---|---|
| `M≤32` | kernel launch、行规约依赖、低 occupancy | 优先单 kernel 和较小状态 |
| `M≈128` | 延迟向带宽过渡 | full-row 与 segmented 需要独立路由 |
| `M≥512` | HBM、cache、访存 | 坚持单遍输入，避免 scratch |

海光/沐曦在 W4 下分别达到 1331/1498 GB/s；A100、平头哥在 W8 下达到 1374/2170 GB/s。统一 num_warps 会直接损失有效带宽。


## 总体方案：统一单遍数据流，再按 shape 和芯片选择执行映射

瓶颈分析表明，本题的主要瓶颈不是计算量，而是访存带宽。同时，但统一的物理执行配置无法同时适配六类芯片。因此整体优化分为两个层次：
1. **统一数据流，减少访存开销**：所有后端均采用“输入单遍读取、片上完成 RMS 与分组规约、四路结果直接写回”，避免张量的重复读写和额外 kernel开销。
2. **分层执行映射，深度专用优化：**
   - 小 `M` 使用整行常驻，优先减少启动与控制开销；大 `M,D` 使用分段流式计算，限制寄存器和 UB 状态。
   - 再根据不同芯片的 warp/wave 宽度、计算单元数量和昇腾双 AIV 能力，选择不同的规约树、并发度和按行分工。

后续优化均围绕这两个层次展开：统一数据流负责逼近 I/O 下界，分层执行映射负责把同一数据流映射到专用执行配置和专用硬件。

## 核心技术一：芯片专用分支

**官方评测硬件：**

| 平台 | 确认型号 | 设备名 / 架构 | 计算单元 | 显存 | warp / wave | 关键运行时 |
|---|---|---|---:|---:|---:|---|
| 海光 Hygon | BW1000 | `BW` / `gfx936` | 80 CU | 64 GiB | — | PyTorch 2.4.1，HIP 6.1 |
| 沐曦 MetaX | MetaX C550 | `MetaX C550` | 104 CU | 约 64 GiB | 64 | PyTorch 2.8.0 + MetaX 3.3，CUDA ABI 11.6 |
| 华为昇腾 | Ascend 910B4 | `Ascend910B4-1` | — | 约 61 GiB | — | PyTorch 2.6.0，PrivateUse1=`npu` |
| NVIDIA | A100-SXM4-40GB | `NVIDIA A100-SXM4-40GB` / SM 8.0 | 108 SM | 约 40 GiB | 32 | PyTorch 2.9.0，CUDA 12.8 |
| 平头哥 T-Head | PPU-ZW810E（真武 810E） | `PPU-ZW810E` | 64 CU | 约 96 GiB | 32 | PyTorch 2.9.0，CUDA ABI 12.9 |
| 天数智芯 | Iluvatar BI-V150 | `Iluvatar BI-V150` / CC 7.1 | 16 CU | 32 GiB | 64 | PyTorch 2.7.1，CUDA ABI 10.2| 

**架构差异的具体优化影响：**

| 架构差异 | 对本题的影响 | 对应分支 |
|---|---|---|
| warp32 vs wave64 | group size 128/256 的规约树不同 | warp32 分段；wave64 先做 2/4 个原生最大值 |
| CU/SM 数量不同 | resident rows 与 program 粒度不同 | 16-CU 天数避免巨型 tile；108-SM A100提高行并行 |
| 昇腾 Vector/Cube 分离 | RMS/quant 是向量规约，不适合搬到 Cube | 使用 AIV 按行分工，不做 Cube RMS |
| UB/寄存器容量不同 | D8192 全行状态可能 spill 或 lowering 失败 | 2048/4096 segment、短 live range |
| 编译器转换路径不同 | INT8 cast、mask、cache hint 的成本不同 | direct-int8 与 backend-specific store |

关键原则是：**共享数学公式，不共享物理 tile。** 同一组 `D/G` 在不同后端上可能对应完全不同的最优规约树和并发度。

## 核心技术二：单遍 preweighted 融合

加载 `x` 与 `residual` 后立即同时执行三件事：
```text
r        = fp32(x) + fp32(residual)
sumsq   += r * r
weighted = r * fp32(gamma)
```
`r` 直接写入 `residual_out`；`weighted` 与局部 group max 留在片上。整行 `sumsq` 得到 `inv_rms` 后，再把已保留的 `weighted` 转成 `norm_out`、`scale` 和 `x_q`。

与“Add → RMSNorm → Quant”三个 kernel 相比，它删除了：
- `r` 的一次完整写后重读；
- `norm_out` 的一次完整写后重读；
- 两次额外 kernel launch；
- 中间张量的 cache 污染。

收益的上界不来自少做几次乘法，而来自把流量压到接近 `B_min`。因此这个融合是所有后端共同的第一原则。

## 核心技术三：单 program 分段流式

大 `D` 下不能同时常驻完整行的所有 FP32 状态。实现把列切为 2048/4096 宽 segment，但仍在同一 program 中完成：
1. 每段读取 `x/residual/gamma`；
2. 累加全行 `sumsq`；
3. 保存该段 `weighted` 和 group max；
4. 得到全行 `inv_rms` 后依次写回各段。

这与“两 kernel 分段”有本质区别：
| 方案 | 输入读取 | 全局中间量 | 风险 |
|---|---:|---:|---|
| 两阶段 RMS + quant | 至少两遍 | 有 | 带宽翻倍、launch 增加 |
| 全行常驻 | 一遍 | 无 | D8192 live state 过宽 |
| **单 program 分段流式** | **一遍** | **无** | 通过 segment 控制状态 |

segment 不是越大越好：过小会增加 helper 和 slice 循环，过大则降低 occupancy。最终按后端和形状选择 2048/4096，而不是使用单一全局常量。

## 核心技术四：昇腾双 AIV 按行独立分工

昇腾 910B4 可通过 `sub_vec_id()` 暴露两个 Vector 子核。最有效的映射不是两个 AIV 协作一行，而是：
```text
row = 2 * program_id + sub_vec_id
```
每个 AIV 独立完成一整行的 add、RMS、group max 和量化，并写入互不重叠的地址。这样：
- 不需要跨 AIV 合并 `sumsq` 或 group max；
- 不需要原子操作或全局 partial；
- 两个子核都得到有效工作；
- `unit_flag=True` 只承担受支持的同步语义。
  
官方单变量测试中，昇腾分项从单 AIV 结构的 1.51x，依次提升到双 AIV 局部路由的 1.71x、全大形状的 1.87x，再到启用 `unit_flag` 的 2.07x。

相反，曾尝试Cube RMS、输入 DSA 地址空间转换和 multibuffer 均没有形成可复现正收益。

## 核心技术五：让规约树匹配 warp/wave，而非抽象 group size

`G=128/256` 在 warp32、wave64 和 AIV 上不是同一个物理问题。

### NVIDIA：两行共享 `gamma`
A100 的大 `M` 分支让一个 program 处理两行。`gamma` 没有行维度，同一 segment 只加载一次并广播到两行；两行的输出区域完全分离。收益来自减少 `gamma` 指令与 cache 压力，同时保持足够的 row-level parallelism。

### 天数智芯：wave64 分层最大值
BI-V150 用 64-lane wave。对 G128/G256：
```text
G128: 2 × max64 → max2
G256: 4 × max64 → max4
```
相比把 128/256 个元素作为一个超宽状态交给通用 reduction，这种结构缩短跨 wave 依赖链，并降低 group max 的 live state。它体现了同一数学规约必须匹配硬件原生执行宽度。

## 其他有效优化汇总

| 优化 | 作用 | 使用边界 |
|---|---|---|
| 小 `M` full-row | 最少 launch 与控制逻辑 | 并行度足够前优先 |
| `cache_modifier` 区分输入/输出 | 输入偏 cache、流式输出减少污染 | 后端支持时启用 |
| direct-int8 store | 避免多余中间转换 | 昇腾、平头哥、天数专线 |
| `num_stages=1/2` 分形状选择 | 控制流水与寄存器占用 | D4096/7168 与 D8192 分开 |
| 静态 backend tag | 入口只做一次设备识别 | 避免热路径动态查询 |
| 按行独占写回 | 删除 atomic 和 partial reduce | 所有正式路径的共同约束 |

另外，曾尝试但未拿到收益的方向包括：
- 单纯 block pointer 改写
- Cube RMS
- 全局 P/partial 式多阶段
- 过宽 row tile 和盲目 multibuffer。

这些方法没有减少主 I/O，反而增加状态或编译风险，导致没有取得正收益。

## 最终成绩
**榜单成绩：4.38x，rank 1，六个平台全部通过。**
