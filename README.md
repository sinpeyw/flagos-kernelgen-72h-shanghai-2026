# FlagOS KernelGen 72H 上海站 2026

本仓库开源 FlagOS KernelGen 72H 上海站三道赛题的最终实现及优化报告。代码基于 PyTorch、Triton 与 FlagTree/Triton-TLE，针对六类国产及通用加速器后端分别调度。本文所有成绩均采用赛事最终官方榜单成绩。

## 最终官方榜单

| 题目 | 最终排名 | 最终平均加速比 | 海光 | 沐曦 | 昇腾 | NVIDIA | 平头哥 | 天数智芯 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Fused Add + RMSNorm + Group Quant | **1** | **4.38x** | 4.39x | 2.90x | 5.98x | 3.99x | 4.71x | 4.92x |
| DSA TopK Page Table Transform | **5** | **4.24x** | 7.05x | 5.62x | 0.28x | 7.02x | 7.53x | 10.05x |
| MLA Backward (NoPE, dK+dV) | **2** | **54.04x** | 38.60x | 51.12x | 35.87x | 91.70x | 85.55x | 44.86x |

## 发布代码

| 题目 | 文件 | 说明 |
|---|---|---|
| Task 01 | [`src/task01_fused_add_rmsnorm_group_quant.py`](src/task01_fused_add_rmsnorm_group_quant.py) | 融合残差、RMSNorm 与分组量化的六后端实现 |
| Task 02 | [`src/task02_dsa_topk_page_table_transform.py`](src/task02_dsa_topk_page_table_transform.py) | 分层精确 TopK 与页表转换实现 |
| Task 03 | [`src/task03_mla_bwd_nope_dkdv.py`](src/task03_mla_bwd_nope_dkdv.py) | MLA NoPE backward 与共享 dK+dV 实现 |

## 技术报告

- [Task 01：融合访存、行分组与昇腾双 AIV 映射](reports/task01_fused_add_rmsnorm_group_quant.md)
- [Task 02：分层精确 TopK、结构化快路径与后端专用选择器](reports/task02_dsa_topk_page_table_transform.md)
- [Task 03：共享 P/dS 生产、dK+dV 融合与受控归约](reports/task03_mla_bwd_nope_dkdv.md)

## 目录结构

```text
.
├── src/                # 可直接用于赛事接口的单文件实现
├── reports/            # 三道题的技术报告
├── CITATION.cff
├── LICENSE             # MIT
└── README.md
```

## 使用

这些文件沿用赛事要求的函数签名，并依赖赛事版 PyTorch、Triton 及可选的 FlagTree/Triton-TLE 扩展；它们不是脱离赛事运行时即可执行的命令行程序。将对应单文件作为提交源码，或把入口函数接入相同的评测环境即可。

## 相关项目

- [FlagTree](https://github.com/flagos-ai/FlagTree)
- [Triton](https://triton-lang.org/)
- [FlagTree User Guide](https://docs.flagos.io/projects/FlagTree/en/latest/user_guide/user-guide.html)

## License

本仓库采用 [MIT License](LICENSE)，可自由使用、复制、修改和分发；再分发时需保留版权与许可声明。
