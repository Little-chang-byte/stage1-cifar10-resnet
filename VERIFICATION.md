# 项目验证记录

## 环境验证

执行环境：`D:\后端_test\project\.venv-codex-pytorch`

实际输出：

```text
PyTorch version: 2.15.0+cpu
Torchvision version: 0.29.2+cpu
CUDA available: False
model output shape: (2, 10)
PyTorch environment test: PASS
```

## 数据分析验证

执行：

```powershell
.\.venv-codex-pytorch\Scripts\python.exe src\data_analysis.py
```

生成：

- `outputs/class_counts.csv`
- `outputs/class_distribution.png`

## 训练和评估验证

执行 small 模式训练后生成：

- `weights/best_resnet.pt`
- `outputs/training_curve.png`

评估实际输出：

```text
test_accuracy=0.3840 (192/500)
```

该结果来自 500 个测试样本。small 模式用于 CPU 环境下的流程验证，不能等同于完整数据集和充分训练后的最终指标。

## Git 验证

远程仓库：<https://github.com/Little-chang-byte/stage1-cifar10-resnet>

主分支：`main`

仓库保留了从项目初始化、数据分析、模型实现、训练推理、环境配置到最终材料整理的分阶段提交历史。
