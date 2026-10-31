# CIFAR-10 手写 ResNet 图像分类

这是 AI 方向第一阶段考核的最小可运行项目。项目使用 CIFAR-10 数据集，先用 Pandas/Matplotlib 做类别统计，再使用 PyTorch 手写残差网络完成训练、验证、保存和推理。

## 运行

```powershell
pip install -r requirements.txt
python src/data_analysis.py
python src/train.py
python src/evaluate.py
python src/predict.py
```

默认是 small 模式，方便 CPU 快速验证；`python src/train.py --full` 可运行更完整训练。首次运行需要联网下载 CIFAR-10。

## 目录

- `src/model.py`：BasicBlock 与 SmallResNet
- `src/dataset.py`：数据集和 DataLoader
- `src/data_analysis.py`：类别统计和图表
- `src/train.py`：训练、验证、权重保存
- `src/evaluate.py`：测试集评估
- `src/predict.py`：输入形状和模型推理检查
- `outputs/`：真实运行生成的图表
- `weights/`：训练生成的权重

## 说明

本项目优先保证流程完整和结果可复现。最终指标以实际运行日志为准，不在代码或文档中预先填写虚构数值。

## 当前验证结果

项目已在独立 CPU 环境中验证：PyTorch `2.15.0+cpu`、torchvision `0.29.2+cpu`，模型输入 `[2, 3, 32, 32]` 时输出 `[2, 10]`。small 模式在 500 个测试样本上的实际准确率为 `0.3840 (192/500)`。该结果用于验证完整流程，不代表完整 CIFAR-10 训练的最终性能。

## 训练配置

small 模式使用 2000 个训练样本、500 个测试样本、batch size 64、Adam 优化器、学习率 0.001 和 3 个 epoch。由于当前机器没有可用 NVIDIA CUDA 驱动，实验使用 CPU 完成。

详细验证输出见 `VERIFICATION.md`，考核要求对应关系见 `202500800011-常歌-后端阶段一考核.md`。
