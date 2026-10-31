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
