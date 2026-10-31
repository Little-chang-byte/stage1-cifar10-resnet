$ErrorActionPreference = 'Stop'
python src/data_analysis.py
python src/train.py
python src/evaluate.py
python src/predict.py
Write-Host '演示完成，请展示 outputs/ 和 weights/ 目录。'
