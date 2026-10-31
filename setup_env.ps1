$ErrorActionPreference = 'Stop'
Write-Host '项目环境：Windows CPU；未检测到 nvidia-smi，因此不启用 CUDA。'
conda env create --prefix "$PSScriptRoot\.conda" -f "$PSScriptRoot\environment.yml" --yes
conda run --prefix "$PSScriptRoot\.conda" python -c "import torch, torchvision; print('torch=', torch.__version__); print('torchvision=', torchvision.__version__); print('cuda_available=', torch.cuda.is_available())"
