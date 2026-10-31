import torch
from model import SmallResNet
def main():
    x=torch.randn(1,3,32,32); m=SmallResNet(); print('shape_check=',tuple(m(x).shape)); print('model_ready=True')
if __name__=='__main__': main()
