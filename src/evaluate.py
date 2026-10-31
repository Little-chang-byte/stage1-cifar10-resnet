import torch
from model import SmallResNet
from dataset import get_loaders
def main():
    _,loader=get_loaders(small=True); m=SmallResNet(); m.load_state_dict(torch.load('weights/best_resnet.pt',map_location='cpu')); m.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader: c+=(m(x).argmax(1)==y).sum().item(); t+=len(y)
    print(f'test_accuracy={c/t:.4f} ({c}/{t})')
if __name__=='__main__': main()
