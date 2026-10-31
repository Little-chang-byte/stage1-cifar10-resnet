import argparse, torch, matplotlib.pyplot as plt
from pathlib import Path
from model import SmallResNet
from dataset import get_loaders

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--full',action='store_true'); args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; train, val=get_loaders(small=not args.full)
    model=SmallResNet().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=torch.nn.CrossEntropyLoss(); best=0; hist=[]
    for epoch in range(3 if not args.full else 10):
        model.train(); total=correct=running=0
        for x,y in train:
            x,y=x.to(device),y.to(device); opt.zero_grad(); z=model(x); loss=loss_fn(z,y); loss.backward(); opt.step(); running+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); total+=len(y)
        model.eval(); vc=vt=0
        with torch.no_grad():
            for x,y in val: z=model(x.to(device)); vc+=(z.argmax(1).cpu()==y).sum().item(); vt+=len(y)
        acc=vc/vt; hist.append((running/total,acc)); print(f'epoch {epoch+1}: loss={running/total:.4f}, val_acc={acc:.4f}')
        if acc>=best: best=acc; Path('weights').mkdir(exist_ok=True); torch.save(model.state_dict(),'weights/best_resnet.pt')
    Path('outputs').mkdir(exist_ok=True); plt.plot([x[0] for x in hist],label='loss'); plt.plot([x[1] for x in hist],label='val_acc'); plt.legend(); plt.savefig('outputs/training_curve.png',dpi=150); print('device=',device,'best_val_acc=',best)
if __name__=='__main__': main()
