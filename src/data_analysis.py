from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets

def main():
    out=Path('outputs'); out.mkdir(exist_ok=True)
    ds=datasets.CIFAR10(root='data', train=True, download=True)
    names=ds.classes; counts=[0]*len(names)
    for y in ds.targets: counts[y]+=1
    pd.DataFrame({'class':names,'count':counts}).to_csv(out/'class_counts.csv',index=False)
    plt.figure(figsize=(10,4)); plt.bar(names,counts); plt.xticks(rotation=35); plt.tight_layout(); plt.savefig(out/'class_distribution.png',dpi=150); plt.close()
    print(pd.DataFrame({'class':names,'count':counts}).to_string(index=False))
if __name__=='__main__': main()
