from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets

def main():
    out=Path('outputs'); out.mkdir(exist_ok=True)
    ds=datasets.CIFAR10(root='data', train=True, download=True)
    names=ds.classes; counts=[0]*len(names)
    for y in ds.targets: counts[y]+=1
    sample_array = np.stack([np.asarray(ds.data[i], dtype=np.float32) for i in range(min(100, len(ds.data)))])
    channel_mean = sample_array.mean(axis=(0, 1, 2)) / 255.0
    channel_std = sample_array.std(axis=(0, 1, 2)) / 255.0
    pd.DataFrame({'channel':['R','G','B'], 'mean':channel_mean, 'std':channel_std}).to_csv(out/'pixel_statistics.csv', index=False)
    pd.DataFrame({'class':names,'count':counts}).to_csv(out/'class_counts.csv',index=False)
    plt.figure(figsize=(10,4)); plt.bar(names,counts); plt.xticks(rotation=35); plt.tight_layout(); plt.savefig(out/'class_distribution.png',dpi=150); plt.close()
    print(pd.DataFrame({'class':names,'count':counts}).to_string(index=False))
    print('numpy_sample_shape=', sample_array.shape)
    print('numpy_channel_mean=', channel_mean.round(4).tolist())
    print('numpy_channel_std=', channel_std.round(4).tolist())
if __name__=='__main__': main()
