from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from model import SmallResNet


def main(index=0):
    """Load one real CIFAR-10 test image and save its prediction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    raw = datasets.CIFAR10(root='data', train=False, download=True)
    ds = datasets.CIFAR10(root='data', train=False, download=False, transform=transform)
    image, label = raw[index]
    tensor, _ = ds[index]

    model = SmallResNet(num_classes=len(raw.classes)).to(device)
    weight_path = Path('weights/best_resnet.pt')
    if not weight_path.exists():
        raise FileNotFoundError(f'模型权重不存在: {weight_path}')
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted = int(probabilities.argmax().item())

    out = Path('outputs')
    out.mkdir(exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'True: {raw.classes[label]} | Pred: {raw.classes[predicted]}')
    plt.tight_layout()
    plt.savefig(out / 'single_prediction.png', dpi=150)
    plt.close()
    print('image_index=', index)
    print('true_label=', raw.classes[label])
    print('predicted_label=', raw.classes[predicted])
    print('confidence=', f'{probabilities[predicted].item():.4f}')
    print('prediction_image=', out / 'single_prediction.png')


if __name__=='__main__': main()
