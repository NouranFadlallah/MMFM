import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

from fusion.config import cfg
from fusion.data.dataset import TripleImageDataset
from fusion.models.fusion_model import FusionLateModel


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(loader):
        x1,x2,x3,y,pres = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = y.to(device)
        pres = pres.to(device)

        # modality dropout augmentation
        if cfg.modality_dropout_prob > 0:
            mask = torch.bernoulli(torch.ones_like(pres.float()) * (1 - cfg.modality_dropout_prob)).bool()
            pres = pres & mask
            # zero-out inputs where dropped
            x1 = x1 * pres[:,0].float().view(-1,1,1,1)
            x2 = x2 * pres[:,1].float().view(-1,1,1,1)
            x3 = x3 * pres[:,2].float().view(-1,1,1,1)

        opt.zero_grad()
        fused, branch_logits, weights = model(x1,x2,x3,pres)
        loss = criterion(fused, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x1,x2,x3,y,pres = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device)
            fused, branch_logits, weights = model(x1,x2,x3,pres)
            preds = fused.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return correct / total


def main():
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    # placeholder transforms
    import torchvision.transforms as T
    t1 = T.Compose([T.Resize((224,224)), T.ToTensor()])
    t2 = T.Compose([T.Resize((224,224)), T.ToTensor()])
    t3 = T.Compose([T.Resize((224,224)), T.ToTensor()])

    # use synthetic data if no csv provided
    csv = 'data/manifest.csv'
    try:
        ds = TripleImageDataset(csv, t1,t2,t3)
    except Exception:
        # create synthetic dataset
        from torch.utils.data import TensorDataset
        import os
        os.makedirs('data', exist_ok=True)
        # create dummy csv with missing entries
        import pandas as pd
        df = pd.DataFrame({'img1':['MISSING']*100, 'img2':['MISSING']*100, 'img3':['MISSING']*100, 'label':[0]*100})
        df.to_csv(csv, index=False)
        ds = TripleImageDataset(csv,t1,t2,t3)

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

    model = FusionLateModel(backbone_names=cfg.backbone_names, pretrained=cfg.pretrained, embedding_dim=cfg.embedding_dim, num_classes=cfg.num_classes, fusion_mode=cfg.fusion_mode)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        loss = train_one_epoch(model, loader, opt, device)
        acc = validate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{cfg.epochs} loss={loss:.4f} val_acc={acc:.4f}')

    torch.save(model.state_dict(), 'fusion_checkpoint.pth')


if __name__ == '__main__':
    main()
