"""Train an ultrasound CycleGAN variant (models/cyclegan.py) between two domains.

Domains are either an MMFM ultrasound dataset name (busbra, busi, busc,
breast, bus_cot), which contributes only its *training* split so the
classifiers' validation/test images never shape the translator, or a folder of
images. The trained checkpoint plugs into training as a preprocessing step:

    python3 training/train.py --dataset breast --single-mode \\
        --us-preprocess cyclegan:checkpoint=cyclegan_cyclegan_breast2busbra.pth

Variants and their defaults (from the notebook's three setups):
  cyclegan  BUS-BRA <-> BrEaST style transfer; lr 2e-4, 100 + 100 decay epochs,
            batch 1, cycle 10, identity 0.5, load 286 / crop 256, image pool 50
  ncia      low -> high quality enhancement; lr 1e-4 constant, 300 epochs,
            batch 2, adv 1 / cycle 10 / VGG 2 / aux 0.1 / identity 5, grad clip 5,
            k-means pseudo-labels (50) every 10 epochs for the first 150
  poas      cyclegan + bottleneck classifier on each domain's own labels
            (dataset names only), cross-entropy weight 1
--paired adds an L1 term between G_A2B(A) and the same-named B image, for
paired data such as the synthetic marker-removal set from
scripts/preprocess_ultrasound.py markers.

Usage:
  python3 scripts/train_cyclegan.py --variant cyclegan --domain-a breast --domain-b busbra
  python3 scripts/train_cyclegan.py --variant ncia --domain-a lowq/ --domain-b highq/
  python3 scripts/train_cyclegan.py --variant cyclegan --paired \\
      --domain-a datasets/ultrasound/markers/marked --domain-b datasets/ultrasound/markers/clean
"""
import argparse
import itertools
import random
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.cyclegan import (
    BottleneckClassifier, ImagePool, PatchDiscriminator, VGGPerceptualLoss, build_generator,
)

DATASET_NAMES = ('busbra', 'busi', 'busc', 'breast', 'bus_cot')
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
VARIANT_DEFAULTS = {
    'cyclegan': dict(lr=2e-4, epochs=100, decay_epochs=100, batch_size=1, load_size=286, pool=50,
                     lambda_adv=1.0, lambda_cycle=10.0, lambda_identity=5.0, identity_a_weight=1.0),
    'ncia': dict(lr=1e-4, epochs=300, decay_epochs=0, batch_size=2, load_size=256, pool=0,
                 lambda_adv=1.0, lambda_cycle=10.0, lambda_identity=5.0, identity_a_weight=0.2,
                 lambda_vgg=2.0, lambda_aux=0.1, grad_clip=5.0, clustering_epochs=150, num_clusters=50),
    'poas': dict(lr=2e-4, epochs=100, decay_epochs=100, batch_size=1, load_size=286, pool=50,
                 lambda_adv=1.0, lambda_cycle=10.0, lambda_identity=5.0, identity_a_weight=1.0, lambda_cls=1.0),
}


# ----------------------------------------------------------------- data

def domain_samples(spec, seed=42):
    """[(path, label or None)] for a dataset name (its training split) or a folder."""
    if spec in DATASET_NAMES:
        from training import train as T

        if spec == 'bus_cot':
            metadata = T._bus_cot_metadata(T.DATASET_KFOLD_SPECS['bus_cot']['default_root'])
            frame, _ = T._patient_split(metadata[metadata['split'] != 'test'], 'patient', seed, 0.1)
        else:
            frame = T._ultrasound_frame(spec, Namespace(max_samples=None, seed=seed, validation_fraction=0.1, image_size=256))[0]
        return [(Path(p), int(l)) for p, l in zip(frame['image'], frame['label'])]
    paths = sorted(p for p in Path(spec).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f'no images found in {spec}')
    return [(p, None) for p in paths]


class UnpairedDataset(torch.utils.data.Dataset):
    """Each A image with a random B image (or, --paired, the B image of the same
    name, with the same random crop/flip). Returns tensors in [-1, 1]."""

    def __init__(self, samples_a, samples_b, size, load_size, variant, paired=False, augment=True):
        self.samples_a, self.samples_b = samples_a, samples_b
        self.size, self.load_size, self.variant = size, load_size, variant
        self.paired, self.augment = paired, augment
        if paired:
            by_name = {p.name: (p, l) for p, l in samples_b}
            self.samples_a = [s for s in samples_a if s[0].name in by_name]
            self.samples_b = [by_name[p.name] for p, _ in self.samples_a]
            if not self.samples_a:
                raise SystemExit('--paired: no file names shared between the two domains')

    def __len__(self):
        return len(self.samples_a)

    def _load(self, path, params):
        from torchvision.transforms import functional as TF

        image = Image.open(path).convert('L')
        if not self.augment:
            image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
        elif self.variant == 'ncia':
            image = TF.affine(image, angle=params['angle'], translate=params['translate'],
                              scale=params['scale'], shear=[0.0])
            image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
        else:
            image = image.resize((self.load_size, self.load_size), Image.Resampling.BICUBIC)
            image = image.crop((params['left'], params['top'], params['left'] + self.size, params['top'] + self.size))
        if self.augment and params['flip']:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array)[None]

    def _params(self):
        margin = self.load_size - self.size
        return {'left': random.randint(0, margin), 'top': random.randint(0, margin), 'flip': random.random() < 0.5,
                'angle': random.uniform(-5, 5), 'scale': random.uniform(0.9, 1.1),
                'translate': [int(random.uniform(-0.05, 0.05) * self.size) for _ in range(2)]}

    def __getitem__(self, index):
        path_a, label_a = self.samples_a[index]
        index_b = index if self.paired else random.randrange(len(self.samples_b))
        path_b, label_b = self.samples_b[index_b]
        params_a = self._params()
        params_b = params_a if self.paired else self._params()
        return {'A': self._load(path_a, params_a), 'B': self._load(path_b, params_b),
                'label_A': -1 if label_a is None else label_a, 'label_B': -1 if label_b is None else label_b,
                'index_A': index, 'index_B': index_b}


# ----------------------------------------------------------------- NCIA pseudo-labels

@torch.no_grad()
def cluster_labels(generator, samples, num_clusters, size, device, previous=None, seed=1234):
    """k-means over the generator's pooled auxiliary features, with cluster ids
    greedily matched to the previous round's so the aux targets stay stable."""
    from sklearn.cluster import KMeans

    dataset = UnpairedDataset(samples, samples, size, size, 'ncia', augment=False)
    features = []
    generator.eval()
    for start in range(0, len(dataset), 16):
        batch = torch.stack([dataset[i]['A'] for i in range(start, min(start + 16, len(dataset)))]).to(device)
        features.append(generator.info_feature(batch).cpu())
    generator.train()
    k = min(num_clusters, len(samples))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(torch.cat(features).numpy())
    if previous is None:
        return labels
    mapping, used = {}, set()
    overlaps = sorted(((np.sum((labels == new) & (previous == old)), new, old)
                       for new in range(k) for old in range(k)), reverse=True)
    for _, new, old in overlaps:
        if new not in mapping and old not in used:
            mapping[new] = old
            used.add(old)
    return np.array([mapping[label] for label in labels])


# ----------------------------------------------------------------- training

def train(args):
    import wandb

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = {**VARIANT_DEFAULTS[args.variant], **{k: v for k, v in vars(args).items() if v is not None}}
    samples_a, samples_b = domain_samples(args.domain_a, args.seed), domain_samples(args.domain_b, args.seed)
    if args.max_images:
        samples_a, samples_b = samples_a[:args.max_images], samples_b[:args.max_images]
    dataset = UnpairedDataset(samples_a, samples_b, args.size, config['load_size'], args.variant, paired=args.paired)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    print(f'{args.variant}: A={args.domain_a} ({len(samples_a)}) B={args.domain_b} ({len(samples_b)}) '
          f'pairs/epoch={len(dataset)} paired={args.paired}')

    G_A2B, G_B2A = build_generator(args.variant).to(device), build_generator(args.variant).to(device)
    D_A, D_B = PatchDiscriminator().to(device), PatchDiscriminator().to(device)
    g_params = list(G_A2B.parameters()) + list(G_B2A.parameters())
    classifier = None
    if args.variant == 'poas':
        labels = {l for _, l in samples_a + samples_b}
        if None in labels:
            raise SystemExit('--variant poas needs labelled domains (use dataset names, not folders)')
        with torch.no_grad():
            shape = G_A2B.encode(torch.zeros(1, 1, args.size, args.size, device=device)).shape[1:]
        classifier = BottleneckClassifier(shape, len(labels)).to(device)
        g_params += list(classifier.parameters())
    opt_G = torch.optim.Adam(g_params, lr=config['lr'], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()), lr=config['lr'], betas=(0.5, 0.999))
    total_epochs = config['epochs'] + config['decay_epochs']

    def linear_decay(epoch):
        return 1.0 - max(0, epoch - config['epochs'] + 1) / float(config['decay_epochs'] + 1)

    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, linear_decay) for o in (opt_G, opt_D)] if config['decay_epochs'] else []
    pool_A, pool_B = ImagePool(config['pool']), ImagePool(config['pool'])
    perceptual = VGGPerceptualLoss().to(device) if args.variant == 'ncia' else None
    clusters_A = clusters_B = None
    out_path = Path(args.out or f'cyclegan_{args.variant}_{Path(args.domain_a).name}2{Path(args.domain_b).name}.pth')
    run = wandb.init(project='mmfm-cyclegan', name=args.run_name or out_path.stem, config=config)

    def save(path):
        torch.save({'variant': args.variant, 'input_nc': 1, 'config': config, 'epoch': epoch + 1,
                    'domain_a': args.domain_a, 'domain_b': args.domain_b,
                    'G_A2B': G_A2B.state_dict(), 'G_B2A': G_B2A.state_dict()}, path)

    for epoch in range(total_epochs):
        if args.variant == 'ncia' and epoch % 10 == 0 and epoch < config['clustering_epochs']:
            clusters_A = cluster_labels(G_A2B, samples_a, config['num_clusters'], args.size, device, clusters_A)
            clusters_B = cluster_labels(G_B2A, samples_b, config['num_clusters'], args.size, device, clusters_B)
        totals = {}
        for batch in loader:
            real_A, real_B = batch['A'].to(device), batch['B'].to(device)
            terms = {}
            if args.variant == 'ncia':
                target_A = torch.as_tensor(clusters_A[batch["index_A"].numpy()], dtype=torch.long, device=device)
                target_B = torch.as_tensor(clusters_B[batch["index_B"].numpy()], dtype=torch.long, device=device)
                fake_B, aux_A = G_A2B(real_A, return_aux=True)
                fake_A, aux_B = G_B2A(real_B, return_aux=True)
                rec_A, aux_rec_A = G_B2A(fake_B, return_aux=True)
                rec_B, aux_rec_B = G_A2B(fake_A, return_aux=True)
                terms['aux'] = config['lambda_aux'] * (F.cross_entropy(aux_A, target_A) + F.cross_entropy(aux_rec_A, target_A)
                                                       + F.cross_entropy(aux_B, target_B) + F.cross_entropy(aux_rec_B, target_B))
                terms['vgg'] = config['lambda_vgg'] * (perceptual(rec_A, real_A) + 0.2 * perceptual(fake_B, real_B))
            else:
                fake_B, fake_A = G_A2B(real_A), G_B2A(real_B)
                rec_A, rec_B = G_B2A(fake_B), G_A2B(fake_A)
            pred_B, pred_A = D_B(fake_B), D_A(fake_A)
            terms['adv'] = config['lambda_adv'] * (F.mse_loss(pred_B, torch.ones_like(pred_B)) +
                                                   F.mse_loss(pred_A, torch.ones_like(pred_A)))
            terms['cycle'] = config['lambda_cycle'] * (F.l1_loss(rec_A, real_A) + F.l1_loss(rec_B, real_B))
            if config['lambda_identity'] > 0:
                terms['identity'] = config['lambda_identity'] * (
                    F.l1_loss(G_A2B(real_B), real_B) + config['identity_a_weight'] * F.l1_loss(G_B2A(real_A), real_A))
            if classifier is not None:
                terms['cls'] = config['lambda_cls'] * (
                    F.cross_entropy(classifier(G_A2B.encode(real_A)), batch['label_A'].to(device)) +
                    F.cross_entropy(classifier(G_B2A.encode(real_B)), batch['label_B'].to(device)))
            if args.paired:
                terms['paired'] = args.lambda_paired * F.l1_loss(fake_B, real_B)
            if args.variant == 'ncia' and epoch == 0:
                # NCIA warm-up: adversarial term at full weight, the rest at 0.1, no identity
                loss_G = terms['adv'] + 0.1 * (terms['cycle'] + terms['aux'] + terms['vgg'])
            else:
                loss_G = sum(terms.values())
            opt_G.zero_grad()
            loss_G.backward()
            if config.get('grad_clip'):
                nn.utils.clip_grad_norm_(g_params, config['grad_clip'])
            opt_G.step()

            fake_A_pool, fake_B_pool = pool_A.query(fake_A), pool_B.query(fake_B)
            loss_D = 0.0
            for D, real, fake in ((D_A, real_A, fake_A_pool), (D_B, real_B, fake_B_pool)):
                pred_real, pred_fake = D(real), D(fake.detach())
                loss_D = loss_D + 0.5 * config['lambda_adv'] * (
                    F.mse_loss(pred_real, torch.ones_like(pred_real)) + F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
            opt_D.zero_grad()
            loss_D.backward()
            if config.get('grad_clip'):
                nn.utils.clip_grad_norm_(itertools.chain(D_A.parameters(), D_B.parameters()), config['grad_clip'])
            opt_D.step()
            for name, value in {**terms, 'G': loss_G, 'D': loss_D}.items():
                totals[name] = totals.get(name, 0.0) + float(value) / len(loader)
        for scheduler in schedulers:
            scheduler.step()
        log = {'epoch': epoch + 1, **{f'loss/{k}': v for k, v in totals.items()}}
        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == total_epochs:
            grid = torch.cat([real_A[:1], fake_B[:1], rec_A[:1], real_B[:1], fake_A[:1], rec_B[:1]], dim=-1)
            log['samples'] = wandb.Image(((grid[0, 0].detach().clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy(),
                                         caption='A | A->B | A->B->A | B | B->A | B->A->B')
        run.log(log)
        print(f'epoch {epoch + 1}/{total_epochs} ' + ' '.join(f'{k}={v:.3f}' for k, v in totals.items()), flush=True)
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save(out_path)
    save(out_path)
    run.save(str(out_path))
    run.finish()
    print(f'saved {out_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=tuple(VARIANT_DEFAULTS), default='cyclegan')
    parser.add_argument('--domain-a', required=True, help=f'dataset name {DATASET_NAMES} or image folder')
    parser.add_argument('--domain-b', required=True)
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--lambda-paired', type=float, default=10.0)
    parser.add_argument('--size', type=int, default=256)
    for name in ('lr', 'batch_size', 'epochs', 'decay_epochs', 'load_size'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=float if name == 'lr' else int, default=None)
    parser.add_argument('--max-images', type=int, default=None, help='cap each domain (the notebook used 100-200)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--sample-every', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', default=None)
    parser.add_argument('--run-name', default=None)
    train(parser.parse_args(argv))


if __name__ == '__main__':
    main()
