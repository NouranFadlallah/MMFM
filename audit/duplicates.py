"""Section C: near-duplicate detection (aHash/dHash/pHash candidates confirmed by
SSIM on thumbnails), duplicate grouping, and split-leakage reporting."""
import numpy as np
import pandas as pd
from scipy import ndimage

HASH_COLUMNS = ('ahash', 'dhash', 'phash')
MAX_HAMMING = 4          # candidate if any of the three 64-bit hashes is within this distance
SSIM_THRESHOLD = 0.95    # paper's "structurally identical" threshold
SSIM_SIZE = 64
SSIM_BATCH = 4096
CHUNK = 512


def _hash_matrix(frame):
    return np.stack([frame[c].to_numpy(dtype=np.int64).view(np.uint64) for c in HASH_COLUMNS], axis=1)


def candidate_blocks(frame, groups=None, skip_same_group=False):
    """Yield (i, j, distances) blocks of pairs i < j with any hash distance
    <= MAX_HAMMING, one row-chunk at a time so callers can confirm and discard
    candidates as they go instead of holding them all in memory."""
    hashes = _hash_matrix(frame)
    n = len(hashes)
    group_codes = pd.factorize(pd.Series(groups).fillna('__none__'))[0] if groups is not None else None
    for start in range(0, n, CHUNK):
        block = hashes[start:start + CHUNK]
        distances = np.stack([np.bitwise_count(block[:, k:k + 1] ^ hashes[None, :, k]) for k in range(3)], axis=-1)
        close = (distances <= MAX_HAMMING).any(axis=-1)
        close &= np.arange(n)[None, :] > np.arange(start, start + len(block))[:, None]
        if skip_same_group and group_codes is not None:
            close &= group_codes[start:start + len(block)][:, None] != group_codes[None, :]
        ii, jj = np.nonzero(close)
        if len(ii):
            yield ii + start, jj, distances[ii, jj]


def _downsample(thumbs):
    if thumbs.shape[1] == SSIM_SIZE:
        return thumbs.astype(np.float32)
    factor = thumbs.shape[1] // SSIM_SIZE
    n = thumbs.shape[0]
    return thumbs[:, :SSIM_SIZE * factor, :SSIM_SIZE * factor].reshape(
        n, SSIM_SIZE, factor, SSIM_SIZE, factor).mean(axis=(2, 4)).astype(np.float32)


def ssim_pairs(images_a, images_b):
    """Mean SSIM (Wang et al. 2004, Gaussian window sigma=1.5) for batches of
    same-size grayscale images in 0-255."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: ndimage.gaussian_filter(x, sigma=(0, 1.5, 1.5))
    mu_a, mu_b = blur(images_a), blur(images_b)
    var_a = blur(images_a * images_a) - mu_a ** 2
    var_b = blur(images_b * images_b) - mu_b ** 2
    cov = blur(images_a * images_b) - mu_a * mu_b
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return ssim_map.mean(axis=(1, 2))


def find_duplicates(frame, thumbs, skip_same_group=False):
    """Confirmed near-duplicate pairs with distances, SSIM and both rows'
    labels/splits; returns (pairs, number of hash candidates checked)."""
    frame = frame.reset_index(drop=True)
    valid = frame[list(HASH_COLUMNS)].notna().all(axis=1).to_numpy()
    index = np.nonzero(valid)[0]
    sub = frame.iloc[index]
    groups = sub['group'] if 'group' in sub else None
    small = _downsample(thumbs)
    kept, candidates = [], 0
    for i, j, d in candidate_blocks(sub, groups, skip_same_group):
        candidates += len(i)
        i, j = index[i], index[j]
        for s in range(0, len(i), SSIM_BATCH):
            scores = ssim_pairs(small[i[s:s + SSIM_BATCH]], small[j[s:s + SSIM_BATCH]])
            keep = scores >= SSIM_THRESHOLD
            if keep.any():
                kept.append((i[s:s + SSIM_BATCH][keep], j[s:s + SSIM_BATCH][keep],
                             d[s:s + SSIM_BATCH][keep], scores[keep]))
    if not kept:
        return pd.DataFrame(), candidates
    i, j, d, scores = (np.concatenate(parts) for parts in zip(*kept))
    pairs = pd.DataFrame({
        'i': i, 'j': j, 'ahash_dist': d[:, 0], 'dhash_dist': d[:, 1], 'phash_dist': d[:, 2], 'ssim': scores,
    })
    for side, col in (('a', 'i'), ('b', 'j')):
        rows = frame.loc[pairs[col].to_numpy()]
        for field in ('dataset', 'sample_id', 'label_name', 'label', 'used', 'group', 'split', 'fold'):
            pairs[f'{field}_{side}'] = rows[field].to_numpy() if field in rows else None
    pairs['same_group'] = (pairs['group_a'] == pairs['group_b']) & pairs['group_a'].notna()
    pairs['label_conflict'] = pairs['label_a'].notna() & pairs['label_b'].notna() & (pairs['label_a'] != pairs['label_b'])
    pairs['group_id'] = _union_find(pairs['i'].to_numpy(), pairs['j'].to_numpy())
    return pairs, candidates


def _union_find(i, j):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(i, j):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(a) for a in i]
    codes = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    return [codes[r] for r in roots]


def split_leakage(pairs):
    """Duplicate pairs where both images are used for training but land in
    different hold-out splits or different k-folds (and aren't already the same
    patient/case, which the group-aware splitters handle)."""
    if pairs.empty:
        return pairs
    both_used = pairs['used_a'].astype(bool) & pairs['used_b'].astype(bool)
    split_differs = pairs['split_a'].notna() & pairs['split_b'].notna() & (pairs['split_a'] != pairs['split_b'])
    fold_differs = pairs['fold_a'].notna() & pairs['fold_b'].notna() & (pairs['fold_a'] != pairs['fold_b'])
    leaks = pairs[both_used & (split_differs | fold_differs) & ~pairs['same_group']].copy()
    leaks['leak_type'] = np.where(split_differs[leaks.index], 'holdout_split', 'kfold')
    return leaks


def group_split_conflicts(frame):
    """Patients/cases whose images were assigned to more than one hold-out split."""
    used = frame[frame['used'].astype(bool) & frame['group'].notna() & frame['split'].notna()]
    counts = used.groupby('group')['split'].nunique()
    return counts[counts > 1].index.tolist()
