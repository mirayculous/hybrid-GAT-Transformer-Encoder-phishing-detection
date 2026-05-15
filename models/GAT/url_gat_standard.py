# ==============================================================================
# PHISHING DETECTION — URL-ONLY MODEL (STANDARD GAT)
# Dataset : url.xlsx → kolom: Category (spam/ham), Data (URL)
# Platform: Kaggle Notebook (GPU T4 recommended)
# Ablation : Model tunggal URL saja menggunakan Standard GATConv
# ==============================================================================

# ==============================================================================
# CELL 1 — Install Dependencies
# ==============================================================================
"""
!pip install torch-geometric -q
!pip install tldextract       -q
!pip install openpyxl         -q
!pip install torch-scatter torch-sparse -q \
    -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cu118.html
"""

# ==============================================================================
# CELL 2 — Imports & Seed
# ==============================================================================
import os, re, math, json, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from urllib.parse import urlparse

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import tldextract

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = True

print("Imports OK")


# ==============================================================================
# CELL 3 — Configuration
# ==============================================================================
class Config:
    # Paths
    URL_DATA_PATH   = "/kaggle/input/datasets/guchiopara/look-before-you-leap/URL.xlsx"
    PROCESSED_DIR   = "/kaggle/working/processed_gat_std"
    MODEL_SAVE_PATH = "/kaggle/working/best_model_gat_std.pt"
    HISTORY_PATH    = "/kaggle/working/history_gat_std.json"

    # Label
    LABEL_MAP = {'spam': 1, 'ham': 0}

    # Split
    TEST_SIZE = 0.15
    VAL_SIZE  = 0.15

    # URL Graph
    CHAR_EMBED_DIM = 32
    MAX_NODE_TEXT  = 50
    NODE_FEAT_DIM  = 64

    # GAT (Standard GATConv)
    GAT_HIDDEN_DIM = 64
    GAT_OUT_DIM    = 128
    GAT_HEADS      = 4
    GAT_DROPOUT    = 0.3
    GAT_LAYERS     = 2

    # Classifier
    FUSION_DIM     = 128
    DROPOUT        = 0.3

    # Training
    BATCH_SIZE  = 32
    ACCUM_STEPS = 2           # efektif batch = 32×2 = 64
    EPOCHS      = 30
    LR          = 1e-3
    WEIGHT_DECAY= 1e-4
    PATIENCE    = 5
    GRAD_CLIP   = 1.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
print(f"Model           : URL-only | Standard GAT")
print(f"Device          : {cfg.DEVICE}")
print(f"Effective Batch : {cfg.BATCH_SIZE * cfg.ACCUM_STEPS}")


# ==============================================================================
# CELL 4 — Load Dataset (URL only)
# ==============================================================================
def load_dataset():
    print("=" * 55)
    print(" LOADING URL DATASET")
    print("=" * 55)

    df = pd.read_excel(cfg.URL_DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"\nURL file : {len(df)} baris | kolom: {df.columns.tolist()}")

    df = df.rename(columns={'category': 'label', 'data': 'url'})
    df['label'] = df['label'].str.lower().str.strip().map(cfg.LABEL_MAP)

    before = len(df)
    df.dropna(subset=['url', 'label'], inplace=True)
    df['url']   = df['url'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)
    df = df[df['url'] != '']
    df.reset_index(drop=True, inplace=True)

    print(f"\nSetelah cleaning : {len(df)} baris (dihapus: {before - len(df)})")
    print(f"  spam (phishing)  : {(df['label'] == 1).sum()}")
    print(f"  ham (legitimate) : {(df['label'] == 0).sum()}")
    return df


def stratified_split(df):
    df_tv, df_test = train_test_split(
        df, test_size=cfg.TEST_SIZE, stratify=df['label'], random_state=SEED)
    df_train, df_val = train_test_split(
        df_tv, test_size=cfg.VAL_SIZE / (1 - cfg.TEST_SIZE),
        stratify=df_tv['label'], random_state=SEED)

    for d in [df_train, df_val, df_test]:
        d.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 55)
    print(" STRATIFIED SPLIT")
    print("=" * 55)
    for name, d in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        print(f"  {name:5s}: {len(d):5d} sampel  "
              f"(spam={d['label'].sum():4d}, ham={(d['label'] == 0).sum():4d})")
    return df_train, df_val, df_test


# ==============================================================================
# CELL 5 — URL → Hierarchical Graph Builder
# ==============================================================================
_CHARS     = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
CHAR_VOCAB = {ch: idx + 1 for idx, ch in enumerate(_CHARS)}
CHAR_VOCAB['<PAD>'] = 0
CHAR_VOCAB['<UNK>'] = len(CHAR_VOCAB)
VOCAB_SIZE = len(CHAR_VOCAB)

BRAND_LIST = [
    'amazon','google','paypal','apple','microsoft','facebook','netflix',
    'instagram','twitter','linkedin','dropbox','ebay','bank','bca',
    'mandiri','bni','bri','dhl','fedex','chase','citibank','hsbc',
    'visa','mastercard','yahoo','gmail','wellsfargo','bankofamerica',
    'steam','shopee','tokopedia'
]
RISKY_TLDS = {
    'tk','ml','ga','cf','gq','xyz','top','club','online','site',
    'info','pw','cc','su','ws','icu'
}
SENSITIVE_KEYWORDS = [
    'login','signin','verify','account','secure','banking','update',
    'confirm','password','credential','wallet','authenticate',
    'validation','recovery','support','suspended','unlock',
    'billing','payment','invoice','urgent'
]
NODE_TYPES = {
    'root':0,'subdomain':1,'subpart':2,'hyptok':3,
    'domain':4,'tld':5,'path':6,'pathseg':7,'query':8
}
NUM_NODE_TYPES = len(NODE_TYPES)


def _levenshtein(s1, s2):
    if not s1 or not s2: return 1.0
    if len(s1) < len(s2): s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1] / max(len(s1), len(s2))


def _min_brand_dist(token):
    token = token.lower()
    return 1.0 if not token else min(_levenshtein(token, b) for b in BRAND_LIST)


def _encode_chars(text, max_len=50):
    text = text.lower()[:max_len]
    idx  = [CHAR_VOCAB.get(c, CHAR_VOCAB['<UNK>']) for c in text]
    idx += [0] * (max_len - len(idx))
    return idx


def _node_features(text, ntype_str):
    text = (text or '').lower(); tl = len(text)
    ntype = NODE_TYPES.get(ntype_str, 0)
    bd    = _min_brand_dist(text)
    GEO   = {'jp','co','id','uk','us','de','fr','cn','au','ca','in','sg','my'}
    freq  = defaultdict(int)
    for c in text: freq[c] += 1
    ent   = (-sum((v / tl) * math.log2(v / tl + 1e-9) for v in freq.values())
             if tl > 0 else 0.0)
    oh    = [0] * NUM_NODE_TYPES; oh[ntype] = 1
    return [
        min(tl, 100) / 100.0, int(bd < 0.2), float(bd),
        int(text in GEO), int(any(k in text for k in SENSITIVE_KEYWORDS)),
        min(text.count('-'), 10) / 10.0,
        sum(c.isdigit() for c in text) / (tl + 1e-6),
        min(sum(c in '!@#$%^&*()[]{}<>' for c in text), 10) / 10.0,
        min(ent, 8.0) / 8.0, int(text in RISKY_TLDS),
        int(text.replace('.', '').replace('-', '').isdigit()),
        int('%' in text),
    ] + oh


def build_url_graph(url):
    try:
        uc  = url if url.startswith('http') else 'http://' + url
        ext = tldextract.extract(uc); parsed = urlparse(uc)
    except Exception:
        x  = torch.zeros((1, cfg.MAX_NODE_TEXT), dtype=torch.long)
        hc = torch.zeros((1, 21), dtype=torch.float)
        ei = torch.zeros((2, 0), dtype=torch.long)
        return Data(char_indices=x, handcrafted=hc, edge_index=ei, num_nodes=1)

    subdomain = ext.subdomain or ''; domain = ext.domain or ''
    tld = ext.suffix or ''; path = parsed.path or ''; query = parsed.query or ''
    nc, nhc, nid, edges = [], [], {}, []

    def add_node(name, text, ntype):
        i = len(nc)
        nc.append(_encode_chars(text, cfg.MAX_NODE_TEXT))
        nhc.append(_node_features(text, ntype))
        nid[name] = i; return i

    def ae(u, v): edges.append((u, v)); edges.append((v, u))

    root = add_node('root', url[:cfg.MAX_NODE_TEXT], 'root')
    dom  = add_node('domain', domain, 'domain'); ae(root, dom)
    tln  = add_node('tld', tld, 'tld'); ae(dom, tln)

    if subdomain:
        sub = add_node('subdomain', subdomain, 'subdomain')
        ae(root, sub); ae(sub, dom)
        parts = [p for p in subdomain.split('.') if p][:6]; pp = sub
        for pi, part in enumerate(parts):
            pn = add_node(f'sp_{pi}', part, 'subpart'); ae(sub, pn)
            if pi > 0: ae(pp, pn)
            pp = pn
            toks = [t for t in part.split('-') if t]
            if len(toks) > 1:
                pt = pn
                for ti, tok in enumerate(toks[:8]):
                    tn = add_node(f'ht_{pi}_{ti}', tok, 'hyptok'); ae(pn, tn)
                    if ti > 0: ae(pt, tn)
                    pt = tn

    if path and path != '/':
        pn = add_node('path', path[:cfg.MAX_NODE_TEXT], 'path')
        ae(root, pn); ae(dom, pn)
        segs = [s for s in path.split('/') if s][:6]; ps = pn
        for si, seg in enumerate(segs):
            sn = add_node(f'pseg_{si}', seg, 'pathseg'); ae(pn, sn)
            if si > 0: ae(ps, sn)
            ps = sn

    if query:
        qn = add_node('query', query[:cfg.MAX_NODE_TEXT], 'query'); ae(root, qn)

    ct = torch.tensor(nc, dtype=torch.long)
    ht = torch.tensor(nhc, dtype=torch.float)
    ei = (torch.tensor(edges, dtype=torch.long).t().contiguous()
          if edges else torch.zeros((2, 0), dtype=torch.long))
    return Data(char_indices=ct, handcrafted=ht, edge_index=ei, num_nodes=len(nc))


# ==============================================================================
# CELL 6 — Preprocessing Pipeline (URL only)
# ==============================================================================
def preprocess_split(df, split_name):
    save_path = os.path.join(cfg.PROCESSED_DIR, f'{split_name}.pt')
    if os.path.exists(save_path):
        print(f"[{split_name:5s}] ✓ Cache ditemukan → load dari {save_path}")
        return torch.load(save_path, weights_only=False)

    print(f"\n[{split_name:5s}] Memproses {len(df)} sampel...")
    samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        try:
            url_graph = build_url_graph(str(row['url']))
            label     = int(row['label'])
            samples.append({
                'url_graph': url_graph,
                'label'    : torch.tensor(label, dtype=torch.long)
            })
        except Exception as e:
            print(f"  ⚠ Skip baris {idx}: {e}")

    torch.save(samples, save_path)
    print(f"[{split_name:5s}] Tersimpan {len(samples)} sampel → {save_path}")
    return samples


def run_full_preprocessing():
    df = load_dataset()
    df_train, df_val, df_test = stratified_split(df)
    return (preprocess_split(df_train, 'train'),
            preprocess_split(df_val,   'val'),
            preprocess_split(df_test,  'test'))


# ==============================================================================
# CELL 7 — Dataset & DataLoader
# ==============================================================================
class URLDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_fn(batch):
    return {
        'url_batch': Batch.from_data_list([b['url_graph'] for b in batch]),
        'label'    : torch.stack([b['label'] for b in batch]),
    }


def make_loaders(train_data, val_data, test_data):
    kw = dict(collate_fn=collate_fn, num_workers=4,
               pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return (
        DataLoader(URLDataset(train_data), batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw),
        DataLoader(URLDataset(val_data),   batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
        DataLoader(URLDataset(test_data),  batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
    )


# ==============================================================================
# CELL 8 — Model: URL-GAT Standard
# ==============================================================================
class CharEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_dim, out_dim):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size + 2, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        emb  = self.emb(x)
        mask = (x != 0).float().unsqueeze(-1)
        return self.proj((emb * mask).sum(1) / (mask.sum(1) + 1e-6))


class URLGATStandard(nn.Module):
    """
    URL-only phishing detector menggunakan Standard GATConv (Veličković et al., 2018).
    Pipeline: Char-level embedding + 21 handcrafted features
              → node projection → 2-layer GAT
              → global mean+max pool → MLP classifier
    """
    def __init__(self):
        super().__init__()
        char_out       = 32
        self.char_emb  = CharEmbedder(VOCAB_SIZE, cfg.CHAR_EMBED_DIM, char_out)
        self.node_proj = nn.Sequential(
            nn.Linear(char_out + 21, cfg.NODE_FEAT_DIM),
            nn.LayerNorm(cfg.NODE_FEAT_DIM), nn.ReLU())

        # ── Standard GATConv layers ──────────────────────────────────────────
        self.gat_layers = nn.ModuleList()
        in_dim = cfg.NODE_FEAT_DIM
        for i in range(cfg.GAT_LAYERS):
            last  = (i == cfg.GAT_LAYERS - 1)
            out_d = cfg.GAT_OUT_DIM if last else cfg.GAT_HIDDEN_DIM
            heads = 1 if last else cfg.GAT_HEADS
            self.gat_layers.append(
                GATConv(in_dim, out_d, heads=heads, concat=not last,
                        dropout=cfg.GAT_DROPOUT, add_self_loops=True))
            in_dim = out_d * heads if not last else out_d

        self.drop     = nn.Dropout(cfg.GAT_DROPOUT)
        self.pool_proj = nn.Sequential(
            nn.Linear(cfg.GAT_OUT_DIM * 2, cfg.GAT_OUT_DIM),
            nn.LayerNorm(cfg.GAT_OUT_DIM), nn.ReLU())

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(cfg.GAT_OUT_DIM, cfg.FUSION_DIM),
            nn.LayerNorm(cfg.FUSION_DIM), nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_DIM, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, url_batch):
        ci = url_batch.char_indices.to(cfg.DEVICE)
        hc = url_batch.handcrafted.to(cfg.DEVICE)
        ei = url_batch.edge_index.to(cfg.DEVICE)
        bv = url_batch.batch.to(cfg.DEVICE)

        x = self.node_proj(torch.cat([self.char_emb(ci), hc], dim=-1))
        if ei.shape[1] == 0:
            ei = torch.zeros((2, 1), dtype=torch.long, device=cfg.DEVICE)

        for i, gat in enumerate(self.gat_layers):
            x = gat(x, ei)
            if i < cfg.GAT_LAYERS - 1: x = F.elu(self.drop(x))

        graph_emb = self.pool_proj(
            torch.cat([global_mean_pool(x, bv), global_max_pool(x, bv)], dim=-1))
        return self.classifier(graph_emb)


# ==============================================================================
# CELL 9 — Training Utilities
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.best = None; self.counter = 0; self.stop = False

    def __call__(self, score):
        if self.best is None or score > self.best + self.min_delta:
            self.best = score; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True
        return self.stop


def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        with autocast():
            logits = model(batch['url_batch'].to(cfg.DEVICE))
            labels = batch['label'].to(cfg.DEVICE)
            loss   = criterion(logits, labels) / cfg.ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % cfg.ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        total_loss += loss.item() * cfg.ACCUM_STEPS
        preds_all.extend(logits.detach().argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    if len(loader) % cfg.ACCUM_STEPS != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

    n = len(loader)
    return (total_loss / n,
            accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average='binary'))


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all, probs_all = 0.0, [], [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        with autocast():
            logits = model(batch['url_batch'].to(cfg.DEVICE))
            labels = batch['label'].to(cfg.DEVICE)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        probs_all.extend(F.softmax(logits, -1)[:, 1].cpu().tolist())
        preds_all.extend(logits.argmax(-1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    n = len(loader)
    return {
        'loss'  : total_loss / n,
        'acc'   : accuracy_score(labels_all, preds_all),
        'prec'  : precision_score(labels_all, preds_all, average='binary', zero_division=0),
        'rec'   : recall_score(labels_all, preds_all, average='binary', zero_division=0),
        'f1'    : f1_score(labels_all, preds_all, average='binary', zero_division=0),
        'auc'   : roc_auc_score(labels_all, probs_all),
        'preds' : preds_all,
        'labels': labels_all,
    }


# ==============================================================================
# CELL 10 — Main Training Loop
# ==============================================================================
def train():
    train_data, val_data, test_data = run_full_preprocessing()
    train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data)
    print(f"\nDataLoaders → Train:{len(train_loader)} | Val:{len(val_loader)} | Test:{len(test_loader)} batch")

    model     = URLGATStandard().to(cfg.DEVICE)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel           : URL-only | Standard GAT")
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    total_steps = (len(train_loader) // cfg.ACCUM_STEPS) * cfg.EPOCHS
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    criterion  = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=cfg.PATIENCE)
    scaler     = GradScaler()

    history  = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1','val_auc']}
    best_f1  = 0.0

    print("\n" + "=" * 60)
    print("  TRAINING: URL-only | Standard GAT")
    print("=" * 60)

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        print("─" * 50)

        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1','val_auc'],
                        [tr_loss, val['loss'], tr_acc, val['acc'], tr_f1, val['f1'], val['auc']]):
            history[k].append(v)

        print(f"  Train → Loss:{tr_loss:.4f}  Acc:{tr_acc:.4f}  F1:{tr_f1:.4f}")
        print(f"  Val   → Loss:{val['loss']:.4f}  Acc:{val['acc']:.4f}  "
              f"Prec:{val['prec']:.4f}  Rec:{val['rec']:.4f}  "
              f"F1:{val['f1']:.4f}  AUC:{val['auc']:.4f}")

        if val['f1'] > best_f1:
            best_f1 = val['f1']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_f1': val['f1'], 'val_auc': val['auc']},
                       cfg.MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved! (F1={val['f1']:.4f})")

        if early_stop(val['f1']):
            print(f"\n⚠  Early stopping di epoch {epoch}"); break

    with open(cfg.HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory tersimpan → {cfg.HISTORY_PATH}")
    return model, test_loader


# ==============================================================================
# CELL 11 — Final Test Evaluation
# ==============================================================================
def final_test(model, test_loader):
    print("\n" + "=" * 60)
    print("  FINAL TEST EVALUATION — URL-only | Standard GAT")
    print("=" * 60)

    ckpt = torch.load(cfg.MODEL_SAVE_PATH, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    print(f"Model terbaik dari epoch {ckpt['epoch']} (Val F1={ckpt['val_f1']:.4f})\n")

    res = eval_epoch(model, test_loader, nn.CrossEntropyLoss())

    print(f"  Loss      : {res['loss']:.4f}")
    print(f"  Accuracy  : {res['acc']:.4f}  ({res['acc'] * 100:.2f}%)")
    print(f"  Precision : {res['prec']:.4f}")
    print(f"  Recall    : {res['rec']:.4f}")
    print(f"  F1-Score  : {res['f1']:.4f}")
    print(f"  AUC-ROC   : {res['auc']:.4f}")
    print()
    print(classification_report(res['labels'], res['preds'],
                                target_names=['Ham (Legitimate)', 'Spam (Phishing)']))

    cm = confusion_matrix(res['labels'], res['preds'])
    print(f"Confusion Matrix:")
    print(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
    print(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")
    return res


# ==============================================================================
# CELL 12 — Jalankan
# ==============================================================================
if __name__ == '__main__':
    model, test_loader = train()
    results = final_test(model, test_loader)

    print("\n" + "=" * 60)
    print("  SELESAI! — URL-only | Standard GAT")
    print(f"  Model   → {cfg.MODEL_SAVE_PATH}")
    print(f"  History → {cfg.HISTORY_PATH}")
    print("=" * 60)
