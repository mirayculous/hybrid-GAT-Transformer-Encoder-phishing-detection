# ==============================================================================
# PHISHING DETECTION — HTML-ONLY MODEL (TinyBERT)
# Dataset : html.xlsx → kolom: Category (spam/ham), Data (HTML content)
# Platform: Kaggle Notebook (GPU T4 recommended)
# Ablation : Model tunggal HTML saja menggunakan TinyBERT_General_4L_312D
#
# TinyBERT (Jiao et al., 2020) — Knowledge Distillation dari BERT
#   • 4 transformer layers (vs BERT-base: 12 layer)
#   • Hidden dim: 312 (vs BERT-base: 768) — ← PERHATIAN: beda dengan model lain!
#   • Intermediate dim: 1200, 12 attention heads (masing-masing dim 26)
#   • ~14.5M parameter (vs BERT-base: ~110M) — ~7.5x lebih kecil
#   • Distillasi 2 tahap: General Distillation + Task-specific Distillation
#   • Cocok untuk resource-constrained deployment / perbandingan efisiensi
# ==============================================================================

# ==============================================================================
# CELL 1 — Install Dependencies
# ==============================================================================
"""
!pip install transformers -q
!pip install beautifulsoup4 -q
!pip install openpyxl -q
"""

# ==============================================================================
# CELL 2 — Imports & Seed
# ==============================================================================
import os, re, json, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel   # ← TinyBERT menggunakan BertModel

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
    HTML_DATA_PATH  = "/kaggle/input/datasets/guchiopara/look-before-you-leap/html.xlsx"
    PROCESSED_DIR   = "/kaggle/working/processed_tinybert"
    MODEL_SAVE_PATH = "/kaggle/working/best_model_tinybert.pt"
    HISTORY_PATH    = "/kaggle/working/history_tinybert.json"

    # Label
    LABEL_MAP = {'spam': 1, 'ham': 0}

    # Split
    TEST_SIZE = 0.15
    VAL_SIZE  = 0.15

    # Model — TinyBERT 4-layer 312-dim
    BERT_MODEL      = "huawei-noah/TinyBERT_General_4L_312D"
    BERT_HIDDEN_DIM = 312        # ← BEDA! TinyBERT hidden dim = 312, bukan 768
    MAX_HTML_LEN    = 256
    STAT_FEAT_DIM   = 16
    HTML_PROJ_DIM   = 128

    # Classifier — disesuaikan dengan hidden dim lebih kecil
    FUSION_DIM     = 256
    CLASSIFIER_DIM = 128
    DROPOUT        = 0.3

    # Training — LR lebih tinggi karena model lebih kecil
    BATCH_SIZE  = 32             # ← bisa lebih besar karena model lebih ringan
    ACCUM_STEPS = 2              # efektif batch = 32×2 = 64
    EPOCHS      = 30
    LR          = 5e-5           # ← lebih tinggi karena model kecil
    WEIGHT_DECAY= 1e-4
    PATIENCE    = 5
    GRAD_CLIP   = 1.0

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
os.makedirs(cfg.PROCESSED_DIR, exist_ok=True)
print(f"Model           : HTML-only | TinyBERT ({cfg.BERT_MODEL})")
print(f"Hidden dim      : {cfg.BERT_HIDDEN_DIM}  ← lebih kecil dari model lain (768)")
print(f"Device          : {cfg.DEVICE}")
print(f"Effective Batch : {cfg.BATCH_SIZE * cfg.ACCUM_STEPS}")
print(f"MAX_HTML_LEN    : {cfg.MAX_HTML_LEN}")


# ==============================================================================
# CELL 4 — Tokenizer
# ==============================================================================
tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)


# ==============================================================================
# CELL 5 — Load Dataset (HTML only)
# ==============================================================================
def load_dataset():
    print("=" * 55)
    print(" LOADING HTML DATASET")
    print("=" * 55)

    df = pd.read_excel(cfg.HTML_DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"\nHTML file : {len(df)} baris | kolom: {df.columns.tolist()}")

    df = df.rename(columns={'category': 'label', 'data': 'html'})
    df['label'] = df['label'].str.lower().str.strip().map(cfg.LABEL_MAP)

    before = len(df)
    df.dropna(subset=['html', 'label'], inplace=True)
    df['html']  = df['html'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)
    df = df[df['html'] != '']
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
# CELL 6 — HTML Feature Extraction
# ==============================================================================
def extract_html_text(raw_html):
    if not isinstance(raw_html, str) or not raw_html.strip(): return "[PAD]"
    try:
        soup = BeautifulSoup(raw_html[:500_000], 'html.parser')
        for tag in soup(['script', 'style', 'noscript', 'head']): tag.decompose()

        visible = re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True)).lower().strip()
        struct  = []
        for form in soup.find_all('form'):
            struct.append(f"formaction {form.get('action', '')[:80]}")
        for inp in soup.find_all('input'):
            struct.append(f"input_{inp.get('type', 'text').lower()} {inp.get('name', '').lower()}")
        for a in soup.find_all('a', href=True)[:20]:
            struct.append(f"href {a['href'][:80]}")
        for sc in soup.find_all('script', src=True)[:10]:
            struct.append(f"scriptsrc {sc['src'][:60]}")
        for fr in soup.find_all('iframe', src=True)[:5]:
            struct.append(f"iframe {fr['src'][:60]}")

        return visible[:500] + ' [SEP] ' + ' '.join(struct)[:200]
    except Exception:
        return "[PAD]"


def extract_stat_features(raw_html):
    if not isinstance(raw_html, str) or not raw_html.strip():
        return [0.0] * cfg.STAT_FEAT_DIM
    try:
        soup        = BeautifulSoup(raw_html[:500_000], 'html.parser')
        links       = soup.find_all('a', href=True)
        ext_links   = [a for a in links if a['href'].startswith('http')]
        scripts     = soup.find_all('script')
        ext_scripts = soup.find_all('script', src=True)
        forms       = soup.find_all('form')
        ext_forms   = [f for f in forms if str(f.get('action', '')).startswith('http')]
        inputs      = soup.find_all('input')
        pwd_inputs  = soup.find_all('input', {'type': 'password'})
        hid_inputs  = soup.find_all('input', {'type': 'hidden'})
        iframes     = soup.find_all('iframe')
        images      = soup.find_all('img')
        hl, tl      = len(raw_html), len(soup.get_text())

        feats = [
            min(len(forms), 20) / 20.0,      min(len(inputs), 30) / 30.0,
            min(len(links), 100) / 100.0,    min(len(ext_links), 100) / 100.0,
            min(len(scripts), 30) / 30.0,    min(len(ext_scripts), 20) / 20.0,
            min(len(iframes), 10) / 10.0,    min(len(images), 50) / 50.0,
            int(bool(pwd_inputs)),            int(bool(hid_inputs)),
            int(bool(ext_forms)),             int(bool(iframes)),
            len(ext_links) / (len(links) + 1e-6),
            len(ext_scripts) / (len(scripts) + 1e-6),
            min(tl, 50000) / 50000.0,        min(hl, 500000) / 500000.0,
        ]
        return [min(max(f, 0.0), 1.0) for f in feats]
    except Exception:
        return [0.0] * cfg.STAT_FEAT_DIM


def tokenize_html(text):
    if not text: text = "[PAD]"
    tokens = tokenizer.tokenize(text)
    if len(tokens) > cfg.MAX_HTML_LEN - 2:
        half   = (cfg.MAX_HTML_LEN - 3) // 2
        tokens = tokens[:half] + ['[SEP]'] + tokens[-half:]

    enc = tokenizer(
        tokenizer.convert_tokens_to_string(tokens),
        max_length=cfg.MAX_HTML_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids'     : enc['input_ids'].squeeze(0),
        'attention_mask': enc['attention_mask'].squeeze(0),
    }


# ==============================================================================
# CELL 7 — Preprocessing Pipeline (HTML only)
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
            html_raw    = str(row['html'])
            html_text   = extract_html_text(html_raw)
            html_tokens = tokenize_html(html_text)
            html_stat   = torch.tensor(extract_stat_features(html_raw), dtype=torch.float)
            label       = int(row['label'])

            samples.append({
                'html_input_ids': html_tokens['input_ids'],
                'html_attn_mask': html_tokens['attention_mask'],
                'html_stat'     : html_stat,
                'label'         : torch.tensor(label, dtype=torch.long)
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
# CELL 8 — Dataset & DataLoader
# ==============================================================================
class HTMLDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def collate_fn(batch):
    return {
        'html_input_ids': torch.stack([b['html_input_ids'] for b in batch]),
        'html_attn_mask': torch.stack([b['html_attn_mask'] for b in batch]),
        'html_stat'     : torch.stack([b['html_stat']      for b in batch]),
        'label'         : torch.stack([b['label']          for b in batch]),
    }


def make_loaders(train_data, val_data, test_data):
    kw = dict(collate_fn=collate_fn, num_workers=4,
               pin_memory=True, prefetch_factor=2, persistent_workers=True)
    return (
        DataLoader(HTMLDataset(train_data), batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw),
        DataLoader(HTMLDataset(val_data),   batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
        DataLoader(HTMLDataset(test_data),  batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
    )


# ==============================================================================
# CELL 9 — Model: HTML-only TinyBERT
# ==============================================================================
class HTMLTinyBERT(nn.Module):
    """
    HTML-only phishing detector menggunakan TinyBERT_General_4L_312D.

    TinyBERT adalah model hasil knowledge distillation dari BERT-base.
    Memiliki 4 layer transformer dan hidden dim 312 (bukan 768).
    Berguna sebagai baseline ringan untuk analisis trade-off efisiensi vs akurasi.

    Arsitektur fusion disesuaikan dengan hidden dim 312:
      312 (CLS) + 128 (stat_proj) → fusion [→128] → classifier
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(cfg.BERT_MODEL)

        # Freeze semua, unfreeze layer transformer terakhir (layer ke-3, idx -1)
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.bert.encoder.layer[-1].parameters():
            p.requires_grad = True

        self.stat_proj = nn.Sequential(
            nn.Linear(cfg.STAT_FEAT_DIM, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU())

        # Input fusion: 312 + 128 = 440 (bukan 768+128=896 seperti model lain)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.BERT_HIDDEN_DIM + 128, 256),   # 440 → 256
            nn.LayerNorm(256), nn.ReLU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(256, cfg.HTML_PROJ_DIM))            # 256 → 128

        self.classifier = nn.Sequential(
            nn.Linear(cfg.HTML_PROJ_DIM, cfg.FUSION_DIM),
            nn.LayerNorm(cfg.FUSION_DIM), nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.FUSION_DIM, cfg.CLASSIFIER_DIM),
            nn.ReLU(), nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.CLASSIFIER_DIM, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, stat_feat):
        cls = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state[:, 0, :]
        fused = self.fusion(torch.cat([cls, self.stat_proj(stat_feat)], dim=-1))
        return self.classifier(fused)


# ==============================================================================
# CELL 10 — Training Utilities
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
            logits = model(
                batch['html_input_ids'].to(cfg.DEVICE),
                batch['html_attn_mask'].to(cfg.DEVICE),
                batch['html_stat'].to(cfg.DEVICE))
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
            logits = model(
                batch['html_input_ids'].to(cfg.DEVICE),
                batch['html_attn_mask'].to(cfg.DEVICE),
                batch['html_stat'].to(cfg.DEVICE))
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
# CELL 11 — Main Training Loop
# ==============================================================================
def train():
    train_data, val_data, test_data = run_full_preprocessing()
    train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data)
    print(f"\nDataLoaders → Train:{len(train_loader)} | Val:{len(val_loader)} | Test:{len(test_loader)} batch")

    model     = HTMLTinyBERT().to(cfg.DEVICE)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel           : HTML-only | TinyBERT ({cfg.BERT_MODEL})")
    print(f"Hidden dim      : {cfg.BERT_HIDDEN_DIM}")
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    bert_trainable = [p for p in model.bert.parameters() if p.requires_grad]
    other_params   = [p for p in model.parameters()
                      if not any(p is bp for bp in model.bert.parameters())]

    optimizer = torch.optim.AdamW([
        {'params': bert_trainable, 'lr': cfg.LR,      'weight_decay': cfg.WEIGHT_DECAY},
        {'params': other_params,   'lr': cfg.LR * 10, 'weight_decay': cfg.WEIGHT_DECAY},
    ])

    total_steps = (len(train_loader) // cfg.ACCUM_STEPS) * cfg.EPOCHS
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)
    criterion   = nn.CrossEntropyLoss()
    early_stop  = EarlyStopping(patience=cfg.PATIENCE)
    scaler      = GradScaler()

    history = {k: [] for k in ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1','val_auc']}
    best_f1 = 0.0

    print("\n" + "=" * 60)
    print(f"  TRAINING: HTML-only | TinyBERT")
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
# CELL 12 — Final Test Evaluation
# ==============================================================================
def final_test(model, test_loader):
    print("\n" + "=" * 60)
    print(f"  FINAL TEST EVALUATION — HTML-only | TinyBERT")
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
# CELL 13 — Jalankan
# ==============================================================================
if __name__ == '__main__':
    model, test_loader = train()
    results = final_test(model, test_loader)

    print("\n" + "=" * 60)
    print("  SELESAI! — HTML-only | TinyBERT")
    print(f"  Model   → {cfg.MODEL_SAVE_PATH}")
    print(f"  History → {cfg.HISTORY_PATH}")
    print("=" * 60)
