"""
========================================================================
 PHISHING DETECTION — TRANSFORMER ENCODER (SecBERT Pretrained)
 Dataset  :
    - train.csv
    - val.csv
    - test.csv

 Arsitektur:
    HTML Text
      → AutoTokenizer (SecBERT)
      → AutoModel / SecBERT (pretrained, partial freeze)
      → Attention Pooling atas last_hidden_state

    Numerical Features
      → Feature Projection

    Fusion
      → MLP Classifier

 Optimizations:
    ✓ Pretrained SecBERT (jackaduma/SecBERT) — domain cybersecurity
    ✓ Partial Freezing (freeze semua kecuali 2 layer terakhir + pooler)
    ✓ Attention Pooling atas token sequence
    ✓ Mixed Precision Training (GPU)
    ✓ Gradient Accumulation
    ✓ AdamW + Cosine Scheduler dengan Warmup
    ✓ Label Smoothing
    ✓ Weighted Loss
    ✓ Early Stopping
    ✓ ROC Curve + Loss Curve + Confusion Matrix
    ✓ Complete Evaluation Metrics
========================================================================
"""

# ============================================================
# INSTALL (jalankan di Kaggle cell terpisah jika perlu)
# !pip install transformers -q
# ============================================================

import os
import math
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# ── Gunakan AutoTokenizer & AutoModel agar fleksibel ──────────────────
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup
)

warnings.filterwarnings("ignore")

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True

print(f"\n[INFO] Device : {DEVICE}")

# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = "/kaggle/input/datasets/raynaldadikasumarga/split-url-html/split/90_10/HTML/train.csv"
VAL_CSV   = "/kaggle/input/datasets/raynaldadikasumarga/split-url-html/split/90_10/HTML/val.csv"
TEST_CSV  = "/kaggle/input/datasets/raynaldadikasumarga/split-url-html/split/90_10/HTML/test.csv"

# ── SecBERT: pretrained di domain cybersecurity ───────────────────────
PRETRAINED_MODEL = "jackaduma/SecBERT"

# SecBERT berbasis BERT-base → 12 transformer layer (index 0–11)
# Freeze layer 0–9, unfreeze layer 10–11 (2 layer terakhir)
UNFREEZE_LAST_N_LAYERS = 4

MAX_SEQ_LEN  = 256          # max token (batas asli 512)
BERT_DIM     = 768          # hidden size SecBERT (sama dengan BERT-base)

DROPOUT      = 0.2

BATCH_SIZE   = 32           # lebih kecil karena BERT-base lebih berat
GRAD_ACCUM   = 2            # effective batch = 16 * 2 = 32
NUM_EPOCHS   = 50
LR           = 3e-5         # LR kecil untuk fine-tuning
WEIGHT_DECAY = 1e-2
# PATIENCE     = 5

MODEL_PATH   = "best_secbert_html.pt"

# ============================================================
# NUMERICAL FEATURES
# ============================================================

NUM_FEAT_COLS = [
    "word_count",
    "num_forms",
    "num_inputs",
    "num_links",
    "num_ext_links",
    "num_scripts",
    "num_ext_scripts",
    "num_iframes",
    "num_images",
    "has_password_input",
    "has_hidden_input",
    "has_external_form",
    "has_iframe",
    "ext_link_ratio",
    "ext_script_ratio",
    "text_length",
    "html_length"
]

# ============================================================
# DATASET
# ============================================================

class PhishingDataset(Dataset):
    """
    Setiap sampel menghasilkan:
      - input_ids      : [MAX_SEQ_LEN]   token id dari SecBERT tokenizer
      - attention_mask : [MAX_SEQ_LEN]   1=real token, 0=padding
      - num_feat       : [17]            fitur numerik ternormalisasi
      - label          : int (0/1)
    """

    def __init__(
        self,
        df,
        tokenizer,
        scaler      = None,
        fit_scaler  : bool = False
    ):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer

        num_arr = (
            df[NUM_FEAT_COLS]
            .fillna(0)
            .values
            .astype(np.float32)
        )

        if fit_scaler:
            self.scaler  = StandardScaler()
            self.num_arr = self.scaler.fit_transform(num_arr)
        else:
            assert scaler is not None, "Scaler wajib diisi untuk val/test set."
            self.scaler  = scaler
            self.num_arr = scaler.transform(num_arr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        html = str(row["html"])

        # Tokenisasi SecBERT via AutoTokenizer
        enc = self.tokenizer(
            html,
            max_length      = MAX_SEQ_LEN,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )

        input_ids      = enc["input_ids"].squeeze(0)       # [MAX_SEQ_LEN]
        attention_mask = enc["attention_mask"].squeeze(0)  # [MAX_SEQ_LEN]

        num_feat = torch.tensor(
            self.num_arr[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return input_ids, attention_mask, num_feat, label


# ============================================================
# ATTENTION POOLING
# ============================================================

class AttentionPooling(nn.Module):
    """
    Soft attention pooling atas seluruh token sequence
    dari SecBERT last_hidden_state.
    Lebih informatif daripada sekadar mengambil [CLS] token.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, attention_mask=None):
        """
        x              : [B, T, hidden_dim]
        attention_mask : [B, T]  (1=valid, 0=padding)
        """
        scores = self.attn(x).squeeze(-1)       # [B, T]

        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask == 0, -1e4
            )

        weights = torch.softmax(scores, dim=1)  # [B, T]
        pooled  = torch.sum(
            x * weights.unsqueeze(-1), dim=1
        )                                        # [B, hidden_dim]

        return pooled


# ============================================================
# MODEL
# ============================================================

class PhishingSecBERT(nn.Module):
    """
    Arsitektur:
        HTML
          → AutoTokenizer (SecBERT)
          → AutoModel / SecBERT (BERT-base, 12 layer)
            · Layer 0–9   : Frozen
            · Layer 10–11 : Trainable (UNFREEZE_LAST_N_LAYERS=2)
          → last_hidden_state [B, T, 768]
          → AttentionPooling  [B, 768]
                                              ↓
        Numerical [17]                        ↓
          → Linear(17→128)+LN+GELU+Drop       ↓
          → Linear(128→64)+GELU   [B, 64]     ↓
                                    └── concat → [B, 832]
                                              ↓
                              Linear(832→256) + LN + GELU + Drop
                                              ↓
                              Linear(256→128) + LN + GELU + Drop
                                              ↓
                              Linear(128→2)

    Catatan arsitektur SecBERT vs DistilBERT:
        - SecBERT  : 12 layer, akses via bert.encoder.layer[i]
        - DistilBERT: 6 layer, akses via bert.transformer.layer[i]
    """

    def __init__(self, num_dim: int = len(NUM_FEAT_COLS)):
        super().__init__()

        # ── SecBERT Backbone ──────────────────────────────────
        self.bert = AutoModel.from_pretrained(PRETRAINED_MODEL)

        # Freeze seluruh parameter SecBERT terlebih dahulu
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze N layer terakhir dari encoder
        # SecBERT (BERT-base) → 12 layer via self.bert.encoder.layer
        total_layers  = len(self.bert.encoder.layer)          # = 12
        unfreeze_from = total_layers - UNFREEZE_LAST_N_LAYERS  # = 10

        for i in range(unfreeze_from, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze pooler juga (representasi [CLS] token)
        if hasattr(self.bert, "pooler") and self.bert.pooler is not None:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True

        # ── Attention Pooling ─────────────────────────────────
        self.pooling = AttentionPooling(BERT_DIM)

        # ── Numerical Features Projector ──────────────────────
        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 64),
            nn.GELU()
        )

        # ── Fusion Classifier ─────────────────────────────────
        # BERT_DIM(768) + num_proj_out(64) = 832
        self.classifier = nn.Sequential(

            nn.Linear(BERT_DIM + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(DROPOUT / 2),

            nn.Linear(128, 2)
        )

        self._init_new_weights()

    def _init_new_weights(self):
        """Inisialisasi hanya layer baru (bukan SecBERT backbone)."""
        for m in [self.pooling, self.num_proj, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, input_ids, attention_mask, num_feats):
        # SecBERT encoding
        bert_out = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        # last_hidden_state : [B, T, 768]
        hidden = bert_out.last_hidden_state

        # Attention Pooling → [B, 768]
        html_vec = self.pooling(hidden, attention_mask)

        # Numerical features → [B, 64]
        num_vec  = self.num_proj(num_feats)

        # Concat → [B, 832]
        fused  = torch.cat([html_vec, num_vec], dim=-1)

        # Classifier → [B, 2]
        logits = self.classifier(fused)

        return logits


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion, amp_scaler, scheduler):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []

    optimizer.zero_grad()

    for step, (input_ids, attn_mask, num_feats, labels) in enumerate(loader):
        input_ids  = input_ids.to(DEVICE)
        attn_mask  = attn_mask.to(DEVICE)
        num_feats  = num_feats.to(DEVICE)
        labels     = labels.to(DEVICE)

        with autocast():
            logits = model(input_ids, attn_mask, num_feats)
            loss   = criterion(logits, labels) / GRAD_ACCUM

        amp_scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    return total_loss / len(loader), acc


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds_all, probs_all, labels_all = [], [], []

    for input_ids, attn_mask, num_feats, labels in loader:
        input_ids  = input_ids.to(DEVICE)
        attn_mask  = attn_mask.to(DEVICE)
        num_feats  = num_feats.to(DEVICE)
        labels     = labels.to(DEVICE)

        logits = model(input_ids, attn_mask, num_feats)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item()
        preds_all.extend(logits.argmax(dim=1).cpu().numpy())
        probs_all.extend(probs.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(labels_all,  preds_all)
    prec     = precision_score(labels_all, preds_all, zero_division=0)
    rec      = recall_score(labels_all,    preds_all, zero_division=0)
    f1       = f1_score(labels_all,        preds_all, zero_division=0)
    auc      = roc_auc_score(labels_all,   probs_all)

    return avg_loss, acc, prec, rec, f1, auc, preds_all, probs_all, labels_all


# ============================================================
# PLOTTING
# ============================================================

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss",      marker="o", markersize=3)
    plt.plot(val_losses,   label="Validation Loss", marker="x", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss — SecBERT")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve_secbert.png", dpi=150)
    plt.close()
    print("✓ loss_curve_secbert.png")


def plot_evaluation(labels, preds, probs):
    auc = roc_auc_score(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    cm  = confusion_matrix(labels, preds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"]
    )
    axes[0].set_title("Confusion Matrix — SecBERT")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    axes[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — SecBERT")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("evaluation_secbert.png", dpi=150)
    plt.close()
    print("✓ evaluation_secbert.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("  PHISHING DETECTION — SecBERT + NUMERICAL FEATURES")
    print("=" * 65)

    # ── Load Dataset ─────────────────────────────────────────────
    print("\n[1/6] Loading dataset ...")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    print(f"     Train : {len(train_df):,}")
    print(f"     Val   : {len(val_df):,}")
    print(f"     Test  : {len(test_df):,}")

    counts = train_df["label"].value_counts().sort_index().values
    print(f"     Label dist (train) : {dict(enumerate(counts.tolist()))}")

    # ── Tokenizer ─────────────────────────────────────────────────
    print("\n[2/6] Loading SecBERT tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    print(f"     Model    : {PRETRAINED_MODEL}")
    print(f"     Vocab size : {tokenizer.vocab_size:,}")

    # ── Dataset & DataLoader ──────────────────────────────────────
    print("[3/6] Building datasets & dataloaders ...")

    train_ds = PhishingDataset(train_df, tokenizer, fit_scaler=True)
    val_ds   = PhishingDataset(val_df,   tokenizer, scaler=train_ds.scaler)
    test_ds  = PhishingDataset(test_df,  tokenizer, scaler=train_ds.scaler)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────
    print("[4/6] Building model ...")
    model = PhishingSecBERT(num_dim=len(NUM_FEAT_COLS)).to(DEVICE)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"     Total params     : {total_p:,}")
    print(f"     Trainable params : {trainable_p:,}")
    print(f"     Frozen  params   : {total_p - trainable_p:,}")
    print(f"     Unfreeze layer   : {12 - UNFREEZE_LAST_N_LAYERS}–11 "
          f"({UNFREEZE_LAST_N_LAYERS} layer terakhir)")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────
    print("[5/6] Setting up training config ...")

    class_weights = torch.tensor(
        counts.sum() / (2.0 * counts),
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    steps_per_epoch  = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps      = steps_per_epoch * NUM_EPOCHS
    warmup_steps     = total_steps // 10

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )

    amp_scaler = GradScaler()

    # ── Training Loop ─────────────────────────────────────────────
    print(f"\n[6/6] Training {NUM_EPOCHS} epochs ...")
    print("─" * 80)
    print(f"{'Ep':>3} {'TrLoss':>8} {'TrAcc':>7} "
          f"{'VlLoss':>8} {'VlAcc':>7} {'VlF1':>7} {'VlAUC':>7}  {'Time':>6}")
    print("─" * 80)

    best_auc     = 0.0
    # patience_ctr = 0
    train_losses = []
    val_losses   = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, amp_scaler, scheduler
        )

        vl_loss, vl_acc, vl_prec, vl_rec, vl_f1, vl_auc, _, _, _ = evaluate(
            model, val_loader, criterion
        )

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        elapsed = time.time() - t0
        flag    = ""

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), MODEL_PATH)
            flag = "  ◄ best"
            # patience_ctr = 0
        # else:
        #     patience_ctr += 1

        print(
            f"{epoch:>3} {tr_loss:>8.4f} {tr_acc:>7.4f} "
            f"{vl_loss:>8.4f} {vl_acc:>7.4f} {vl_f1:>7.4f} "
            f"{vl_auc:>7.4f}  {elapsed:>5.1f}s{flag}"
        )

        # if patience_ctr >= PATIENCE:
        #     print(f"\n  ⚠ Early stopping di epoch {epoch}.")
        #     break

    plot_loss(train_losses, val_losses)

    # ── Test Evaluation ───────────────────────────────────────────
    print(f"\n  Best Val AUC : {best_auc:.4f}")
    print("\n  Loading best model → Test Set ...")

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    te_loss, te_acc, te_prec, te_rec, te_f1, te_auc, \
        te_preds, te_probs, te_labels = evaluate(
            model, test_loader, criterion
        )

    print("\n" + "=" * 65)
    print("  TEST RESULTS")
    print("=" * 65)
    print(f"  Loss      : {te_loss:.4f}")
    print(f"  Accuracy  : {te_acc:.4f}")
    print(f"  Precision : {te_prec:.4f}")
    print(f"  Recall    : {te_rec:.4f}")
    print(f"  F1 Score  : {te_f1:.4f}")
    print(f"  AUC ROC   : {te_auc:.4f}")

    print("\n  Classification Report:")
    print(classification_report(
        te_labels, te_preds,
        target_names=["Legitimate", "Phishing"],
        digits=4
    ))

    print("  Confusion Matrix:")
    cm = confusion_matrix(te_labels, te_preds)
    print(f"  {'':16} Pred-Legit  Pred-Phish")
    print(f"  {'True-Legit':16} {cm[0,0]:>10}  {cm[0,1]:>10}")
    print(f"  {'True-Phish':16} {cm[1,0]:>10}  {cm[1,1]:>10}")

    plot_evaluation(te_labels, te_preds, te_probs)

    print("\n" + "=" * 65)
    print("  OUTPUT FILES")
    print("=" * 65)
    print(f"  ✓ Model            : {MODEL_PATH}")
    print(f"  ✓ Loss Curve       : loss_curve_secbert.png")
    print(f"  ✓ Evaluation Plot  : evaluation_secbert.png")
    print("\n  SELESAI.\n")


# ============================================================

if __name__ == "__main__":
    main()
