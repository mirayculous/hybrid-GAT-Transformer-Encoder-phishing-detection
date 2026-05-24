"""
========================================================================
 PHISHING DETECTION — OPTIMIZED TRANSFORMER ENCODER
 Dataset  :
    - train.csv
    - val.csv
    - test.csv

 Arsitektur:
    HTML Text
      → Custom Tokenizer
      → Embedding
      → Positional Encoding
      → Transformer Encoder
      → Attention Pooling

    Numerical Features
      → Feature Projection

    Fusion
      → MLP Classifier

 Optimizations:
    ✓ Mixed Precision Training (GPU)
    ✓ Gradient Accumulation
    ✓ AdamW + Cosine Scheduler
    ✓ Label Smoothing
    ✓ Weighted Loss
    ✓ Attention Pooling
    ✓ Better Tokenizer
    ✓ Early Stopping
    ✓ ROC Curve
    ✓ Loss Curve
    ✓ Complete Evaluation Metrics
========================================================================
"""

import os
import re
import math
import time
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

print(f"\n[INFO] Device : {DEVICE}")

# ============================================================
# CONFIG
# ============================================================

TRAIN_CSV = "train.csv"
VAL_CSV   = "val.csv"
TEST_CSV  = "test.csv"

MAX_SEQ_LEN = 256
VOCAB_SIZE  = 12000

EMBED_DIM   = 128
D_MODEL     = 256

NUM_HEADS   = 8
NUM_LAYERS  = 4
DIM_FF      = 512

DROPOUT     = 0.2

BATCH_SIZE  = 32
NUM_EPOCHS  = 50

LR          = 2e-4
WEIGHT_DECAY = 1e-2

PATIENCE    = 7

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2

MODEL_PATH = "best_transformer_html.pt"

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
# TOKENIZER
# ============================================================

class HTMLTokenizer:

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size

        self.word2id = {
            "<PAD>": PAD_ID,
            "<UNK>": UNK_ID,
            "<CLS>": CLS_ID
        }

    @staticmethod
    def tokenize(text):

        text = str(text).lower()

        tokens = re.findall(
            r"<[^>]+>|https?://\S+|[a-z0-9]+|[^\w\s]",
            text
        )

        return tokens

    def build_vocab(self, texts):

        counter = Counter()

        for text in texts:
            counter.update(self.tokenize(text))

        for token, _ in counter.most_common(self.vocab_size - 3):
            self.word2id[token] = len(self.word2id)

        print(f"[INFO] Vocabulary Size : {len(self.word2id):,}")

    def encode(self, text):

        tokens = self.tokenize(text)

        ids = [CLS_ID]

        ids.extend([
            self.word2id.get(tok, UNK_ID)
            for tok in tokens
        ])

        ids = ids[:MAX_SEQ_LEN]

        pad_len = MAX_SEQ_LEN - len(ids)

        ids += [PAD_ID] * pad_len

        return ids

# ============================================================
# DATASET
# ============================================================

class PhishingDataset(Dataset):

    def __init__(
        self,
        df,
        tokenizer,
        scaler=None,
        fit_scaler=False
    ):

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

        num_arr = (
            df[NUM_FEAT_COLS]
            .fillna(0)
            .values
            .astype(np.float32)
        )

        if fit_scaler:
            self.scaler = StandardScaler()
            self.num_arr = self.scaler.fit_transform(num_arr)

        else:
            self.scaler = scaler
            self.num_arr = scaler.transform(num_arr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        ids = torch.tensor(
            self.tokenizer.encode(row["html"]),
            dtype=torch.long
        )

        pad_mask = (ids == PAD_ID)

        num_feat = torch.tensor(
            self.num_arr[idx],
            dtype=torch.float32
        )

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return ids, pad_mask, num_feat, label

# ============================================================
# ATTENTION POOLING
# ============================================================

class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):

        scores = self.attn(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e4)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1
        )

        return pooled

# ============================================================
# MODEL
# ============================================================

class PhishingTransformer(nn.Module):

    def __init__(self, vocab_size, num_dim):

        super().__init__()

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            vocab_size,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        self.embed_proj = nn.Linear(
            EMBED_DIM,
            D_MODEL
        )

        # ----------------------------------------------------
        # Positional Encoding
        # ----------------------------------------------------

        pe = torch.zeros(MAX_SEQ_LEN, D_MODEL)

        pos = torch.arange(0, MAX_SEQ_LEN).unsqueeze(1).float()

        div = torch.exp(
            torch.arange(0, D_MODEL, 2).float()
            * (-math.log(10000.0) / D_MODEL)
        )

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

        self.dropout = nn.Dropout(DROPOUT)

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
            norm=nn.LayerNorm(D_MODEL)
        )

        # ----------------------------------------------------
        # Attention Pooling
        # ----------------------------------------------------

        self.pooling = AttentionPooling(D_MODEL)

        # ----------------------------------------------------
        # Numerical Features
        # ----------------------------------------------------

        self.num_proj = nn.Sequential(
            nn.Linear(num_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.GELU()
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(D_MODEL + 64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 2)
        )

        self._init_weights()

    # --------------------------------------------------------

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Embedding):

                nn.init.normal_(m.weight, mean=0, std=0.02)

    # --------------------------------------------------------

    def forward(self, ids, pad_mask, num_feats):

        # Embedding

        x = self.embedding(ids)

        x = self.embed_proj(x)

        x = x + self.pe[:, :x.size(1)]

        x = self.dropout(x)

        # Transformer

        x = self.transformer(
            x,
            src_key_padding_mask=pad_mask
        )

        # Attention Pooling

        html_vec = self.pooling(x, pad_mask)

        # Numerical Features

        num_vec = self.num_proj(num_feats)

        # Fusion

        fused = torch.cat(
            [html_vec, num_vec],
            dim=-1
        )

        logits = self.classifier(fused)

        return logits

# ============================================================
# TRAIN
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()

    total_loss = 0
    preds_all = []
    labels_all = []

    for ids, pm, nf, lb in loader:

        ids = ids.to(DEVICE)
        pm  = pm.to(DEVICE)
        nf  = nf.to(DEVICE)
        lb  = lb.to(DEVICE)

        optimizer.zero_grad()

        with autocast():

            logits = model(ids, pm, nf)

            loss = criterion(logits, lb)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(lb.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)

    return total_loss / len(loader), acc

# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0

    preds_all  = []
    probs_all  = []
    labels_all = []

    for ids, pm, nf, lb in loader:

        ids = ids.to(DEVICE)
        pm  = pm.to(DEVICE)
        nf  = nf.to(DEVICE)
        lb  = lb.to(DEVICE)

        logits = model(ids, pm, nf)

        loss = criterion(logits, lb)

        probs = torch.softmax(logits, dim=1)[:, 1]

        preds = logits.argmax(dim=1)

        total_loss += loss.item()

        preds_all.extend(preds.cpu().numpy())
        probs_all.extend(probs.cpu().numpy())
        labels_all.extend(lb.cpu().numpy())

    acc  = accuracy_score(labels_all, preds_all)
    prec = precision_score(labels_all, preds_all)
    rec  = recall_score(labels_all, preds_all)
    f1   = f1_score(labels_all, preds_all)
    auc  = roc_auc_score(labels_all, probs_all)

    return (
        total_loss / len(loader),
        acc,
        prec,
        rec,
        f1,
        auc,
        preds_all,
        probs_all,
        labels_all
    )

# ============================================================
# PLOT
# ============================================================

def plot_loss(train_losses, val_losses):

    plt.figure(figsize=(8,5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training & Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig("loss_curve.png")

    plt.close()

# ============================================================

def plot_roc(labels, probs):

    fpr, tpr, _ = roc_curve(labels, probs)

    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(7,6))

    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")

    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig("roc_curve.png")

    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================================")
    print(" LOADING DATASET ")
    print("===================================================")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    print(f"Train : {len(train_df):,}")
    print(f"Val   : {len(val_df):,}")
    print(f"Test  : {len(test_df):,}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\n[INFO] Building Vocabulary...")

    tokenizer = HTMLTokenizer()

    tokenizer.build_vocab(train_df["html"])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("[INFO] Building Dataset...")

    train_ds = PhishingDataset(
        train_df,
        tokenizer,
        fit_scaler=True
    )

    val_ds = PhishingDataset(
        val_df,
        tokenizer,
        scaler=train_ds.scaler
    )

    test_ds = PhishingDataset(
        test_df,
        tokenizer,
        scaler=train_ds.scaler
    )

    # --------------------------------------------------------
    # Dataloader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\n[INFO] Building Model...")

    model = PhishingTransformer(
        vocab_size=len(tokenizer.word2id),
        num_dim=len(NUM_FEAT_COLS)
    ).to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"[INFO] Trainable Params : {total_params:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    counts = train_df["label"].value_counts().sort_index().values

    class_weights = torch.tensor(
        counts.sum() / (2.0 * counts),
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    scaler = GradScaler()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n===================================================")
    print(" TRAINING ")
    print("===================================================")

    best_auc = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):

        start = time.time()

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )

        (
            val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            val_auc,
            _,
            _,
            _
        ) = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f} | "
            f"Val AUC {val_auc:.4f} | "
            f"{elapsed:.1f}s"
        )

    # Save best model only

    if val_auc > best_auc:

        best_auc = val_auc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

    print("   ✓ Best model saved")

    # ========================================================
    # TEST
    # ========================================================

    print("\n===================================================")
    print(" TEST EVALUATION ")
    print("===================================================")

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    (
        test_loss,
        test_acc,
        test_prec,
        test_rec,
        test_f1,
        test_auc,
        preds,
        probs,
        labels
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    print(f"\nLoss      : {test_loss:.4f}")
    print(f"Accuracy  : {test_acc:.4f}")
    print(f"Precision : {test_prec:.4f}")
    print(f"Recall    : {test_rec:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"AUC ROC   : {test_auc:.4f}")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n===================================================")
    print(" CLASSIFICATION REPORT ")
    print("===================================================")

    print(
        classification_report(
            labels,
            preds,
            target_names=[
                "Legitimate",
                "Phishing"
            ],
            digits=4
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(labels, preds)

    print("\n===================================================")
    print(" CONFUSION MATRIX ")
    print("===================================================")

    print(f"{'':15} Pred-Legit Pred-Phish")
    print(f"{'True-Legit':15} {cm[0,0]:10} {cm[0,1]:10}")
    print(f"{'True-Phish':15} {cm[1,0]:10} {cm[1,1]:10}")

    # ========================================================
    # PLOT
    # ========================================================

    plot_loss(train_losses, val_losses)

    plot_roc(labels, probs)

    print("\n===================================================")
    print(" FILE OUTPUT ")
    print("===================================================")

    print(f"✓ Model Saved     : {MODEL_PATH}")
    print(f"✓ Loss Curve      : loss_curve.png")
    print(f"✓ ROC Curve       : roc_curve.png")

    print("\nDONE.\n")

# ============================================================

if __name__ == "__main__":
    main()
