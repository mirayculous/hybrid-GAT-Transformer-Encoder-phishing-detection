"""
============================================================================
 PHISHING URL DETECTION — OPTIMIZED GATv2
============================================================================

Dataset:
    - train.csv
    - val.csv
    - test.csv

Kolom:
    - graph_data
    - label

Optimizations:
    ✓ Residual GATv2
    ✓ BatchNorm
    ✓ LayerNorm
    ✓ Multi-head Attention
    ✓ Global Mean + Max Pooling
    ✓ GELU Activation
    ✓ AdamW
    ✓ CosineAnnealingLR
    ✓ Mixed Precision Training
    ✓ Gradient Clipping
    ✓ Label Smoothing
    ✓ ROC Curve
    ✓ Loss Curve
    ✓ Confusion Matrix
    ✓ Full Classification Report

============================================================================
"""
!pip install torch-geometric

import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast, GradScaler

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATv2Conv,
    global_mean_pool,
    global_max_pool
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

warnings.filterwarnings("ignore")

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"\n[INFO] Device : {DEVICE}")

# ============================================================================
# CONFIG
# ============================================================================

TRAIN_PATH = "train.csv"
VAL_PATH   = "val.csv"
TEST_PATH  = "test.csv"

BATCH_SIZE = 64
EPOCHS     = 50

LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 128
HEADS      = 8
DROPOUT    = 0.3

MODEL_PATH = "best_gatv2_model.pt"

# ============================================================================
# LOAD DATASET
# ============================================================================

def load_dataset(csv_file):

    df = pd.read_csv(csv_file)

    data_list = []

    for _, row in df.iterrows():

        label = int(row['label'])

        graph_dict = json.loads(row['graph_data'])

        x = torch.tensor(
            graph_dict['x'],
            dtype=torch.float
        )

        edge_index = torch.tensor(
            graph_dict['edge_index'],
            dtype=torch.long
        )

        # Handle edge case
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)

        y = torch.tensor([label], dtype=torch.long)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        data_list.append(data)

    return data_list

# ============================================================================
# RESIDUAL BLOCK
# ============================================================================

class ResidualGATBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        heads=4,
        dropout=0.3
    ):

        super().__init__()

        self.gat = GATv2Conv(
            in_dim,
            out_dim,
            heads=heads,
            dropout=dropout,
            concat=True
        )

        self.norm = nn.LayerNorm(out_dim * heads)

        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Linear(
            in_dim,
            out_dim * heads
        )

    def forward(self, x, edge_index):

        identity = self.residual(x)

        out = self.gat(x, edge_index)

        out = self.norm(out)

        out = F.gelu(out)

        out = self.dropout(out)

        out = out + identity

        return out

# ============================================================================
# MODEL
# ============================================================================

class OptimizedGATv2(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_dim=128,
        num_classes=2,
        heads=8,
        dropout=0.3
    ):

        super().__init__()

        # ------------------------------------------------------------
        # GAT BLOCK 1
        # ------------------------------------------------------------

        self.block1 = ResidualGATBlock(
            num_features,
            hidden_dim,
            heads=heads,
            dropout=dropout
        )

        # output = hidden_dim * heads

        dim1 = hidden_dim * heads

        # ------------------------------------------------------------
        # GAT BLOCK 2
        # ------------------------------------------------------------

        self.block2 = ResidualGATBlock(
            dim1,
            hidden_dim,
            heads=4,
            dropout=dropout
        )

        dim2 = hidden_dim * 4

        # ------------------------------------------------------------
        # FINAL GAT
        # ------------------------------------------------------------

        self.gat_final = GATv2Conv(
            dim2,
            hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout
        )

        self.norm_final = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------
        # CLASSIFIER
        # mean pool + max pool
        # ------------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(hidden_dim * 2, 256),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(256, 128),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    # ------------------------------------------------------------------------

    def forward(self, x, edge_index, batch):

        x = self.block1(x, edge_index)

        x = self.block2(x, edge_index)

        x = self.gat_final(x, edge_index)

        x = self.norm_final(x)

        x = F.gelu(x)

        x = self.dropout(x)

        # ------------------------------------------------------------
        # Dual Pooling
        # ------------------------------------------------------------

        mean_pool = global_mean_pool(x, batch)

        max_pool = global_max_pool(x, batch)

        x = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        logits = self.classifier(x)

        return logits

# ============================================================================
# TRAIN FUNCTION
# ============================================================================

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

    for data in loader:

        data = data.to(DEVICE)

        optimizer.zero_grad()

        with autocast():

            logits = model(
                data.x,
                data.edge_index,
                data.batch
            )

            loss = criterion(
                logits,
                data.y
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item() * data.num_graphs

        preds = logits.argmax(dim=1)

        preds_all.extend(
            preds.cpu().numpy()
        )

        labels_all.extend(
            data.y.cpu().numpy()
        )

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(
        labels_all,
        preds_all
    )

    return avg_loss, acc

# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0

    all_preds  = []
    all_probs  = []
    all_labels = []

    for data in loader:

        data = data.to(DEVICE)

        logits = model(
            data.x,
            data.edge_index,
            data.batch
        )

        loss = criterion(
            logits,
            data.y
        )

        probs = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * data.num_graphs

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_labels.extend(
            data.y.cpu().numpy()
        )

    avg_loss = total_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)

    prec = precision_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    rec = recall_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    return (
        avg_loss,
        acc,
        prec,
        rec,
        f1,
        auc,
        all_labels,
        all_preds,
        all_probs
    )

# ============================================================================
# PLOT LOSS
# ============================================================================

def plot_loss(train_losses, val_losses):

    plt.figure(figsize=(8,5))

    plt.plot(
        train_losses,
        label='Training Loss',
        marker='o'
    )

    plt.plot(
        val_losses,
        label='Validation Loss',
        marker='x'
    )

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('Training vs Validation Loss')

    plt.legend()

    plt.grid(True)

    plt.savefig("loss_curve_gatv2.png")

    plt.close()

# ============================================================================
# PLOT ROC
# ============================================================================

def plot_roc(labels, probs):

    fpr, tpr, _ = roc_curve(labels, probs)

    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(7,6))

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f'AUC = {auc:.4f}'
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle='--'
    )

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend()

    plt.grid(True)

    plt.savefig("roc_curve_gatv2.png")

    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    print("\n================================================")
    print(" LOADING DATASET ")
    print("================================================")

    train_data = load_dataset(TRAIN_PATH)

    val_data = load_dataset(VAL_PATH)

    test_data = load_dataset(TEST_PATH)

    print(f"Train : {len(train_data):,}")

    print(f"Val   : {len(val_data):,}")

    print(f"Test  : {len(test_data):,}")

    # ------------------------------------------------------------------------
    # DATALOADER
    # ------------------------------------------------------------------------

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ------------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------------

    model = OptimizedGATv2(
        num_features=21,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        heads=HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\n[INFO] Trainable Params : {total_params:,}")

    # ------------------------------------------------------------------------
    # CLASS WEIGHTS
    # ------------------------------------------------------------------------

    labels = [
        data.y.item()
        for data in train_data
    ]

    counts = np.bincount(labels)

    weights = torch.tensor(
        counts.sum() / (2.0 * counts),
        dtype=torch.float32
    ).to(DEVICE)

    # ------------------------------------------------------------------------
    # LOSS
    # ------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = GradScaler()

    # ------------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------------

    print("\n================================================")
    print(" TRAINING ")
    print("================================================")

    best_auc = 0

    train_losses = []

    val_losses = []

    for epoch in range(1, EPOCHS + 1):

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

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f} | "
            f"Val AUC {val_auc:.4f}"
        )

        # Save Best

        if val_auc > best_auc:

            best_auc = val_auc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("   ✓ Best model saved")

    # ------------------------------------------------------------------------
    # PLOT LOSS
    # ------------------------------------------------------------------------

    plot_loss(
        train_losses,
        val_losses
    )

    # ------------------------------------------------------------------------
    # TESTING
    # ------------------------------------------------------------------------

    print("\n================================================")
    print(" TEST EVALUATION ")
    print("================================================")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    (
        test_loss,
        test_acc,
        test_prec,
        test_rec,
        test_f1,
        test_auc,
        final_labels,
        final_preds,
        final_probs
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

    # ------------------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------------------------

    print("\n================================================")
    print(" CLASSIFICATION REPORT ")
    print("================================================")

    print(
        classification_report(
            final_labels,
            final_preds,
            target_names=[
                "Legitimate",
                "Phishing"
            ],
            digits=4
        )
    )

    # ------------------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------------------

    cm = confusion_matrix(
        final_labels,
        final_preds
    )

    plt.figure(figsize=(12,5))

    # CM

    plt.subplot(1,2,1)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[
            'Legitimate',
            'Phishing'
        ],
        yticklabels=[
            'Legitimate',
            'Phishing'
        ]
    )

    plt.title('Confusion Matrix')

    plt.xlabel('Predicted')

    plt.ylabel('True')

    # ROC

    plt.subplot(1,2,2)

    fpr, tpr, _ = roc_curve(
        final_labels,
        final_probs
    )

    plt.plot(
        fpr,
        tpr,
        lw=2,
        label=f'AUC = {test_auc:.4f}'
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle='--'
    )

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend()

    plt.tight_layout()

    plt.savefig("evaluation_gatv2.png")

    plt.show()

    # ------------------------------------------------------------------------
    # ROC SAVE
    # ------------------------------------------------------------------------

    plot_roc(
        final_labels,
        final_probs
    )

    print("\n================================================")
    print(" OUTPUT FILES ")
    print("================================================")

    print(f"✓ Model            : {MODEL_PATH}")

    print(f"✓ Loss Curve       : loss_curve_gatv2.png")

    print(f"✓ ROC Curve        : roc_curve_gatv2.png")

    print(f"✓ Evaluation Plot  : evaluation_gatv2.png")

    print("\nDONE.\n")

# ============================================================================
# RUN
# ============================================================================

if __name__ == '__main__':
    main()
