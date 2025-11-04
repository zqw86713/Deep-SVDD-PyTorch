#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep SVDD with PyOD on MNIST
- Trains only on one "normal" digit (default: 1)
- Treats all other digits as anomalies at test time
- Reports AUROC/AUPRC/F1 and saves a ROC curve
Requirements:
  pip install pyod torch torchvision scikit-learn matplotlib numpy
"""
import os
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, roc_curve
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# PyOD
from pyod.models.deep_svdd import DeepSVDD

def load_mnist(normal_class=1, limit_train=None, limit_test=None, seed=42):
    """
    Load MNIST and build:
      - X_train: only the chosen normal_class (as inliers)
      - X_test: mixture of normal_class (inliers) and others (anomalies)
    Returns X_train, X_test, y_test (1=anomaly, 0=inlier)
    """
    transform = transforms.Compose([transforms.ToTensor()])  # values in [0,1]
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Build train set: only normal_class
    train_imgs = []
    for img, label in train_set:
        if int(label) == int(normal_class):
            train_imgs.append(img.numpy().astype(np.float32))  # (1, 28, 28)

    X_train = np.stack(train_imgs, axis=0)  # (N, 1, 28, 28)
    X_train = X_train.reshape(X_train.shape[0], -1)  # flatten to (N, 784)

    # Optionally subsample for faster experimentation
    if limit_train is not None:
        np.random.seed(seed)
        idx = np.random.choice(len(X_train), size=min(limit_train, len(X_train)), replace=False)
        X_train = X_train[idx]

    # Build test set: mix of inliers (normal_class) and anomalies (others)
    X_test, y_test = [], []
    for img, label in test_set:
        img_np = img.numpy().astype(np.float32).reshape(-1)  # (784,)
        is_anomaly = int(label) != int(normal_class)
        X_test.append(img_np)
        y_test.append(1 if is_anomaly else 0)
    X_test = np.stack(X_test, axis=0)
    y_test = np.array(y_test, dtype=np.int64)

    if limit_test is not None:
        # Keep class balance approximately by shuffling then slicing
        X_test, y_test = shuffle(X_test, y_test, random_state=seed)
        X_test = X_test[:limit_test]
        y_test = y_test[:limit_test]

    return X_train, X_test, y_test

def train_and_eval(normal_class=1, contamination=0.1, representation_size=32, hidden_neurons=(128, 64),
                   epochs=30, batch_size=256, lr=1e-3, limit_train=None, limit_test=None, seed=42,
                   device=None, outdir="./outputs"):
    os.makedirs(outdir, exist_ok=True)
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Config] normal_class={normal_class}, contamination={contamination}, epochs={epochs}, batch_size={batch_size}, device={device}")
    print(f"[Config] hidden_neurons={hidden_neurons}, representation_size={representation_size}, lr={lr}")

    # Data
    X_train, X_test, y_test = load_mnist(normal_class=normal_class, limit_train=limit_train, limit_test=limit_test, seed=seed)
    print(f"[Data] X_train: {X_train.shape}, X_test: {X_test.shape}, positives(anomalies) in test: {y_test.sum()} / {len(y_test)}")

    # ===== Initialize DeepSVDD (PyOD version requiring n_features) =====
    # X_train has shape (N, 784) because MNIST images are flattened from 28×28
    n_features = X_train.shape[1]

    # You may turn preprocessing on/off:
    # - True  : applies StandardScaler inside PyOD
    # - False : skip it (recommended here, since MNIST pixels are already in [0, 1])
    use_preproc = False

    model = DeepSVDD(
        n_features=n_features,  # number of input features
        c=None,  # let the model learn the center automatically
        use_ae=False,  # False = pure DeepSVDD; True = with autoencoder
        hidden_neurons=list(hidden_neurons),  # e.g., [128, 64]
        hidden_activation='relu',  # activation for hidden layers
        output_activation='sigmoid',  # activation for output layer
        optimizer='adam',  # optimizer type
        epochs=epochs,  # training epochs
        batch_size=batch_size,  # mini-batch size
        dropout_rate=0.0,  # set to 0.1–0.2 if you want regularization
        l2_regularizer=0.0,  # weight decay (try 1e-4 or 1e-3 later)
        validation_size=0.1,  # fraction of training data for validation
        preprocessing=use_preproc,  # whether to apply internal preprocessing
        verbose=1,  # print training progress
        random_state=seed,  # reproducibility
        contamination=contamination  # expected fraction of anomalies in test data
    )

    # Train only on inliers
    model.fit(X_train)

    # Scoring and prediction
    scores = model.decision_function(X_test)  # higher = more abnormal
    y_pred = model.predict(X_test)            # 1=outlier, 0=inlier (based on internal threshold)

    # Metrics
    auroc = roc_auc_score(y_test, scores)
    auprc = average_precision_score(y_test, scores)
    f1 = f1_score(y_test, y_pred)
    precision, recall, f1_d, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print("\n[Results]")
    print(f"AUROC:  {auroc:.4f}")
    print(f"AUPRC:  {auprc:.4f}")
    print(f"F1(bin): {f1:.4f}  (precision={precision:.4f}, recall={recall:.4f})")

    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"DeepSVDD (AUROC={auroc:.3f})")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC — MNIST (normal={normal_class})")
    plt.legend()
    roc_path = os.path.join(outdir, f"roc_mnist_deepsvdd_normal{normal_class}.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    print(f"[Saved] ROC curve -> {roc_path}")

    # Also save scores/preds
    np.save(os.path.join(outdir, "scores.npy"), scores)
    np.save(os.path.join(outdir, "y_test.npy"), y_test)
    np.save(os.path.join(outdir, "y_pred.npy"), y_pred)
    print(f"[Saved] scores.npy, y_test.npy, y_pred.npy in {outdir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal_class", type=int, default=1, help="Digit [0-9] considered 'normal' during training")
    parser.add_argument("--contamination", type=float, default=0.1, help="Estimated contamination ratio in test")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--representation_size", type=int, default=32, help="Dimensionality of hypersphere embedding")
    parser.add_argument("--hidden_neurons", type=str, default="128,64", help="Comma-separated hidden layer sizes")
    parser.add_argument("--limit_train", type=int, default=None, help="Optional: cap number of training samples")
    parser.add_argument("--limit_test", type=int, default=None, help="Optional: cap number of test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="'cpu' or 'cuda' (auto if None)")
    parser.add_argument("--outdir", type=str, default="./outputs")
    args = parser.parse_args()

    hidden = tuple(int(x) for x in args.hidden_neurons.split(",")) if args.hidden_neurons else (128, 64)

    train_and_eval(
        normal_class=args.normal_class,
        contamination=args.contamination,
        representation_size=args.representation_size,
        hidden_neurons=hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        seed=args.seed,
        device=args.device,
        outdir=args.outdir
    )

if __name__ == "__main__":
    main()
