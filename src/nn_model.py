import itertools
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import load_split, scores

torch.manual_seed(42)
SCALE = 1000.0  # scale the target so the loss values stay small

X_train, X_val, X_test, y_train, y_val, y_test = load_split()


def to_tensors(X, y):
    X_t = torch.tensor(X.values.astype("float32"))
    y_t = torch.tensor((y.values / SCALE).astype("float32")).view(-1, 1)
    return X_t, y_t


Xtr, ytr = to_tensors(X_train, y_train)
Xva, yva = to_tensors(X_val, y_val)


class SolarNet(nn.Module):
    """Fully connected network: input -> 64 -> 64 -> 1 (with ReLU and dropout)."""

    def __init__(self, n_in, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(lr, batch_size, dropout, epochs=30):
    model = SolarNet(Xtr.shape[1], dropout=dropout)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_losses.append(total / len(Xtr))
        model.eval()
        with torch.no_grad():
            val_losses.append(loss_fn(model(Xva), yva).item())
    return model, train_losses, val_losses


if __name__ == "__main__":
    rows = []
    best = None
    for lr, bs, dr in itertools.product([0.001, 0.01], [64, 256], [0.0, 0.2]):
        model, tl, vl = train(lr, bs, dr)
        model.eval()
        with torch.no_grad():
            pred = model(Xva).numpy().ravel() * SCALE
        mae, rmse, r2 = scores(y_val, pred)
        row = {"learning_rate": lr, "batch_size": bs, "dropout": dr,
               "val_MAE": round(float(mae), 2), "val_RMSE": round(float(rmse), 2),
               "val_R2": round(float(r2), 4)}
        rows.append(row)
        print(row)
        if best is None or rmse < best["rmse"]:
            best = {"rmse": rmse, "model": model, "tl": tl, "vl": vl, "row": row}

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    pd.DataFrame(rows).sort_values("val_RMSE").to_csv(
        "results/nn_tuning_results.csv", index=False)
    torch.save(best["model"].state_dict(), "models/solar_nn.pt")

    plt.plot(best["tl"], label="Training loss")
    plt.plot(best["vl"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error (scaled)")
    plt.title("Neural network loss curve")
    plt.legend()
    plt.savefig("figures/nn_loss_curve.png", dpi=200, bbox_inches="tight")
    print("Best setting:", best["row"])