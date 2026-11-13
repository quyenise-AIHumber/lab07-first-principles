
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from model import BikeNet
from prepare import make_loaders, prepare_data


# 1. Run one epoch: train or validate
def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_absolute_error = 0.0
    count = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:

            # Forward pass
            predictions = model(x)

            # Check prediction and target dimensions
            assert predictions.shape == y.shape

            # Calculate loss
            loss = loss_fn(predictions, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Accumulate metrics over all observations
            n = len(y)
            total_loss += loss.item() * n
            total_absolute_error += (
                (predictions - y).abs().sum().item()
            )
            count += n

    mean_loss = total_loss / count
    mean_mae = total_absolute_error / count

    return mean_loss, mean_mae


# 2. Prove the network can learn one small batch
def check_small_batch(dataset, n_features):
    torch.manual_seed(123)

    model = BikeNet(n_features, hidden=64)
    x = dataset.x[:8]
    y = dataset.y[:8]

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01
    )

    with torch.no_grad():
        initial_loss = loss_fn(model(x), y).item()

    for _ in range(600):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = loss_fn(model(x), y).item()

    print(
        f"Small-batch loss: "
        f"{initial_loss:.4f} -> {final_loss:.4f}"
    )

    assert final_loss < initial_loss * 0.1, (
        "The network failed to learn the small batch."
    )


# 3. Train for multiple epochs
def train_model(
    model, train_loader, val_loader,
    epochs=80, learning_rate=0.001
):
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    best_loss = float("inf")
    best_epoch = None
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):

        train_loss, train_mae = run_epoch(
            model, train_loader, loss_fn, optimizer
        )

        val_loss, val_mae = run_epoch(
            model, val_loader, loss_fn
        )

        # Record every epoch
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_mae,
            "val_loss": val_loss,
            "val_mae": val_mae,
        })

        # Keep the weights from the best validation epoch
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss {train_loss:.3f} | "
                f"Val Loss {val_loss:.3f} | "
                f"Val MAE {val_mae:.3f}"
            )

    if best_state is None:
        raise ValueError("Train for at least one epoch.")

    # Restore the best model, not the last one
    model.load_state_dict(best_state)

    return pd.DataFrame(history), best_epoch


# 4. Run the project
def main():
    data = prepare_data(seed=42)

    train_loader, val_loader, _ = make_loaders(
        data, seed=42, batch_size=64
    )

    n_features = data["train"].x.shape[1]

    # Verify the model output shape
    torch.manual_seed(42)
    model = BikeNet(n_features, hidden=32)

    sample_x = data["train"].x[:64]
    sample_y = data["train"].y[:64]

    print("Input shape:", sample_x.shape)
    print("Prediction shape:", model(sample_x).shape)
    print("Target shape:", sample_y.shape)

    # A working network should memorize a tiny batch
    check_small_batch(data["train"], n_features)

    # Reset seed before the main training run
    torch.manual_seed(42)
    model = BikeNet(n_features, hidden=32)

    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=80,
        learning_rate=0.001,
    )

    # Save history for Part 4
    Path("reports").mkdir(exist_ok=True)
    history.to_csv("reports/history.csv", index=False)

    best_row = history.loc[
        history["epoch"] == best_epoch
    ].iloc[0]

    print(f"\nBest epoch: {best_epoch}")
    print(
        f"Best-epoch validation MAE: "
        f"{best_row['val_mae']:.2f} bikes"
    )
    print("Saved: reports/history.csv")


if __name__ == "__main__":
    main()
