import argparse
import torch
from tqdm import tqdm
import torch.optim as optim
import optuna

from data.loader import get_dataloaders
from utils.config import load_config, get_class
from utils.plot_batch_pred_vs_actual import plot_batch_pred_vs_actual

def objective(trial, config_path="config/optuna_default.json", save_dir="states/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)

    # --- Suggest hyperparameters ---
    lr = trial.suggest_loguniform("lr", config["train_params"]["lr"] / 10, config["train_params"]["lr"] * 10)

    # Update config dynamically
    config["train_params"]["lr"] = lr

    ModelClass = get_class("models", config["model"])
    model = ModelClass(**config["model_params"]).to(device)

    LossClass = get_class("losses", config["loss"])
    loss_fn = LossClass(**config["loss_params"])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_epochs = config["train_params"]["num_epochs"]

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=config["train_params"]["batch_size"],
        train_size=0.5,
        val_size=0.1
    )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X, mask, y_tensor in tqdm(train_loader, desc=f"Trial {trial.number}, Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            X, mask, y_tensor = X.to(device), mask.to(device), y_tensor.to(device)

            pred_tensor, _ = model(X, mask, log_diversity=False)
            loss = loss_fn(pred_tensor, y_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_tensor, x_mask, y_tensor in val_loader:
                x_tensor, x_mask, y_tensor = x_tensor.to(device), x_mask.to(device), y_tensor.to(device)
                pred_tensor, _ = model(x_tensor, x_mask)
                loss = loss_fn(pred_tensor, y_tensor)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    # Save best model of this trial
    save_path = f"{save_dir}/best_model_trial_{trial.number}.pt"
    torch.save(best_model_state, save_path)

    return best_val_loss


def run_optuna(config_path="config/optuna_default.json", save_dir="states/", n_trials=20):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, config_path, save_dir), n_trials=n_trials)

    print("\n=== Optimization Complete ===")
    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)

    # Optional: plot from the best trial
    best_model_path = f"{save_dir}/best_model_trial_{study.best_trial.number}.pt"
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = get_class("models", config["model"])
    model = ModelClass(**config["model_params"]).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    _, _, test_loader = get_dataloaders(batch_size=config["train_params"]["batch_size"])
    with torch.no_grad():
        for x_tensor, x_mask, y_tensor in test_loader:
            x_tensor, x_mask, y_tensor = x_tensor.to(device), x_mask.to(device), y_tensor.to(device)
            pred_tensor, _ = model(x_tensor, x_mask)
            plot_batch_pred_vs_actual(pred_tensor, y_tensor, plot_path=f"plots/best_trial_pred_vs_actual_{study.best_trial.number}.png", n=8)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/optuna_default.json")
    parser.add_argument("--save_dir", type=str, default="states/")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    run_optuna(config_path=args.config, save_dir=args.save_dir, n_trials=args.trials)