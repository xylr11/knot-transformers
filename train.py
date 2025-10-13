import argparse
from data.loader import get_dataloaders
import torch
from tqdm import tqdm
import torch.optim as optim
from utils.config import load_config
from utils.config import get_class
from utils.plot_batch_pred_vs_actual import plot_batch_pred_vs_actual

def train(config_path="config/default.json", save_path="states/best_model_default.pt"):
    """Train a model using the specified config file and save the best model state
    and plot predictions vs actual values for a batch
     Args:
        config_path (str): Path to the configuration file.
        save_path (str): Path to save the best model state.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Using device:", device)

    config = load_config(config_path)

    ModelClass = get_class("models", config["model"])
    model = ModelClass(**config["model_params"]).to(device)
    print(f"Loaded model: {model.__class__.__name__}, Params: {config['model_params']}")
    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total number of parameters: {total_params}")

    LossClass = get_class("losses", config["loss"])
    loss_fn = LossClass(**config["loss_params"])
    print(f"Loaded loss: {loss_fn.__class__.__name__}, Params: {config['loss_params']}")

    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
    num_epochs = config['train_params']['num_epochs']

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['train_params']['batch_size'],
        train_size=.0050,
        val_size=.0025
        train_size=.5,
        val_size=.01
    )
    
    print(f"Loaded optimizer and dataloaders.")

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, (X, mask, y_tensor) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            optimizer.zero_grad()
            X = X.to(device)
            mask = mask.to(device)
            y_tensor = y_tensor.to(device)
            pred_tensor, stats = model(X, mask, log_diversity=False)
            loss = loss_fn(pred_tensor, y_tensor)
            loss.backward()
            optimizer.step()

            if i % 50 == 0 and stats is not None:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                for s in stats["encoder"]:
                    print(f"  Encoder Layer {s['layer']} - Var(set): {s['var_set']:.4f}, Var(batch): {s['var_batch']:.4f}")
                for s in stats["decoder"]:
                    print(f"  Decoder Layer {s['decoder_layer']} - Var(Q): {s['var_queries']:.4f}, Var(batch): {s['var_batch']:.4f}, Var(out): {s['var_out']:.4f}")

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_tensor, x_mask, y_tensor in val_loader:
                x_tensor = x_tensor.to(device)
                x_mask = x_mask.to(device)
                y_tensor = y_tensor.to(device)

                pred_tensor, _ = model(x_tensor, x_mask)
                pred_tensor, _ = model(x_tensor, x_mask)
                loss = loss_fn(pred_tensor, y_tensor)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    torch.save(best_model_state, save_path)
    with torch.no_grad():
        model.eval()
        for x_tensor, x_mask, y_tensor in train_loader:
            x_tensor = x_tensor.to(device)
            x_mask = x_mask.to(device)
            y_tensor = y_tensor.to(device)

            pred_tensor, _ = model(x_tensor, x_mask)

            plot_batch_pred_vs_actual(pred_tensor, y_tensor, plot_path=plot_path, n=8)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--save', type=str, required=False)
    parser.add_argument('--plot', type=str, required=False)
    args = parser.parse_args()
    train(args.config, args.save, args.plot)
    

