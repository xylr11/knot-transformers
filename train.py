import argparse
from data.loader import get_dataloaders
import torch
from tqdm import tqdm
import torch.optim as optim
from utils.config import load_config
from utils.config import get_class

def train(config_path="config/default.json", save_path="states/best_model_default.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available(), "Device count:", torch.cuda.device_count())
    print("Using device:", device)

    config = load_config(config_path)

    ModelClass = get_class("models", config["model"])
    model = ModelClass(**config["model_params"]).to(device)
    print(f"Loaded model: {model.__class__.__name__}, Params: {config['model_params']}")

    LossClass = get_class("losses", config["loss"])
    loss_fn = LossClass(**config["loss_params"])
    print(f"Loaded loss: {loss_fn.__class__.__name__}, Params: {config['loss_params']}")

    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
    num_epochs = config['train_params']['num_epochs']

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['train_params']['batch_size']
    )
    
    print(f"Loaded optimizer and dataloaders.")

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x_tensor, x_mask, y_tensor in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            x_tensor = x_tensor.to(device)
            x_mask = x_mask.to(device)
            y_tensor = y_tensor.to(device)

            pred_tensor = model(x_tensor, x_mask)
            loss = loss_fn(pred_tensor, y_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_tensor, x_mask, y_tensor in val_loader:
                x_tensor = x_tensor.to(device)
                x_mask = x_mask.to(device)
                y_tensor = y_tensor.to(device)

                pred_tensor = model(x_tensor, x_mask)
                loss = loss_fn(pred_tensor, y_tensor)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    torch.save(best_model_state, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False)
    parser.add_argument('--save', type=str, required=False)
    args = parser.parse_args()
    train(args.config, args.save)
