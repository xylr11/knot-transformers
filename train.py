from data.loader import get_dataloaders
import torch
from tqdm import tqdm
import torch.optim as optim
from utils.config import load_config
from utils.config import get_model
from utils.config import get_loss

def train(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config(config_path)

    model = get_model(config['model'], config['model_params'])
    loss_fn = get_loss(config['loss'], config['loss_params'])
    optimizer = optim.Adam(model.parameters(), lr=config['train_params']['lr'])
    num_epochs = config['train_params']['num_epochs']

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config['train_params']['batch_size']
    )

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

    torch.save(best_model_state, "best_model.pt")

if __name__ == "__main__":
    pass