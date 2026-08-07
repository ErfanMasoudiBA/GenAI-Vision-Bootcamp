import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from data.data_loader import get_dataloaders
from models.model import get_finetuned_resnet18, get_pretrained_resnet18
from scripts.train import train_model
from utils.visualization import plot_training_curves


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        config["data_path"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    print("Initializing Fine-Tuned Model (Extra Credit)...")
    model = get_finetuned_resnet18(num_classes=config["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        trainable_params,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print("Starting training...")
    model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=config["epochs"],
        device=device,
        save_path="saved_models/best_tl_model.pth",
    )

    plot_training_curves(
        history, "Transfer Learning (ResNet18)", "tl_training_curves.png"
    )


if __name__ == "__main__":
    main()
