import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from data.data_loader import get_dataloaders
from models.model import CustomCNN
from scripts.train import train_model
from utils.visualization import plot_training_curves


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config["data_path"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
    )
    dataloaders = {"train": train_loader, "val": val_loader}

    print("Initializing Custom CNN (Trained from Scratch)...")
    model = CustomCNN(num_classes=config["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print("Starting training (30 Epochs)...")
    model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=config["epochs"],
        device=device,
        save_path="saved_models/best_custom_cnn.pth",
    )

    plot_training_curves(history, "Custom CNN (Scratch)", "custom_cnn_curves.png")

    print("\nRunning inference on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    print("\n--- Classification Report (Custom CNN) ---")
    print(
        classification_report(
            all_labels, all_preds, target_names=classes, zero_division=0
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix - Custom CNN")
    plt.ylabel("Actual Emotion")
    plt.xlabel("Predicted Emotion")
    plt.tight_layout()
    plt.savefig("custom_cnn_cm.png", dpi=300)
    print("\nConfusion matrix saved to custom_cnn_cm.png")


if __name__ == "__main__":
    main()
