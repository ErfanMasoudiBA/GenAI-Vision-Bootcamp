import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data.data_loader import get_dataloaders
from models.model import get_pretrained_resnet18


def evaluate_model():
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load test data
    _, _, test_loader = get_dataloaders(
        config["data_path"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
    )

    # Initialize model and load the best weights
    model = get_pretrained_resnet18(num_classes=config["num_classes"])
    model.load_state_dict(
        torch.load("saved_models/best_tl_model.pth", map_location=device)
    )
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Standard FER2013 classes
    classes = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    print("\n--- Classification Report ---")
    print(
        classification_report(
            all_labels, all_preds, target_names=classes, zero_division=0
        )
    )

    # Generate and save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title("Confusion Matrix on Test Set")
    plt.ylabel("Actual Emotion")
    plt.xlabel("Predicted Emotion")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    evaluate_model()
