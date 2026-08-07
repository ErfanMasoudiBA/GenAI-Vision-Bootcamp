import matplotlib.pyplot as plt
import numpy as np
import torchvision


def plot_training_curves(history, experiment_name, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 6))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Training Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Validation Loss", marker="o")
    plt.title(f"{experiment_name}\nLoss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Training Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker="o")
    plt.title(f"{experiment_name}\nAccuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Training curves saved to {save_path}")


def imshow(inp, title=None, save_path=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(10, 5))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# if __name__ == "__main__":
#     from data.data_loader import get_dataloaders

#     train_loader, _, _ = get_dataloaders(
#         "dataset/fer2013.csv", batch_size=8, image_size=224
#     )
#     inputs, classes = next(iter(train_loader))
#     out = torchvision.utils.make_grid(inputs)
#     imshow(
#         out,
#         title="Sample Training Batch with Augmentation",
#         save_path="sample_batch.png",
#     )
