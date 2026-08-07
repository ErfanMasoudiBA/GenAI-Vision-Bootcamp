from models.model import (
    CustomCNN,
    get_pretrained_resnet18,
    get_scratch_resnet18,
    count_parameters,
)

print("--- Custom CNN ---")
model1 = CustomCNN()
total, train, frozen = count_parameters(model1)
print(f"Total: {total:,} | Trainable: {train:,} | Frozen: {frozen:,}\n")

print("--- Pretrained ResNet18 (Transfer Learning) ---")
model2 = get_pretrained_resnet18()
total, train, frozen = count_parameters(model2)
print(f"Total: {total:,} | Trainable: {train:,} | Frozen: {frozen:,}\n")

print("--- Scratch ResNet18 ---")
model3 = get_scratch_resnet18()
total, train, frozen = count_parameters(model3)
print(f"Total: {total:,} | Trainable: {train:,} | Frozen: {frozen:,}\n")
