import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from transformers import ResNetForImageClassification
import os
import random
import numpy as np

# --- 0. Setup and Configuration ---
# Automatically select MPS (for Apple Silicon), CUDA, or CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("MPS (Metal Performance Shaders) is available. Using GPU acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA is available. Using GPU acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("MPS or CUDA not available. Using CPU.")

# Ensure all tensors are float32 for MPS compatibility
torch.set_default_dtype(torch.float32)

# Training Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 8
TEST_SPLIT_SIZE = 0.2

# Data and Model Paths
DATA_BASE_DIR = "data/Animals"
MODEL_SAVE_PATH = "animal_classifier_finetuned.pth"

# --- 1. Data Loading and Preparation ---
print("Loading and preprocessing images...")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def manual_split(samples, labels, test_size=0.2, random_state=42):
    """Manually splits data into training and validation sets."""
    random.seed(random_state)
    combined = list(zip(samples, labels))
    random.shuffle(combined)
    
    split_idx = int(len(combined) * (1 - test_size))
    train_combined = combined[:split_idx]
    val_combined = combined[split_idx:]
    
    train_samples, train_labels = zip(*train_combined) if train_combined else ([], [])
    val_samples, val_labels = zip(*val_combined) if val_combined else ([], [])
    
    return list(train_samples), list(train_labels), list(val_samples), list(val_labels)

# Load data with labels
all_samples = []
all_labels = []
class_to_idx = {"cats": 0, "dogs": 1, "snakes": 2}
idx_to_class = {v: k for k, v in class_to_idx.items()}

data_dirs = {name: os.path.join(DATA_BASE_DIR, name) for name in class_to_idx.keys()}

for class_name, dir_path in data_dirs.items():
    if not os.path.isdir(dir_path):
        print(f"Warning: Directory not found at {dir_path}")
        continue
    class_idx = class_to_idx[class_name]
    for file in os.listdir(dir_path):
        try:
            img_path = os.path.join(dir_path, file)
            # Check for valid image extensions
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                all_samples.append(img_tensor)
                all_labels.append(class_idx)
        except Exception as e:
            print(f"Could not read or process {img_path}: {e}")

if not all_samples:
    raise ValueError("No images were loaded. Please check the data directories and image files.")

# Split data and create DataLoaders
train_samples, train_labels, val_samples, val_labels = manual_split(
    all_samples, all_labels, test_size=TEST_SPLIT_SIZE
)

train_dataset = TensorDataset(torch.stack(train_samples), torch.tensor(train_labels))
val_dataset = TensorDataset(torch.stack(val_samples), torch.tensor(val_labels))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"Dataset created: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

# --- 2. Model Preparation ---
print("Preparing pre-trained ResNet-50 model for fine-tuning...")

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Freeze all parameters in the model first
for param in model.parameters():
    param.requires_grad = False

# Modify and unfreeze the final layer for our 3 classes
num_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=num_features, out_features=len(class_to_idx))
)

# Unfreeze only the parameters of the new classifier for training
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(DEVICE)

# --- 3. Model Fine-Tuning ---
criterion = torch.nn.CrossEntropyLoss()
# Create an optimizer that only updates the unfrozen classifier parameters
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

print("\n--- Starting Fine-Tuning ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.logits.max(1)[1]
            val_corrects += torch.sum(preds == labels.data)
    
    val_acc = val_corrects.float() / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

print("--- Fine-Tuning Finished ---")

# --- 4. Save the Fine-Tuned Model ---
# We save the state_dict, which is the recommended way.
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel fine-tuned and saved to {MODEL_SAVE_PATH}")

# --- 5. Final Evaluation (Optional) ---
print("\n--- Final Evaluation on Validation Set ---")
model.eval()
total_correct = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        preds = outputs.logits.max(1)[1]
        total_correct += torch.sum(preds == labels.data)
        total_samples += len(labels)

final_accuracy = total_correct.float() / total_samples
print(f"Final accuracy of the fine-tuned model on the validation set: {final_accuracy:.2%}")
