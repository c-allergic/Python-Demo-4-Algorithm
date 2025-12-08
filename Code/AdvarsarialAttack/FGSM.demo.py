# An demo of FGSM attack
from re import S
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# --- IMPORT THE CUSTOM MODEL ---
from classfication_model import DogNet
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')

DEVICE = torch.device("cpu")

# --- 1. Load the Self-Trained Model ---
print("Loading self-trained model...")
model = DogNet(num_classes=3)

# --- FIX: Perform a dummy forward pass to initialize dynamic layers ---
# This ensures that fc1 is created before loading the state dict.
# We use a dummy input of the same size as the training data.
dummy_input = torch.randn(1, 3, 224, 224) # Assuming 224x224 was used for training
with torch.no_grad():
    _ = model(dummy_input)

MODEL_WEIGHTS_PATH = "animal_classifier.pth"
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)
print("Model loaded successfully.")

# The preprocess should match what was used during training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper functions for denormalization/normalization are no longer needed ---
# because the model was trained on normalized data and we will attack in that space.
# We will only need denormalization for visualization.

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return tensor * std[:, None, None] + mean[:, None, None]

# define FGSM attack function
def fgsm_attack(img,epsilon,data_grad):
    # collect element-wise sign of gradient
    sign_data_grad = data_grad.sign()
    # create perturbed image
    perturbed_img = img + epsilon * sign_data_grad
    # clip the image to the valid normalized range
    # The clamp range should be based on the normalization parameters
    # min_val = (0 - mean) / std, max_val = (1 - mean) / std
    min_val = torch.tensor([-2.1179, -2.0357, -1.8044])
    max_val = torch.tensor([2.2489, 2.4286, 2.6400])
    perturbed_img = torch.clamp(perturbed_img, min=min_val[None, :, None, None], max=max_val[None, :, None, None])
    return perturbed_img
    
# single sample attack
def generate_adv_example(model,image,label,epsilon=.03):
    image.requires_grad = True
    
    # forward process
    # The custom model returns raw logits directly
    logits = model(image)
    init_pred = logits.max(1,keepdim=True)[1]
    
    # calculate the loss
    loss = torch.nn.CrossEntropyLoss()(logits,label)
    
    # BP
    model.zero_grad()
    loss.backward()
    
    # collect data gradient
    data_grad = image.grad.data
    
    # call FGSM attack
    perturbed_image = fgsm_attack(image,epsilon,data_grad)
    adv_pred = model(perturbed_image).max(1,keepdim=True)[1]
    
    return perturbed_image, adv_pred, init_pred

# --- 2. Load Data and Filter for Correctly Classified Samples ---
all_samples = []
all_labels = []
class_to_idx = {"cats": 0, "dogs": 1, "snakes": 2}
idx_to_class = {v: k for k, v in class_to_idx.items()}

data_base_dir = "data/Animals"
data_dirs = {"cats": os.path.join(data_base_dir, "cats"),
             "dogs": os.path.join(data_base_dir, "dogs"),
             "snakes": os.path.join(data_base_dir, "snakes")}

print("Loading images...")
# load 10 pictures from each folder
for class_name, dir_path in data_dirs.items():
    class_idx = class_to_idx[class_name]
    # random select 10 pictures from each folder
    import random
    files = random.sample(os.listdir(dir_path), 10)
    for file in files:
        try:
            img_path = os.path.join(dir_path, file)
            img = Image.open(img_path).convert("RGB") # Ensure 3 channels
            img_tensor = preprocess(img)
            all_samples.append(img_tensor)
            all_labels.append(class_idx)
        except Exception as e:
            print(f"Could not read {img_path}: {e}")

if not all_samples:
    print("No images were loaded. Please check the data directories.")
else:
    # Stack samples and labels into tensors
    image_batch = torch.stack(all_samples).to(DEVICE)
    label_batch = torch.tensor(all_labels).to(DEVICE)

    # --- 3. Find Correctly Classified Samples ---
    with torch.no_grad():
        outputs = model(image_batch)
        initial_preds = outputs.max(1)[1]

    correctly_classified_mask = (initial_preds == label_batch)
    
    # Filter the data to only include correctly classified samples
    correct_images = image_batch[correctly_classified_mask]
    correct_labels = label_batch[correctly_classified_mask]
    correct_initial_preds = initial_preds[correctly_classified_mask]

    print(f"\nFound {len(correct_images)} correctly classified samples out of {len(all_labels)}.")

    # --- 4. Perform FGSM Attack on Correct Samples ---
    adv_examples = []
    for i in range(len(correct_images)):
        image_to_attack = correct_images[i].unsqueeze(0)
        label_to_attack = correct_labels[i].unsqueeze(0)
        
        perturbed_image, adv_pred, init_pred = generate_adv_example(model, image_to_attack, label_to_attack)
        
        # Check if the attack was successful (prediction changed)
        if adv_pred.item() != init_pred.item():
            # Store the original image tensor for visualization
            original_image_tensor = correct_images[i]
            adv_examples.append((perturbed_image.squeeze().cpu(), label_to_attack.item(), adv_pred.item(), init_pred.item(), original_image_tensor.cpu()))

    print(f"Successfully attacked {len(adv_examples)} out of {len(correct_images)} correctly classified samples.")

    # --- 5. Visualize Results ---
    if adv_examples:
        # Limit the number of examples to display to a maximum of 5, randomly selected
        import random
        
        examples_to_display = random.sample(adv_examples, 5) if len(adv_examples) > 5 else adv_examples
        num_examples = len(examples_to_display)
        print(f"Displaying {num_examples} successful attacks...")
        plt.figure(figsize=(5 * num_examples, 8)) # Adjusted figure size for horizontal layout
        
        for i, (perturbed_data, true_label, adv_pred_label, init_pred_label, original_img_tensor) in enumerate(examples_to_display):
            
            # Display Original Image
            plt.subplot(3, num_examples, i + 1) # 3 rows, num_examples cols, position i+1
            orig_img_np = denormalize(original_img_tensor).squeeze().detach().numpy().transpose(1, 2, 0)
            orig_img_np = np.clip(orig_img_np, 0, 1)
            plt.imshow(orig_img_np)
            plt.title(f"Original\nTrue: {idx_to_class[true_label]}\nPred: {idx_to_class[init_pred_label]}")
            plt.axis('off')

            # Display Adversarial Image
            plt.subplot(3, num_examples, num_examples + i + 1) # Second row
            adv_img_np = denormalize(perturbed_data).squeeze().detach().numpy().transpose(1, 2, 0)
            adv_img_np = np.clip(adv_img_np, 0, 1)
            plt.imshow(adv_img_np)
            plt.title(f"Adversarial\nTrue: {idx_to_class[true_label]}\nPred: {idx_to_class[adv_pred_label]}")
            plt.axis('off')
            
            # Display Perturbation
            plt.subplot(3, num_examples, 2 * num_examples + i + 1) # Third row
            perturbation = denormalize(perturbed_data) - denormalize(original_img_tensor)
            perturbation_np = perturbation.squeeze().detach().numpy().transpose(1, 2, 0)
            perturbation_np = np.clip(perturbation_np * 20, 0, 1)
            plt.imshow(perturbation_np)
            plt.title("Perturbation (x20)")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    else:
        print("No successful attacks to display.")
