# demo for C&W attack algorithm
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore",category=UserWarning, module="torchvision") 

# import the model
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.eval()

# load the data
transform = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=10,shuffle=True)

# define the CW attack function
def cw_attack(model,images,target,c =.05,max_iter = 20,lr = 1e-3 ):
    # get the batch size
    batch_size = images.size(0)
    if batch_size == 0:
        return torch.tensor([]) # Return empty tensor if no images to attack

    # initialize the perturbation and optimizer
    # delta must be a leaf tensor for the optimizer.
    # Using torch.nn.Parameter is a clean way to define an optimizable tensor.
    delta = torch.nn.Parameter(torch.rand_like(images) * 0.01)
    optimizer = torch.optim.Adam([delta],lr=lr)
    
    # iteration loop
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # forward: use tanh to keep image in [0,1] range
        perturbed_images = torch.tanh(images + delta) * 0.5 + 0.5
        logits = model(perturbed_images)
        
        # l2 norm loss (per image in batch)
        l2_loss = torch.norm(delta.view(batch_size,-1),p=2, dim=1).mean()
        
        # classification loss
        target_logits = logits.gather(1,target.unsqueeze(1)).squeeze()
        # Ensure torch.eye is on the correct device
        max_other_logits = torch.max(logits - torch.eye(10, device=images.device)[target]*1e9, dim=1)[0]
        classification_loss = torch.clamp(max_other_logits - target_logits, min = 0).mean()
        loss = l2_loss + c * classification_loss
        
        # BP
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = model(perturbed_images).max(1)[1]
            # Use .all() for tensor comparison in a batch
            if (preds == target).all():
                print(f"Iteration {i+1}, successfully perturbed all samples to target")
                break
    
    return perturbed_images

# test the algorithm for one batch
# Get one batch of data from the loader
data, orig_labels = next(iter(testloader))

# --- Pre-filter samples that are already correctly classified ---
with torch.no_grad():
    initial_preds = model(data).max(1)[1]

correctly_classified_mask = (initial_preds == orig_labels)

# Filter the data, labels, and create attack targets for the correctly classified samples
data_to_attack = data[correctly_classified_mask]
labels_of_attack_data = orig_labels[correctly_classified_mask]
num_to_attack = data_to_attack.size(0)

print(f"Original batch size: {data.size(0)}, Correctly classified: {num_to_attack}")

# Define a target class for the attack (e.g., class 2: 'bird')
attack_target = torch.full_like(labels_of_attack_data, 2)

print("Attacking correctly classified images...")
perturbed_data = cw_attack(model, data_to_attack, attack_target)
print("Attack finished.")

# --- Prepare data for visualization ---
# We will visualize all original images, but only show adversarial versions for those attacked.
final_perturbed_data = data.clone() # Start with original data
if num_to_attack > 0:
    final_perturbed_data[correctly_classified_mask] = perturbed_data # Replace attacked ones with perturbed versions

# Get initial and adversarial predictions for the full batch
with torch.no_grad():
    init_pred = model(data).max(1)[1]
    adv_pred = model(final_perturbed_data).max(1)[1]

# --- Filter for successfully attacked samples for visualization ---
success_mask = torch.zeros_like(correctly_classified_mask)
if num_to_attack > 0:
    # A success is when an attacked sample's prediction matches the attack target
    success_mask[correctly_classified_mask] = (adv_pred[correctly_classified_mask] == attack_target)

# Get the data for the successful attacks
successful_orig_data = data[success_mask]
successful_adv_data = final_perturbed_data[success_mask]
successful_init_pred = init_pred[success_mask]
successful_adv_pred = adv_pred[success_mask]
successful_orig_labels = orig_labels[success_mask]
num_successful = successful_orig_data.size(0)

print(f"Attack success rate for this batch: {num_successful / num_to_attack if num_to_attack > 0 else 0:.2%}")

# plot the successfully attacked examples
if num_successful > 0:
    print(f"Displaying {num_successful} successfully attacked samples...")
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(num_successful * 2.5, 5)) # Adjust figure size dynamically
    
    for i in range(num_successful):
        # Display Original Image
        plt.subplot(2, num_successful, i + 1)
        img_to_show = np.clip(successful_orig_data[i].squeeze().detach().numpy().transpose(1, 2, 0), 0, 1)
        plt.imshow(img_to_show)
        plt.title(f"Original\nPred: {classes[successful_init_pred[i].item()]}\nTrue: {classes[successful_orig_labels[i].item()]}")
        plt.axis('off')

        # Display Adversarial Image
        plt.subplot(2, num_successful, i + 1 + num_successful)
        adv_img_to_show = np.clip(successful_adv_data[i].squeeze().detach().numpy().transpose(1, 2, 0), 0, 1)
        plt.imshow(adv_img_to_show)
        # The target is the same for all, so we can index attack_target[0] if num_to_attack > 0
        target_class_name = classes[attack_target[0].item()] if num_to_attack > 0 else "N/A"
        plt.title(f"Adversarial\nPred: {classes[successful_adv_pred[i].item()]}\nTarget: {target_class_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No samples were successfully attacked in this batch.")

