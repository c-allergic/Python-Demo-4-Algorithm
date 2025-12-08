# define and train a Classification Model
# import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import PIL.Image as Image
import os

# define the model
class DogNet(nn.Module):
    def __init__(self,num_classes=3):
        super(DogNet, self).__init__()
        # define kernel for RGB image
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.fc1 = None
        self.fc2 = nn.Linear(128,num_classes)
        self.pooling = nn.MaxPool2d(kernel_size=2,padding=1) # return half of the height and width
        
    def forward(self,x):
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        if self.fc1 is None:
            flattened_size = x.shape[1] * x.shape[2] * x.shape[3]
            self.fc1 = nn.Linear(in_features=flattened_size, out_features=128)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        x = self.fc2(x)
        return x # Return raw logits, not argmax or softmax
    
# data loader
def load_data(root='./data/Animals'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # load all the images from the root directory
    class_to_idx = {'cats':0,'dogs':1,'snakes':2}
    labels = []
    images = []
    for class_name in class_to_idx:
        for image_name in os.listdir(os.path.join(root,class_name)):
            images.append(transform(Image.open(os.path.join(root,class_name,image_name))))
            labels.append(class_to_idx[class_name])
    return images, labels

# split the data into training and validation sets
def split_data(images, labels, split_ratio=.8):
    # shuffle the data
    import random
    data = list(zip(images, labels))
    random.shuffle(data)

    # split the data into training and validation sets
    train_data = data[:int(len(data)*split_ratio)]
    val_data = data[int(len(data)*split_ratio):]
    return train_data, val_data

# train the model
def train_model(model, data, epochs=8, learning_rate=1e-3):
    # define the loss function
    criterion = nn.CrossEntropyLoss()
    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # train the model
    for epoch in range(epochs):
        for sample in data:
            image  = sample[0].unsqueeze(0)
            label = torch.tensor([sample[1]], dtype=torch.long)
            
            # forward pass
            outputs = model(image)
            loss = criterion(outputs, label)
            # backward pass and update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return model


if __name__ == '__main__':
    images, labels = load_data()
    model = DogNet()
    train_data, val_data = split_data(images, labels)
    train_model(model, train_data)
    # save the model
    torch.save(model.state_dict(), 'animal_classifier.pth')
    print(f"Model saved successfully to 'animal_classifier.pth'")
    
    