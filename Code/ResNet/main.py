import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import ResNet50

def main():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    model = ResNet50(image_channels=3, num_classes=10).to(device)
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"训练集大小: {len(train_dataset)} 张图片, {len(train_loader)} 个批次")
    print(f"测试集大小: {len(test_dataset)} 张图片, {len(test_loader)} 个批次")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []

    for _ in range(10):
        total_train_loss = 0 
        train_total = 0
        train_correct = 0
        
        model.train() # 设置为训练模式
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
    
            # 1. 前向传播
            scores = model(data)
            loss = criterion(scores, targets)
            total_train_loss += loss.item()
            _, predict = scores.max(1)
            train_total += targets.size(0)
            train_correct += predict.eq(targets).sum().item()
            # 2. 反向传播
            optimizer.zero_grad()
            loss.backward()
    
            # 3. 更新权重
            optimizer.step()
    
        avg_train_loss = total_train_loss/len(train_loader)
        avg_train_acc = train_correct / train_total
        train_loss.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        
        # 测试
        model.eval()
        total_test_loss = 0
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(device)
                targets = targets.to(device)
                scores = model(data)
                loss = criterion(scores, targets)
                total_test_loss += loss.item()
                _,predict = scores.max(1)
                test_total += targets.size(0)
                test_correct += predict.eq(targets).sum().item()
                
    
        avg_test_loss = total_test_loss/len(test_loader)
        avg_test_acc = test_correct / test_total
        test_loss.append(avg_test_loss)
        test_acc.append(avg_test_acc)
        
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.show()
    
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.legend()
    plt.show()
    
    # 保存模型权重到本地文件
    # save_path = "./resnet_cifar10_quicktest.pth"
    # torch.save(model.state_dict(), save_path)
    # print(f"\n[4] Model weights saved to '{save_path}'")
    
if __name__ == "__main__":
    main()