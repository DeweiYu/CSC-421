import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tabulate import tabulate

# Define Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def add_noise(images, mean=0, std=0.2):
    noise = std * torch.randn_like(images) + mean
    return images + noise

def test_accuracy(net, testloader, device, add_noise_fn=None):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            if add_noise_fn:
                images = add_noise_fn(images)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 32
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training
    for epoch in range(5):  # Reduced epochs for simplicity
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.3f}')

    print('Finished Training')

    # Save Model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test Accuracy Without Noise
    accuracy = test_accuracy(net, testloader, device)
    print(f'Test Accuracy (No Noise): {accuracy * 100:.2f}%')

    # Test Accuracy With Noise
    accuracy_with_noise = test_accuracy(net, testloader, device, add_noise_fn=add_noise)
    print(f'Test Accuracy (With Noise): {accuracy_with_noise * 100:.2f}%')

    # Visualization of Noise Effect
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Original Images
    print('Original Images')
    imshow(torchvision.utils.make_grid(images))

    # Noisy Images
    noisy_images = add_noise(images)
    print('Noisy Images')
    imshow(torchvision.utils.make_grid(noisy_images))
