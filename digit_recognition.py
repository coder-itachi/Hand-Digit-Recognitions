import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import idx2numpy
from torchvision import transforms
import matplotlib.pyplot as plt

# Custom MNIST Dataset Loader
class CustomMNIST(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = idx2numpy.convert_from_file(images_path)
        self.labels = idx2numpy.convert_from_file(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the path to your local dataset files
train_images_path = './data/MNIST/raw/train-images.idx3-ubyte'
train_labels_path = './data/MNIST/raw/train-labels.idx1-ubyte'
test_images_path = './data/MNIST/raw/t10k-images.idx3-ubyte'
test_labels_path = './data/MNIST/raw/t10k-labels.idx1-ubyte'

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor()    # Normalize between [0, 1]
])

# Create datasets
train_dataset = CustomMNIST(train_images_path, train_labels_path, transform=transform)
test_dataset = CustomMNIST(test_images_path, test_labels_path, transform=transform)

# Create data loaders
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Display an example image from the dataset
dataiter = iter(trainloader)
images, labels = next(dataiter)
plt.imshow(images[0].numpy().squeeze(), cmap='gray')
plt.title(f"Label: {labels[0]}")
plt.show()

# Define the neural network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First hidden layer (input: 28x28 image)
        self.fc2 = nn.Linear(128, 64)       # Second hidden layer
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 digits)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten image from 28x28 to 784
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to second layer
        x = torch.log_softmax(self.fc3(x), dim=1)  # Log-Softmax for output
        return x

# Instantiate the network
model = Network()

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1)  # Flatten images
        optimizer.zero_grad()  # Clear gradients

        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}", flush=True)

# Evaluate the model on test data
correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print(f"Model Accuracy = {correct_count/all_count * 100}%")

# Making predictions
images, labels = next(iter(testloader))
img = images[0].view(1, 784)

with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
predicted_label = ps.argmax()
print(f"Predicted Label: {predicted_label}")
plt.imshow(images[0].numpy().squeeze(), cmap='gray')
plt.title(f"Actual Label: {labels[0]}")
plt.show()
