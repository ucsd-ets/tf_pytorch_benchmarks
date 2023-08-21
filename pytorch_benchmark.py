title = "Beginning PyTorch Benchmark"
dashes = "-" * (len(title) + 10)  # Adding 10 extra dashes

formatted_text = f"{dashes}\n{title}\n{dashes}"
print(formatted_text)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

print(f"If this program exits without printing anything else, check if nvidia-smi returns full vram. If the vram is all taken up, the IPykernel might be holding all the vram, which may be caused by Tensorflow running on the IPykernel.")

# Checkpoint 0: Load the Fashion MNIST dataset
start_time = time.time()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

end_time = time.time()
print(f"Time taken for Loading Data: {end_time - start_time:.4f} seconds")

# No explicit preprocessing step in PyTorch as it's handled in the transforms
print(f"No explicit preprocessing step in PyTorch as it's handled in the transforms")

# Checkpoint 2: Define the model
start_time = time.time()
program_start = start_time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

model = Net()

end_time = time.time()
print(f"Time taken for Model Definition: {end_time - start_time:.4f} seconds")

# Checkpoint 3: Compile the model
start_time = time.time()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

end_time = time.time()
print(f"Time taken for Model Compilation: {end_time - start_time:.4f} seconds")

# Checkpoint 4: Train the model
start_time = time.time()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(5):
    train(model, device, train_loader, optimizer, criterion, epoch)

end_time = time.time()
print(f"Time taken for Training: {end_time - start_time:.4f} seconds")

# Checkpoint 5: Evaluate the model
start_time = time.time()

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

test_loss, test_acc = test(model, device, test_loader, criterion)

end_time = time.time()
print(f"Time taken for Evaluation: {end_time - start_time:.4f} seconds")
print(f"Total time taken (without loading data): {end_time - program_start:.4f} seconds")
print('\nTest accuracy:', test_acc)
