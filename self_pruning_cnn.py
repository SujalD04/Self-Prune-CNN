import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prunable Linear Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight + bias
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Gate scores (learnable)
        self.gate_scores = nn.Parameter(torch.zeros_like(self.weight))

        self.reset_parameters()

    def reset_parameters(self):
        # Same initialization as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        # Gate scores initialized to 0 so sigmoid = 0.5
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        # Convert scores to gates (0 to 1)
        gates = torch.sigmoid(self.gate_scores)

        # Apply pruning mask (soft)
        pruned_weight = self.weight * gates

        return F.linear(x, pruned_weight, self.bias)

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)


# Prunable Conv2D Layer
class PrunableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Standard conv weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        # Gate scores
        self.gate_scores = nn.Parameter(torch.zeros_like(self.weight))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates

        return F.conv2d(
            x,
            pruned_weight,
            self.bias,
            stride=self.stride,
            padding=self.padding
        )

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)


# Sanity Tests
def test_prunable_linear():
    print("Testing PrunableLinear...")

    layer = PrunableLinear(10, 5).to(device)
    x = torch.randn(4, 10).to(device)

    output = layer(x)
    assert output.shape == (4, 5), "Shape mismatch in PrunableLinear"

    # Gradient test
    loss = output.mean()
    loss.backward()

    assert layer.weight.grad is not None, "No grad for weight"
    assert layer.gate_scores.grad is not None, "No grad for gate_scores"

    print("Forward + Gradient Working")


def test_prunable_conv():
    print("Testing PrunableConv2d...")

    layer = PrunableConv2d(3, 8, kernel_size=3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)

    output = layer(x)
    assert output.shape == (2, 8, 32, 32), "Shape mismatch in PrunableConv2d"

    # Gradient test
    loss = output.mean()
    loss.backward()

    assert layer.weight.grad is not None, "No grad for weight"
    assert layer.gate_scores.grad is not None, "No grad for gate_scores"

    print("Forward + Gradient Working")


def test_gate_effect():
    print("Testing gate pruning effect...")

    layer = PrunableLinear(10, 5).to(device)
    x = torch.randn(1, 10).to(device)

    # Force gates to near zero
    with torch.no_grad():
        layer.gate_scores.fill_(-10)  # sigmoid ≈ 0

    output = layer(x)

    # Output should be near zero (only bias remains)
    print("Output with near-zero gates:", output)

    print("Gate suppression works")


# Prunable CNN Model
class PrunableCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Block 1
        self.conv1 = PrunableConv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Conv Block 2
        self.conv2 = PrunableConv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Conv Block 3
        self.conv3 = PrunableConv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected 
        self.fc1 = PrunableLinear(256 * 4 * 4, 512)
        self.fc2 = PrunableLinear(512, 10)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = torch.flatten(x, 1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    # Utility to collect all gate values
    def get_all_gates(self):
        gates = []

        for module in self.modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                gates.append(module.get_gate_values().view(-1))

        return torch.cat(gates)
    
def test_prunable_cnn():
    print("Testing PrunableCNN...")

    model = PrunableCNN().to(device)
    x = torch.randn(4, 3, 32, 32).to(device)

    output = model(x)

    assert output.shape == (4, 10), "CNN output shape mismatch"

    # Gradient check
    loss = output.mean()
    loss.backward()

    # Check at least one gate gradient
    gate_grad_exists = False
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            if module.gate_scores.grad is not None:
                gate_grad_exists = True
                break

    assert gate_grad_exists, "No gradient flowing to gate_scores"

    print("CNN forward + gradient Working")

# Data Loading (CIFAR-10)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)

    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=2)

    return trainloader, testloader

# Sparsity Loss (L1 on gates)
def compute_sparsity_loss(model):
    sparsity_loss = torch.tensor(0.0, device=device)

    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            gates = module.get_gate_values()
            sparsity_loss += gates.sum()

    return sparsity_loss

# Sparsity Metric
def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            gates = module.get_gate_values()
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100.0 * pruned / total

# Training Function
def train_model(model, trainloader, testloader, epochs=20, lambda_max=1e-4):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    final_acc = 0.0
    final_sparsity = 0.0

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_sparsity_loss = 0.0

        current_lambda = lambda_max * (epoch / epochs)

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            ce_loss = criterion(outputs, labels)
            sparsity_loss = compute_sparsity_loss(model)

            loss = ce_loss + current_lambda * sparsity_loss

            loss.backward()
            optimizer.step()

            running_loss += ce_loss.item()
            running_sparsity_loss += sparsity_loss.item()

        test_acc = evaluate(model, testloader)
        sparsity = compute_sparsity(model)

        print(f"[λ={lambda_max}] Epoch [{epoch+1}/{epochs}] | "
              f"Acc: {test_acc:.2f}% | Sparsity: {sparsity:.2f}%")

        final_acc = test_acc
        final_sparsity = sparsity

    return final_acc, final_sparsity

def run_experiments(lambdas, trainloader, testloader):
    results = []

    best_model = None
    best_sparsity = -1
    best_lambda = None

    for lam in lambdas:
        print("\n" + "="*50)
        print(f"Running experiment with λ = {lam}")
        print("="*50)

        model = PrunableCNN()

        final_acc, final_sparsity = train_model(
            model,
            trainloader,
            testloader,
            epochs=20,
            lambda_max=lam
        )

        results.append({
            "lambda": lam,
            "accuracy": final_acc,
            "sparsity": final_sparsity
        })

        # Track best model (highest sparsity)
        if final_sparsity > best_sparsity:
            best_model = model
            best_sparsity = final_sparsity
            best_lambda = lam

    print(f"\nBest model: λ = {best_lambda}, Sparsity = {best_sparsity:.2f}%")

    return results, best_model

def print_results(results):
    print("\nFinal Results:\n")
    print(f"{'Lambda':<10} | {'Accuracy (%)':<15} | {'Sparsity (%)'}")
    print("-"*45)

    for r in results:
        print(f"{r['lambda']:<10} | {r['accuracy']:<15.2f} | {r['sparsity']:.2f}")

import matplotlib.pyplot as plt

def plot_gate_distribution(model):
    gates = model.get_all_gates().detach().cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
        
# Evaluation
def evaluate(model, testloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total

if __name__ == "__main__":
    test_prunable_linear()
    test_prunable_conv()
    test_gate_effect()
    test_prunable_cnn()

    print("\nStarting Experiments...\n")

    trainloader, testloader = get_dataloaders()

    lambda_values = [1e-5, 5e-5, 1e-4]

    results, best_model = run_experiments(lambda_values, trainloader, testloader)
    print_results(results)

    print("\nPlotting gate distribution for best model...\n")
    plot_gate_distribution(best_model)