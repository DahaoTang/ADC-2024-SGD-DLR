import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import json
import time
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    initial_lr: float = 0.01
    alpha: float = 0.1  # Control parameter for decreasing learning rate
    beta: float = 0.1   # Control parameter for increasing learning rate
    eta_min: float = 1e-6
    eta_max: float = 0.05
    batch_size: int = 128
    num_workers: int = 2

class CustomCNN(nn.Module):
    """CNN architecture as described in the paper"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

class LyapunovTrainer:
    """Implementation of Lyapunov-based training as described in the paper"""
    def __init__(self, model: nn.Module, config: TrainingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize P matrix for Lyapunov function
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.P = torch.ones(param_count, device=device)  # Diagonal P matrix
        
    def compute_gradient_vector(self) -> torch.Tensor:
        """Concatenate all gradients into a single vector"""
        return torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        
    def compute_lyapunov_value(self, gradient: torch.Tensor) -> torch.Tensor:
        """Compute Lyapunov function value V(x) = J^T P J"""
        return torch.dot(gradient * self.P, gradient)
        
    def adjust_learning_rate(self, optimizer: optim.Optimizer, delta_V: float, current_lr: float) -> float:
        """
        Adjust learning rate based on Lyapunov stability condition
        As per Algorithm 1 in the paper
        """
        if delta_V > 0:  # System is unstable
            new_lr = max(current_lr / (1 + self.config.alpha * delta_V), 
                        self.config.eta_min)
        else:  # System is stable
            new_lr = min(current_lr * (1 + self.config.beta), 
                        self.config.eta_max)
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, optimizer: optim.Optimizer) -> Dict:
        """Train for one epoch using Lyapunov-based dynamic learning rate"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        current_lr = self.config.initial_lr
        metrics = {'learning_rates': [], 'delta_Vs': [], 'Vs': []}
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass and compute loss
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Compute Lyapunov function value before update
            J_t = self.compute_gradient_vector()
            V_t = self.compute_lyapunov_value(J_t)
            
            # Update parameters
            optimizer.step()
            
            # Compute new Lyapunov function value after update
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            J_t1 = self.compute_gradient_vector()
            V_t1 = self.compute_lyapunov_value(J_t1)
            
            # Compute change in Lyapunov function
            delta_V = V_t1 - V_t
            
            # Adjust learning rate based on stability condition
            current_lr = self.adjust_learning_rate(optimizer, delta_V.item(), current_lr)
            
            # Record metrics
            metrics['learning_rates'].append(current_lr)
            metrics['delta_Vs'].append(delta_V.item())
            metrics['Vs'].append(V_t.item())
            
            # Compute accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_loss += loss.item() * target.size(0)
            
        return {
            'loss': total_loss / total,
            'accuracy': 100. * correct / total,
            **metrics
        }

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * target.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        return {
            'loss': total_loss / total,
            'accuracy': 100. * correct / total
        }

def setup_data(dataset_name: str, config: TrainingConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Setup data loaders for either CIFAR-10 or CIFAR-100"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if dataset_name == "CIFAR-10":
        dataset_class = datasets.CIFAR10
        num_classes = 10
    elif dataset_name == "CIFAR-100":
        dataset_class = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
        
    train_dataset = dataset_class(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers)
        
    return train_loader, test_loader, num_classes

def run_experiment(dataset_name: str, num_epochs: int, config: TrainingConfig):
    """Run the complete experiment comparing SGD-DLR with standard SGD"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, num_classes = setup_data(dataset_name, config)
    
    # Setup models and optimizers
    model_sgd_dlr = CustomCNN(num_classes=num_classes).to(device)
    model_sgd = CustomCNN(num_classes=num_classes).to(device)
    
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=config.initial_lr)
    lyapunov_trainer = LyapunovTrainer(model_sgd_dlr, config, device)
    basic_trainer = LyapunovTrainer(model_sgd, config, device)  # Using same trainer without DLR
    
    # Create experiment directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = f"{dataset_name}_experiment_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train with SGD-DLR
        sgd_dlr_metrics = lyapunov_trainer.train_epoch(
            train_loader, 
            optim.SGD(model_sgd_dlr.parameters(), lr=config.initial_lr)
        )
        sgd_dlr_val = lyapunov_trainer.validate(test_loader)
        
        # Train with standard SGD
        sgd_metrics = basic_trainer.train_epoch(train_loader, optimizer_sgd)
        sgd_val = basic_trainer.validate(test_loader)
        
        # Save epoch results
        epoch_results = {
            'epoch': epoch,
            'sgd_dlr': {**sgd_dlr_metrics, **sgd_dlr_val},
            'sgd': {**sgd_metrics, **sgd_val}
        }
        
        with open(os.path.join(exp_dir, f'epoch_{epoch}.json'), 'w') as f:
            json.dump(epoch_results, f)
            
    return exp_dir

if __name__ == "__main__":
    # Run experiments
    config = TrainingConfig()
    
    # CIFAR-10 experiment
    print("Starting CIFAR-10 experiment...")
    run_experiment("CIFAR-10", num_epochs=50, config=config)
    
    # CIFAR-100 experiment
    print("\nStarting CIFAR-100 experiment...")
    run_experiment("CIFAR-100", num_epochs=100, config=config)
