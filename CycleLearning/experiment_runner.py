# experiment_runner.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import FFNetwork, LeNet, FFResNet
import matplotlib.pyplot as plt
import os


class DataLoaderEMNIST:
    def __init__(self, train_batch_size, test_batch_size, shuffle=True):
        self.train_dataloader = DataLoader(
            datasets.EMNIST(
                root="_data",
                split="byclass",
                train=True,
                download=True,
                transform=ToTensor(),
            ),
            batch_size=train_batch_size,
            shuffle=shuffle,
        )

        self.test_dataloader = DataLoader(
            datasets.EMNIST(
                root="_data",
                split="byclass",
                train=False,
                download=True,
                transform=ToTensor(),
            ),
            batch_size=test_batch_size,
            shuffle=shuffle,
        )

    def get(self):
        return self.train_dataloader, self.test_dataloader


class TrainingManager:
    def __init__(self, model, dataloader, optimizer, loss_fn, device, scheduler=None):
        self.model = model
        self.train_loader, self.test_loader = dataloader.get()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
        }
        self.best_accuracy = 0.0

    def train_step(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        yh = self.model(x)
        loss = self.loss_fn(yh, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), (yh.argmax(1) == y).type(torch.float).sum().item()

    def test_step(self, x, y):
        self.model.eval()
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            yh = self.model(x)
            loss = self.loss_fn(yh, y)
        return loss.item(), (yh.argmax(1) == y).type(torch.float).sum().item()

    def train_epoch(self):
        train_loss, train_correct = 0, 0
        for batch, (x, y) in enumerate(self.train_loader):
            loss, correct = self.train_step(x, y)
            train_loss += loss
            train_correct += correct
        if self.scheduler:
            self.scheduler.step()

        self.history["train_loss"].append(train_loss / len(self.train_loader))
        self.history["train_accuracy"].append(
            100.0 * train_correct / len(self.train_loader.dataset)
        )

    def test_epoch(self):
        test_loss, test_correct = 0, 0
        for x, y in self.test_loader:
            loss, correct = self.test_step(x, y)
            test_loss += loss
            test_correct += correct
        accuracy = 100.0 * test_correct / len(self.test_loader.dataset)
        self.history["test_loss"].append(test_loss / len(self.test_loader))
        self.history["test_accuracy"].append(accuracy)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            torch.save(self.model.state_dict(), "results/best_model.pth")
            print("Saved best model with accuracy: {:.2f}%".format(self.best_accuracy))

    def fit(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self.train_epoch()
            self.test_epoch()
            print(
                f'Train Loss: {self.history["train_loss"][-1]:.4f}, Train Acc: {self.history["train_accuracy"][-1]:.2f}%'
            )
            print(
                f'Test Loss: {self.history["test_loss"][-1]:.4f}, Test Acc: {self.history["test_accuracy"][-1]:.2f}%'
            )

    def plot_metrics(self, plot_name):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["test_loss"], label="Test Loss")
        plt.title("Loss over epochs")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_accuracy"], label="Train Accuracy")
        plt.plot(self.history["test_accuracy"], label="Test Accuracy")
        plt.title("Accuracy over epochs")
        plt.legend()
        plt.savefig(os.path.join("results", f"{plot_name}.png"))


# Example usage
if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    # Hyperparameters and DataLoader
    batch_size = 512
    learning_rate = 1e-2
    epochs = 2
    dataloader = DataLoaderEMNIST(batch_size, 1024)

    # Model, Optimizer, and Loss Function
    model = FFNetwork().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Optionally, define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training Manager
    manager = TrainingManager(
        model, dataloader, optimizer, loss_fn, device, scheduler=None
    )
    manager.fit(epochs)
    manager.plot_metrics()
