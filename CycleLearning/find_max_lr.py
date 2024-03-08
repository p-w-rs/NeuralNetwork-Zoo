# find_max_lr.py
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from models import (
    FFNetwork,
    LeNet,
    FFResNet,
)
from torch.optim import SGD


def find_lr(
    model, dataloader, optimizer, criterion, init_lr, final_lr, num_iter, device
):
    lr_lambda = lambda x: np.exp(x * np.log(final_lr / init_lr) / (num_iter - 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    losses = []
    lrs = []
    best_loss = float("inf")

    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx == num_iter:
            break
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Break if the loss is diverging
        if loss > 4 * best_loss:
            break
        if loss < best_loss:
            best_loss = loss

        loss.backward()
        optimizer.step()

        # Record the current learning rate and loss
        lrs.append(optimizer.param_groups[0]["lr"])
        losses.append(loss.item())

        scheduler.step()

    return lrs, losses


def run_experiments(models, batch_sizes, init_lr, final_lr, num_iter, device):
    for model_name, model_class in models.items():
        for batch_size in batch_sizes:
            print(f"Running experiment for {model_name} with batch size {batch_size}")

            # DataLoader setup
            train_loader = DataLoader(
                datasets.EMNIST(
                    "_data",
                    split="byclass",
                    train=True,
                    download=True,
                    transform=transforms.ToTensor(),
                ),
                batch_size=batch_size,
                shuffle=True,
            )

            # Model, Optimizer, Loss Function setup
            model = model_class().to(device)
            optimizer = SGD(model.parameters(), lr=init_lr)
            criterion = nn.CrossEntropyLoss()

            # Find learning rate
            lrs, losses = find_lr(
                model,
                train_loader,
                optimizer,
                criterion,
                init_lr,
                final_lr,
                num_iter,
                device,
            )

            # Plotting
            plt.figure()
            plt.plot(lrs, losses)
            plt.xscale("log")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.title(f"LR vs Loss for {model_name} with batch size {batch_size}")
            plt.savefig(f"results/{model_name}_batch_{batch_size}.png")
            plt.close()

            # Save max learning rate before loss divergence
            max_lr = lrs[losses.index(min(losses))]
            with open(
                f"results/max_lr_{model_name}_batch_{batch_size}.txt",
                "w",
            ) as f:
                f.write(str(max_lr))


# Device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Running the experiments
models = {"FFNetwork": FFNetwork, "FFResNet": FFResNet, "LeNet": LeNet}
batch_sizes = [32, 128, 512, 1024]
init_lr = 1e-7
final_lr = 3
num_iter = 100  # Adjust based on your dataset size and batch_sizes
run_experiments(models, batch_sizes, init_lr, final_lr, num_iter, device)
