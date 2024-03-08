# super_convergence.py
import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from models import FFNetwork, LeNet, FFResNet
from experiment_runner import DataLoaderEMNIST, TrainingManager

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

models = {"FFNetwork": FFNetwork, "FFResNet": FFResNet, "LeNet": LeNet}
batch_sizes = [32, 128, 512, 1024]
optimizers = {"Adam": optim.Adam, "RMS": optim.RMSprop, "SGD": optim.SGD}
for model_name, model_class in models.items():
    for batch_size in batch_sizes:
        epochs = 100
        max_lr = float(
            open(
                f"results/max_lr_{model_name}_batch_{batch_size}.txt", "r"
            ).readlines()[0]
        )
        dataloader = DataLoaderEMNIST(batch_size, 1024)
        total_steps = len(dataloader.train_dataloader) * epochs
        model = model_class().to(device)
        loss_fn = nn.CrossEntropyLoss()

        for optimizer_name, optimizer_class in optimizers.items():
            optimizer = optimizer_class(model.parameters(), lr=max_lr / 10)

            scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
            manager = TrainingManager(
                model, dataloader, optimizer, loss_fn, device, scheduler=None
            )
            manager.fit(epochs)
            manager.plot_metrics(
                f"{model_name}_batch_{batch_size}_{optimizer_name}.png"
            )
