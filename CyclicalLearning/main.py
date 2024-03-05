from torch import nn, optim
from experiment_runner import NeuralNetwork, DataLoaderEMNIST, TrainingManager

# batch size, initial learning rate, number of epochs, optimizer, scheduler
# 32, default, 100, SGD, None
# 32, default, 100, RMSProp, None
# 32, default, 100, AdamW, None
# 32, max learning rate found, 100,
experiments = []
