# Lab ML 1

This project implements a convolutional neural network using **EfficientNetV2** to classify images from the **CIFAR-10** dataset. It covers the complete pipeline: downloading the dataset, preprocessing, model training, evaluation, and testing.

## Quick Start Example

Run the following example to train and evaluate the model on CIFAR-10:

```python
import torch
import torch.nn as nn
import torch.optim as optim

import src.download as download
import src.ingestion as ingestion
import src.loader as loader
import src.model as mdl
import src.train_model as train_model
import src.test_model as test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
config = {
    'test_size': 0.2,
    'val_size': 0.2,
    'random_state': 42,
    'lr': 0.001,
}

# Step 1: Download and extract data
cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
cifar10_dir = download.download_and_extract(cifar10_url, "./data/")

# Step 2: Preprocess data
train_df, val_df, test_df = ingestion.process_data(cifar10_dir, config)

# Step 3: Create data loaders
train_loader = loader.create_data_loader(train_df, config)
val_loader = loader.create_data_loader(val_df, config)
test_loader = loader.create_data_loader(test_df, config)

# Step 4: Define model, loss, optimizer
model = mdl.EfficientNetV2(n_classes=10).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"])

# Step 5: Train the model
best_model_path = train_model.train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs=15, device=device)

# Step 6: Evaluate the model
test_model.test_model(model, test_loader, loss_function, device)
```