# PyTorch Tools

A collection of helper functions to streamline PyTorch model training, evaluation, and dataset loading. This script automates environment setup, handles dataset loading, and provides training/testing utilities for classification models.

Features

✅ Automatic Dependency Installation – Installs missing dependencies required for PyTorch workflows.

✅ Dataset Loader – Loads image datasets using torchvision.datasets.ImageFolder and applies transformations.

✅ Training & Testing Loops – Functions to handle training and evaluation for classification models.

✅ Performance Metrics – Computes accuracy, F1-score, and loss for classification tasks.

✅ Visualization – Plots loss and accuracy curves over training epochs.


# Function Descriptions

setup_torch_env(newpackages=None)

Checks for required libraries, installs missing ones, and imports necessary modules for PyTorch work. Users can specify additional packages to install.

create_dataloaders(train_dir, test_dir, train_transform, test_transform, batch_size, train_target_transform=None, test_target_transform=None)

Creates PyTorch DataLoaders for training and testing datasets with applied transformations.

train_step_classification(model, loss_fn, optimizer, train_dataloader, num_classes)

Performs a single training step for a classification model, computing loss, accuracy, and F1 score.

test_step_classification(model, loss_fn, num_classes, test_dataloader)

Runs a test step on a classification model, computing accuracy, F1 score, and loss.

train_classification(model, train_dataloader, test_dataloader, optimizer, loss_f, num_classes, epochs)

Trains a classification model for a given number of epochs, logging and returning accuracy, loss, and F1 scores for both training and testing datasets.

plot_loss_curves(results)

Plots training and testing loss and accuracy curves based on recorded results.
