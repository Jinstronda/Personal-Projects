import subprocess
import sys
import importlib
"""
First we will install all necessary packages and check if they are not present, if they are not present this Script Will Download Them"""
import subprocess
import sys
import os

def setup_torch_env(
    newpackages = None
):
    """Checks for required libraries, installs missing ones, and imports everything needed for PyTorch work.
    Args:
    newpackages : New Packages that wanna be imported by the User apart from the Default Pytorch Libraries
    """
    import importlib
    packages = {
        "torch": "torch",
        "torchvision": "torchvision",
        "torchinfo": "torchinfo",
        "torchmetrics": "torchmetrics",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "pandas": "pandas",
        "requests": "requests",
    }
    if newpackages and isinstance(newpackages,list):
      for module in newpackages:
        if module not in packages:
          packages[module] = module
    for module, package in packages.items():
        try:
            importlib.import_module(module)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    global torch, nn, plt, np, pd, requests, zipfile, Path, DataLoader, datasets, transforms, F1Score, ConfusionMatrix, summary
    import torch
    from torch import nn
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import zipfile
    from pathlib import Path
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchmetrics import F1Score, ConfusionMatrix
    from torchinfo import summary

    print("✅ PyTorch environment setup complete!")
setup_torch_env()

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  train_transform: transforms.Compose,
  test_transform: transforms.Compose,
  batch_size: int,
  train_target_transform = None,
  test_target_transform = None
):
  """Creates Dataloaders for Training and Testing

  Args:
  train_dir : Training Directory.
  test_dir: Testing Directory.
  train_transform: Transforms Compose Function to do Data Transformation in Train Data
  test_transform: Transforms Compose Function To do Data Transformation in Test Data
  batch_size: Size of batches for data Loaders
  test_target_transform: Transforms Compose Function to do Data Transformation in Test Target Data
  train_target_transform: Transforms Compose Function to do Data Transformation in Train Target Data

  Returns
  A tuple of a (train_dataloader, test_dataloader, class_names)
  class_names is a list of the Target Classes
"""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  NUM_WORKERS = os.cpu_count()
  train_data = datasets.ImageFolder(root=train_dir,transform=train_transform, target_transform = train_target_transform)
  test_data = datasets.ImageFolder(root=test_dir, transform=test_transform, target_transform = test_target_transform)
  class_names = train_data.classes
  train_dataloader = DataLoader(
      dataset=train_data,
      num_workers=NUM_WORKERS,
      shuffle=True,
      batch_size = batch_size,
      pin_memory=True
  )
  test_dataloader = DataLoader(
      dataset=test_data,
      num_workers=NUM_WORKERS,
      shuffle=False,
      batch_size = batch_size,
      pin_memory = True
  )
  return (train_dataloader,test_dataloader,class_names)

def train_step_classification(
    model,
    loss_fn,
    optimizer,
    train_dataloader,
    num_classes
  ):
  """Does one Epoch of Training for Classification Tasks

  Args:
  model: A Pytorch Model
  loss_fn: The loss Function
  optimizer: THe Optimizer Being Used
  train_dataloader: The DataLoader for The Training Data
  num_classes: The number of classes for the classification task

  Returns:
  Tuple: (train_accuracy,train_f1,train_loss)
  train_accuracy: The accuracy on the training Data
  train_f1: The F1 Score on the training data for that Epoch
  train_loss: The Loss of that Epoch.
  """
  from torchmetrics import F1Score, ConfusionMatrix, Accuracy
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if num_classes == 2:
    f1_metric = F1Score(task="binary",num_classes=num_classes).to(device)
    accuracy_metric = Accuracy(task="binary",num_classes=num_classes).to(device)
  else:
    f1_metric = F1Score(task="multiclass", num_classes=num_classes).to(device)
    accuracy_metric = Accuracy(task="multiclass",num_classes=num_classes).to(device)
  total_loss = 0.0
  model.to(device)
  model.train()
  for X,y in train_dataloader:
    X,y = X.to(device), y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred,y)
    total_loss += loss.item()
    accuracy_metric.update(y_pred,y)
    f1_metric.update(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  avg_loss = total_loss / len(train_dataloader)
  overall_accuracy = accuracy_metric.compute().item()
  overall_f1 = f1_metric.compute().item()
  accuracy_metric.reset()
  f1_metric.reset()
  return (avg_loss,overall_f1,overall_accuracy)

def test_step_classification(
    model,
    loss_fn,
    num_classes,
    test_dataloader
):
  """ Do a Test Step on the Model for Classification Tasks

  Args:
  model = A pytorch Model
  loss_fn = A pytorch Loss Function
  num_classes = The number of classes in order to create Accuracy and F1 Score
  test_dataloader = The Test Dataloader to perform the Checks.

  Returns:
  Tuple: (test_accuracy,test_f1,test_loss)
  test_accuracy : The Average Accuracy over the Dataloader
  test_f1: The Average f1 score over the DataLoader
  test_loss : The Average loss over the DataLoader
  """
  import torch
  from torchmetrics import F1Score, ConfusionMatrix, Accuracy
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if num_classes == 2:
    f1_metric = F1Score(task="binary",num_classes=num_classes).to(device)
    accuracy_metric = Accuracy(task="binary",num_classes=num_classes).to(device)
  else:
    f1_metric = F1Score(task="multiclass", num_classes=num_classes).to(device)
    accuracy_metric = Accuracy(task="multiclass",num_classes=num_classes).to(device)
  total_loss = 0.0
  model.to(device)
  model.eval() # Puts the model into eval mode to notify the layers
  with torch.inference_mode(): # Apparently this is more effective than no grad
    for X,y in test_dataloader:
      X,y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred,y)
      total_loss += loss.item()
      accuracy_metric.update(y_pred,y)
      f1_metric.update(y_pred,y)
  avg_loss = total_loss / len(test_dataloader)
  overall_accuracy = accuracy_metric.compute().item()
  overall_f1 = f1_metric.compute().item()
  accuracy_metric.reset()
  f1_metric.reset()
  return (avg_loss,overall_f1,overall_accuracy)

def train_classification(model,
                         train_dataloader,
                         test_dataloader,
                         optimizer,
                         loss_f,
                         num_classes,
                         epochs):
    """
    Trains and evaluates a classification model over a specified number of epochs.

    This function performs training using the train_step_classification function
    and evaluates the model by calling the `test_step_classification` function at the end
    of each epoch. It collects metrics and returns them in a Dictionary

    Args:
        model : The PyTorch model to be trained.
        train_dataloader : DataLoader for the training dataset.
        test_dataloader : DataLoader for the testing dataset.
        optimizer ): The optimizer used for training.
        loss_f : The loss function used for training.
        num_classes (: The number of classes in the classification task.
        epochs (: The total number of training epochs.

    Returns:
        dict: A dictionary containing training and testing metrics for each epoch.
              The keys include:
                - train_loss: List of training losses per epoch.
                - train_f": List of training F1 scores per epoch.
                - train_acc: List of training accuracies per epoch.
                - test_loss: List of testing losses per epoch.
                - test_f1: List of testing F1 scores per epoch.
                - "est_acc: List of testing accuracies per epoch.
    """
    from tqdm.auto import tqdm
    results = {
        "train_loss": [],
        "train_f1": [],
        "train_acc": [],
        "test_loss": [],
        "test_f1": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_f1, train_acc = train_step_classification(
            model=model,
            loss_fn=loss_f,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            num_classes=num_classes
        )
        test_loss, test_f1, test_acc = test_step_classification(
            model=model,
            loss_fn=loss_f,
            num_classes=num_classes,
            test_dataloader=test_dataloader
        )
        if epochs < 20:
          print(f"Epoch: {epoch} | "
                f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
        if epochs > 20 and epochs <50:
          if epoch % 5 == 0:
            print(f"Epoch: {epoch} | "
                  f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
        if epochs > 50:
          if epoch%10 == 0:
            print(f"Epoch: {epoch} | "
                  f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Test Acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_f1"].append(train_f1)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_f1"].append(test_f1)
        results["test_acc"].append(test_acc)
    return results

def plot_loss_curves(results):
  """ Plots Loss and Accuracy Curve for a Model With a Results Dictionary based on my Training Function

  Args:
        results (dict): A dictionary containing:
            - "train_loss" (list): Loss values for the training set over epochs.
            - "test_loss" (list): Loss values for the test set over epochs.
            - "train_acc" (list): Accuracy values for the training set over epochs.
            - "test_acc" (list): Accuracy values for the test set over epochs.
  """
  import matplotlib.pyplot as plt
  train_loss = results["train_loss"]
  test_loss = results["test_loss"]
  train_acc = results["train_acc"]
  test_acc = results["test_acc"]
  train_epochs = range(len(train_loss))
  test_epochs = range(len(test_loss))
  plt.figure(figsize=(15,7))
  plt.subplot(1,2,1)
  plt.plot(train_epochs, train_loss, label = "Train Loss")
  plt.plot(test_epochs, test_loss, label = "Test Loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(train_epochs,train_acc,label = "Train Accuracy")
  plt.plot(test_epochs, test_acc, label = "Test Accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()



def download_data(source: str, destination: str, remove_source: bool = True):
    """
    Downloads a zip file from a source URL, extracts it into a destination folder,
    and optionally removes the zip file after extraction.
    """

    data_path = Path("Data/")
    image_path = data_path / destination

    if image_path.exists() and any(image_path.iterdir()):  # Check if folder exists and is not empty
        print(f"✅ {image_path} directory already exists and is not empty. Skipping download...")
        return image_path

    print(f"❌ {image_path} directory not found. Creating it now...")
    image_path.mkdir(parents=True, exist_ok=True)

    # Download the file
    target_file = Path(source).name
    zip_path = data_path / target_file

    try:
        print(f"⬇️ Downloading {target_file} from {source}...")
        response = requests.get(source, stream=True)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):  # Stream download
                f.write(chunk)

        print("✅ Download complete.")

        # Extract the zip file
        print("📦 Extracting contents...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(image_path)

        print(f"✅ Extraction complete. Files saved to {image_path}")

        # Remove source zip file if specified
        if remove_source:
            os.remove(zip_path)
            print(f"🗑️ Removed source zip file: {zip_path}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {source}: {e}")
        return None
    except zipfile.BadZipFile:
        print("❌ Error: Downloaded file is not a valid zip archive.")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

    return image_path
