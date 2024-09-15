from typing import Dict, List, Tuple
import logging

import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,     # type: ignore
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,            # type: ignore
               device: torch.device) -> Tuple[float, float]:
    
    model.train()
    train_loss, train_acc = 0, 0

    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,      # type: ignore
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    model.eval() 
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader,    # type: ignore
          test_dataloader: torch.utils.data.DataLoader,     # type: ignore
          optimizer: torch.optim.Optimizer,                 # type: ignore
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Args:
        model (torch.nn.Module): A model to be trained and tested.
        train_dataloader (torch.utils.data.DataLoader): The dataloader for training.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for testing.
        optimizer (torch.optim.Optimizer): The optimizer.
        loss_fn (torch.nn.Module): The loss function.
        epochs (int): The number of epochs for training.
        device (torch.device): The device where the training and testing happen.

    Returns:
        Dict[str, List]: A dictionary of train_loss/train_acc/test_loss/test_acc.
    """
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        logger.info(f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
