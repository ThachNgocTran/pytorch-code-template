import os
from typing import List, Tuple
import logging

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

NUM_WORKERS = cpu_count if (cpu_count:= os.cpu_count()) is not None else 1

def create_image_dataloaders(train_dir: str, 
                             test_dir: str, 
                             transform: transforms.Compose = transforms.Compose([transforms.ToTensor()]),  # scale to float [0:1]
                             batch_size: int = 32, 
                             num_workers: int = NUM_WORKERS) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Creates training/testing DataLoaders.

    Args:
        train_dir (str): the image folder for training.
        test_dir (str): the image folder for testing.
        transform (transforms.Compose): the transforms on the images. Default: ToTensor().
        batch_size (int): the batch size. Default: 32.
        num_workers (int, optional): the number of workers for DataLoader. Default: NUM_WORKERS.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: train_dataloader, test_dataloader, class_names
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    
    class_names = train_data.classes

    logger.info(f"Found [{len(train_data.imgs)}] training images ([{len(class_names)}] classes), "\
                f"[{len(test_data.imgs)}] testing images ([{len(test_data.classes)}] classes).")
    if not ((test_class_names := set(test_data.classes)) <= (train_class_names := set(train_data.classes))):
        logger.warning(f"Testing image classes are not less than or equal to training image classes."\
            f"[{test_class_names}] vs. [{train_class_names}]")

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (train_dataloader, test_dataloader, class_names)
