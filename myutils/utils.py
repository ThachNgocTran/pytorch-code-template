import torch, torchvision, torchvision.transforms
from pathlib import Path
import logging
from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Save a PyTorch model.

    Args:
        model (torch.nn.Module): the model to be saved.
        target_dir (str): the target directory to save into.
        model_name (str): the model name, ended with ".pth" or ".pt".
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, 
                          exist_ok=True)

    if not (model_name.endswith(".pth") or model_name.endswith(".pt")):
        model_name += ".pt"
    model_save_path = target_dir_path / model_name

    logger.info(f"Saving model [{model_name}] as [{model_save_path}].")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    transform: torchvision.transforms.Compose | None = None,
    image_size: Tuple[int, int] = (224, 224),
    device: torch.device = device,
):
    """Do the prediction, given the model, and plot the image.

    Args:
        model (torch.nn.Module): The model used to predict.
        class_names (List[str]): The set of class names used to show the predicted class based on the predicted index.
        image_path (str): The image path for prediction.
        image_size (Tuple[int, int], optional): The image size. Default: (224, 224).
        transform (torchvision.transforms.Compose, optional): The transform on the image.
        device (torch.device, optional): The device where the prediction takes place.
    """
    img = Image.open(image_path)

    image_transform = transform if transform is not None else \
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   # Imagenet

    ### Predict on image ###
    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
