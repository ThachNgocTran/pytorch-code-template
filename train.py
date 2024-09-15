# %%
import logging, sys

import torch
from torchvision import transforms

from myutils import data_setup, engine, utils
import model

logging.basicConfig(format="%(asctime)s %(levelname)s: [%(name)s] [%(funcName)s]: %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S",
                    stream=sys.stderr)

logger = logging.getLogger(__name__)

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "archive/train"
test_dir = "archive/test"

# Setup target device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create transforms.
data_transform = transforms.Compose([transforms.Resize((28, 28)),
                                     transforms.ToTensor()])

# Create DataLoaders.
train_dataloader, test_dataloader, class_names = data_setup.create_image_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model.
model = model.TinyModel(input_dim = 28 * 28 * 3,
                        hidden_dim = 128,
                        output_dim = 3).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training.
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model.
utils.save_model(model=model,
                 target_dir="models",
                 model_name="TinyModel.pth")


utils.pred_and_plot_image(model=model,
                          class_names=class_names,
                          image_path="archive/test/ADONIS/1.jpg",
                          image_size=(28, 28),
                          device=device)
