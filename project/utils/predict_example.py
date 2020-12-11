import torch

from data_modules.mnist_digits_datamodule import transforms
from data_modules.mnist_digits_datamodule.datasets import TestDataset
from PIL import Image

# the LitModel you import should be the same as the one you used for training!
from models.simple_mnist_classifier.lightning_module import LitModel
from torchvision.transforms import transforms


def predict():
    """
        This method is example of inference with a trained model.
        It Loads trained image classification model from checkpoint.
        Then it loads example image and predicts its label.
        Model used in lightning_module.py should be the same as during training!!!
    """

    CKPT_PATH = "logs/christmas_hackermoon_hackathon/4hhio7dx/checkpoints/epoch=10.ckpt"

    # load model from checkpoint
    trained_model = LitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    data = TestDataset(img_dir="data/snake_dataset/test", transform=transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ]))

    for img in data:
        output = trained_model(img)
        preds = torch.argmax(output, dim=1)
        print(preds)


if __name__ == "__main__":
    predict()
