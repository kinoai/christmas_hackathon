import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_modules.mnist_digits_datamodule import transforms
from data_modules.mnist_digits_datamodule.datasets import TestDataset
from PIL import Image

# the LitModel you import should be the same as the one you used for training!
from models.snake_classifier_v2.lightning_module import LitModel
from torchvision.transforms import transforms


CLASSES = ['agkistrodon-contortrix',
           'agkistrodon-piscivorus',
           'coluber-constrictor',
           'crotalus-atrox',
           'crotalus-horridus',
           'crotalus-ruber',
           'crotalus-scutulatus',
           'crotalus-viridis',
           'diadophis-punctatus',
           'haldea-striatula',
           'heterodon-platirhinos',
           'lampropeltis-californiae',
           'lampropeltis-triangulum',
           'masticophis-flagellum',
           'natrix-natrix',
           'nerodia-erythrogaster',
           'nerodia-fasciata',
           'nerodia-rhombifer',
           'nerodia-sipedon',
           'opheodrys-aestivus',
           'pantherophis-alleghaniensis',
           'pantherophis-emoryi',
           'pantherophis-guttatus',
           'pantherophis-obsoletus',
           'pantherophis-spiloides',
           'pantherophis-vulpinus',
           'pituophis-catenifer',
           'rhinocheilus-lecontei',
           'storeria-dekayi',
           'storeria-occipitomaculata',
           'thamnophis-elegans',
           'thamnophis-marcianus',
           'thamnophis-proximus',
           'thamnophis-radix',
           'thamnophis-sirtalis']


def predict():
    """
        This method is example of inference with a trained model.
        It Loads trained image classification model from checkpoint.
        Then it loads example image and predicts its label.
        Model used in lightning_module.py should be the same as during training!!!
    """

    CKPT_PATH = "../logs/christmas_hackermoon_hackathon/4hhio7dx/checkpoints/epoch=10.ckpt"

    # load model from checkpoint
    trained_model = LitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    breeds = []

    # load data
    test_data = TestDataset(img_dir="../data/snake_dataset/test", transform=transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]))

    BATCH_SIZE = 128
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    for img in test_loader:
        output = trained_model(img)
        preds = torch.argmax(output, dim=1)
        preds = preds.cpu().tolist()
        print(preds)
        for i in range(len(preds)):  # Batch size
            breeds.append(str(CLASSES[preds[i]]))

    print(len(breeds))

    testIds = pd.read_csv("../data/snake_dataset/test.csv")
    print(testIds.head())

    breeds = pd.DataFrame(breeds, columns=["breeds"])
    print(breeds.head())

    # result = pd.merge(testIds, breeds, how='left', left_on=['image_id'], right_on=['breeds'])
    result = testIds.join(breeds)
    result.to_csv("results.csv")


if __name__ == "__main__":
    predict()
