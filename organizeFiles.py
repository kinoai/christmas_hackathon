import shutil
import pandas as pd
from os import listdir, mkdir
from os.path import join
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-trainPath", "--trainPath", type=str, default = "data/train")
parser.add_argument("-trainLabels", "--trainLabels", type=str, default = "data/train.csv")
args = parser.parse_args()


def createFolders(directory, labels):
    for breed in labels["breed"]:
        try:
            mkdir(join(directory, breed))
        except FileExistsError:
            continue

def moveFiles(directory, labels):
    for id, breed in zip(labels["image_id"], labels["breed"]):
        shutil.move(join(directory, id + ".jpg"), join(directory + breed + "/", id + ".jpg"))


trainLabels = pd.read_csv(args.trainLabels)
createFolders(args.trainPath, trainLabels)
moveFiles(args.trainPath, trainLabels)
