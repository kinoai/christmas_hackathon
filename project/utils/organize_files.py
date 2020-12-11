import argparse
import shutil
import sys
from os import mkdir
from os.path import join

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="../data/train")
parser.add_argument("--train_labels", type=str, default="../data/train.csv")
args = parser.parse_args()


def print_(text_to_print):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(text_to_print)


def create_folders(directory, labels):
    for breed in labels["breed"]:
        try:
            mkdir(join(directory, breed))
        except FileExistsError:
            continue


def move_files(directory, labels):
    for image_id, breed in zip(labels["image_id"], labels["breed"]):
        id_path = join(directory + "/", image_id + ".jpg")
        breed_path = join(directory + "/" + breed + "/")

        print_(f"NOW MOVING {image_id} TO NEW HOME AT {breed_path}")
        shutil.move(id_path, breed_path)


train_labels = pd.read_csv(args.train_labels)
create_folders(args.train_path, train_labels)
move_files(args.train_path, train_labels)
