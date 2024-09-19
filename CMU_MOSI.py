import os
from typing import Iterable
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
import random
import numpy as np

from collections import defaultdict
from dataset import DatasetFS
from utils import *

class CMU_MOSIDataset(DatasetFS):
    """
    Class describing the dataset CMU-MOSI dataset. The data are provided in the form:
    1_1_2.4
    where:
    1 is the subject
    1 is the clip
    2.4 is the emotional intensity
    """

    def __init__(self, args: Args, ext: str, verbose=0):
        # super class's constructor
        super().__init__(args, ext, verbose)

        self.info_path = os.path.join(args.DATASET_ARGS["data_path"], "info.txt")

        # set class names
        if args.DATASET_ARGS["store"]:
            self.set_classes()
            # self.shuffle()
            self.store_dataset()
            if self.args.DATASET_ARGS["shuffle"]:
                print("The dataset has been shuffled")
                self.shuffle()
        else:
            if self.args.DATASET_ARGS["shuffle"]:
                print("The dataset has been shuffled")
                self.shuffle()

    # @abstractmethod
    def set_classes(self):
        classes = []
        actors = []
        for fname in self.data_frame["filename"]:
            ext = fname.split(".")[-1]
            fname = fname.split(ext)[0][:-1]
            s = fname.split("_")
            classes.append(np.float32(s[-1]))
            actors.append(s[0])
        

        # replaces classes in the serie
        self.data_frame["class"] = classes
        self.classes = classes
        # add actors
        self.data_frame["actor"] = actors
        self.actors = actors

    def train_test_split(self, five_fold=0) -> list[list]:

        try:
            actors = self.get_data_serie("actor")
        except:
            warn(f"Exec method set_classes() before!")

        actor_uniq = np.sort(list(set(actors)))
        fold= [[],[],[],[],[]]

        for i in range(1,len(actor_uniq)+1):
            fold[i%5].append(str(i))

        # collect data in train, test lists
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        random.shuffle(actor_uniq)
        train_fold = np.sort([item for item in actor_uniq if item not in fold[five_fold]])

        # use a given fold to split actors
        if train_fold is not None:
            for i in range(self.size()):
                if actors[i] in train_fold:
                    X_train_list.append(i)
                    y_train_list.append(self.data_frame.iloc[i, 1])
                else:
                    X_test_list.append(i)
                    y_test_list.append(self.data_frame.iloc[i, 1])

        # final shuffle
        permute = np.random.permutation(len(X_train_list)).tolist()
        X_train_list = np.array(X_train_list)[permute].tolist()
        y_train_list = np.array(y_train_list)[permute].tolist()

        permute = np.random.permutation(len(X_test_list)).tolist()
        X_test_list = np.array(X_test_list)[permute].tolist()
        y_test_list = np.array(y_test_list)[permute].tolist()

        # return
        return X_train_list, X_test_list, y_train_list, y_test_list
