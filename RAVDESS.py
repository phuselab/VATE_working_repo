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

class RAVDESSDataset(DatasetFS):
    """
    Class describing the dataset RAVDESS.

    Filename example: 02-01-06-01-02-01-12.mp4
                      M -V- E -E -S- R -A
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    - Vocal channel (01 = speech, 02 = song).
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    - Emotional intensity (01 = normal, 02 = strong). There is no strong intensity for the 'neutral' emotion.
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    - Repetition (01 = 1st repetition, 02 = 2nd repetition).
    - Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    """

    def __init__(self, args: Args, ext: str, verbose=0, task = None):
        # super class's constructor
        super().__init__(args, ext, verbose)

        self.info_path = os.path.join(args.DATASET_ARGS["data_path"], "info.txt")
        self.task = task

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
        emotions = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
        classes = []
        actors = []
        for fname in self.data_frame["filename"]:
            fname = fname.split(".")[-2]
            s = fname.split("-")
            if self.task == "emot_int":
                classes.append(int(s[2])*int(s[3])-1)
            else:
                classes.append(int(s[2])-1)
            actors.append(s[6])

        # replaces classes in the serie
        self.data_frame["class"] = classes
        self.classes = classes
        # add actors
        self.data_frame["actor"] = actors
        self.actors = actors

    def train_test_split(self, train_fold: list = None, train_perc: int = None, leave_actor: int = None, five_fold=0) -> list[list]:
        """ "
        Builds the training and test sets.

        Args:
           - train_perc: percentage in [0, 100] for the training set, the complement is for test set
           - leave_actor: number of the actor to be left out of the training test
           - five_fold: number of the fold to be left out of the training set

        Return:
           - X_train_list: training list of actors
           - X_test_list: test list of actors
           - y_train_list: training class labels
           - y_test_list: test class labels
        """
        Fold_0 = ["02", "05", "14", "15", "16"]
        Fold_1 = ["03", "06", "07", "13", "18"]
        Fold_2 = ["10", "11", " 12", "19", "20"]
        Fold_3 = ["08", "17", "21", "23", "24"]
        Fold_4 = ["01", "04", "09", "22"]
        Fold = [Fold_0, Fold_1, Fold_2, Fold_3, Fold_4]

        try:
            actors = self.get_data_serie("actor")
        except:
            warn(f"Exec method set_classes() before!")

        # collect data in train, test lists
        X_train_list, X_test_list = [], []
        y_train_list, y_test_list = [], []

        # use percentage to split actors
        if train_fold is None and train_perc is not None:
            if int(train_perc) > 1 or int(train_perc) < 0:
                warn(f"The value for {train_perc} must be in [0,1]")
            else:
                actor_uniq = np.sort(list(set(actors)))
                random.shuffle(actor_uniq)
                train_fold = actor_uniq[0 : int(train_perc * len(actor_uniq))]
        else:
            if leave_actor:
                actor_uniq = np.sort(list(set(actors)))
                random.shuffle(actor_uniq)
                mask = actor_uniq != leave_actor
                train_fold = np.sort(actor_uniq[mask])
            else:
                if five_fold > -1:
                    actor_uniq = np.sort(list(set(actors)))
                    random.shuffle(actor_uniq)
                    train_fold = np.sort([item for item in actor_uniq if item not in Fold[five_fold]])
                else:
                    warn(f"A splitting choice must be made")
        # use a given fold to split actors
        if train_fold is not None:
            for i in range(self.size()):
                # print(actors[i],train_fold )
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
