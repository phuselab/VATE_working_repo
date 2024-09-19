import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import *
import datetime
import json
from video_ts import Contrastive_model, Downstream_model, Downstream_model_finetuned
from transformers import VivitImageProcessor, BertTokenizer
from RAVDESS import RAVDESSDataset
from tqdm import trange
from video import Video
from audio import Audio
from text import Text
from torch.utils.data import Dataset
import pickle 
import os
from torch.utils.data import DataLoader
from transformers import VivitForVideoClassification, BertModel, CLIPModel
import torchaudio
# from torch_geometric.data import Data
from train_test import train_test_contrastive, train_test_downstream
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.TASK_NAME = "Emotion"
        self.ROOT = "/var/data/student_home/agnelli/VATE"
        self.DATASET = RAVDESSDataset
        self.DATASET_NAME = "RAVDESS"
        self.OUTPUT_DIR = f"{self.ROOT}/output/{self.DATASET_NAME}"

        self.DATASET_ARGS = {
            "dataset_name": self.DATASET_NAME,
            "data_path": "/var/data/RAVDESS",
            "store": True,
            "shuffle": True,
            "pkl_fname": f"{self.DATASET_NAME}_data_frame.pkl",
        }
        self.MEDIA = {
            "Audio": {"class": Audio, "ext": "wav", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_audio_media_frame.pkl"},
            "Video": {"class": Video, "ext": "mp4", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_video_media_frame.pkl"},    
}
        
class LoaderDataset(Dataset):
    def __init__(self, x_fold_video, x_fold_audio, y_fold):
        self.x_fold_video = x_fold_video
        self.x_fold_audio = x_fold_audio
        self.y_fold = y_fold

    def __len__(self):
        return len(self.x_fold_video)

    def __getitem__(self, idx):
        item_video = self.x_fold_video[idx]
        item_audio = self.x_fold_audio[idx]
        label = self.get_target(self.y_fold[idx])
        return item_video, item_audio, label
    
    def get_target(self, y, num_classes = 8):
        target = 0
        emotions = ["neutral", "happy", "disgust", "angry", "calm", "fearful", "sad", "surprised"]
        for i, emotion in enumerate(emotions):
            if y == emotion:
                target = i
        return target

class  main():
    def __init__(self, config, num_epochs, batch_size, learning_rate, store):
        self.config = config
        self.store = store
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def training(self):
        image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        model_path = os.path.join(config.ROOT, "output/VATE/best_model_contrastive.pt")
        model = Downstream_model_finetuned(200, 8, model_path)

        model_video = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        bundle = torchaudio.pipelines.HUBERT_BASE
        model_audio = bundle.get_model()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        store = self.store
        RAVDESS = RAVDESSDataset(self.config, ext = 'mp4', verbose=1)
        video_media = Video(
                   self.config, dataset=RAVDESS, filename=self.config.MEDIA["Video"]["pkl_fname"], store=self.config.MEDIA["Video"]["store"], store_info=config.MEDIA["Video"]["store_info"], verbose=1
                )
        RAVDESS = RAVDESSDataset(config, ext = 'wav', verbose=1)
        audio_media = Audio(
                    self.config, dataset=RAVDESS, filename=self.config.MEDIA["Audio"]["pkl_fname"], store=self.config.MEDIA["Audio"]["store"], store_info=self.config.MEDIA["Audio"]["store_info"], verbose=1
                )

        metrics = {
                    "Test": {
                        "Mean Loss": [],
                        "Mean Accuracy": [],
                        "Std Loss": [],
                        "Std Accuracy": [],
                    },
                }

        loss = []
        accuracy = []

        for j in range(5):
            x_train_loader, x_test_loader, y_train_loader, y_test_loader = RAVDESS.train_test_split(five_fold=j)

            max_frame = 200
            if store:
                train_loader_video = []
                train_loader_audio = []
                for i in trange(len(x_train_loader)):
                    # if i<4:
                        video_media.load_video_frames(index = x_train_loader[i])
                        item = video_media.frames
                        item = frame_resampling_np(item, 32)
                        item = image_processor(list(item), return_tensors="pt")
                        with torch.no_grad():
                            item["pixel_values"] = item["pixel_values"].squeeze(1)
                            item = model_video(**item).logits.squeeze(0)
                        train_loader_video.append(item)

                        audio_media.load_audio(index = x_train_loader[i])
                        item_audio = audio_media.compute_feature_hubert()
                        with torch.no_grad():
                            item_audio, _ = model_audio.extract_features(item_audio)
                            item_audio = item_audio[-1][0].mean(0)
                        train_loader_audio.append(item_audio)

                test_loader_video = []
                test_loader_audio = []
                for i in trange(len(x_test_loader)):
                    # if i<1:
                        video_media.load_video_frames(index = x_test_loader[i])
                        item = video_media.frames
                        item = frame_resampling_np(item, 32)
                        item = image_processor(list(item), return_tensors="pt")
                        with torch.no_grad():
                            item["pixel_values"] = item["pixel_values"].squeeze(1)
                            item = model_video(**item).logits.squeeze(0)
                        test_loader_video.append(item)

                        audio_media.load_audio(index = x_test_loader[i])
                        item_audio = audio_media.compute_feature_hubert()
                        with torch.no_grad():
                            item_audio, _ = model_audio.extract_features(item_audio)
                            item_audio = item_audio[-1][0].mean(0)
                        test_loader_audio.append(item_audio)


                train_loader = LoaderDataset(train_loader_video, train_loader_audio, y_train_loader)
                test_loader = LoaderDataset(test_loader_video, test_loader_audio, y_test_loader)
                pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "train_loader.pkl")
                print("Storing train loader...")
                with open(pathout, 'wb') as handle:
                    pickle.dump(train_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Train loader stored into file: " + pathout)
                print("Storing test loader...")
                pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "test_loader.pkl")
                with open(pathout, 'wb') as handle:
                    pickle.dump(test_loader, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Test loader stored into file: " + pathout)
            else:
                print("Loading train loader...")
                pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "train_loader.pkl")
                with open(pathout, 'rb') as handle:
                    train_loader = pickle.load(handle)
                print("Loading test loader...")
                pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "test_loader.pkl")
                with open(pathout, 'rb') as handle:
                    test_loader = pickle.load(handle)

            train_loader = DataLoader(train_loader, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_loader, batch_size=self.batch_size, shuffle=True)

            Training = train_test_downstream(optimizer, criterion, device, self.config)
            acc, ls = Training.training(self.num_epochs, train_loader, test_loader, model)

            loss.append(ls)
            accuracy.append(acc)

        mean_loss = np.mean(loss)
        mean_acc = np.mean(accuracy)
        std_loss = np.std(loss)
        std_accuracy = np.std(accuracy)

        metrics["Test"]["Mean Loss"].append(mean_loss)
        metrics["Test"]["Mean Accuracy"].append(mean_acc)
        metrics["Test"]["Std Loss"].append(std_loss)
        metrics["Test"]["Std Accuracy"].append(std_accuracy)

        output_path = config.ROOT + "/results/downstream_finetuned_RAVDESS.json"
        with open(output_path, "w") as f:
                    json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    config = Config()
    store = False
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01
    training = main(config, num_epochs, batch_size, learning_rate, store)
    training.training()