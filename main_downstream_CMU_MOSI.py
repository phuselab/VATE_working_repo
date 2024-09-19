import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from   utils import *
import datetime
import json
#from swin_transformer import SwinTransformer3D
#from torchvision.models.video import mvit_v1_b
from   video_ts import Downstream_model_finetuned_regressive, Downstream_model_regressive
from transformers import VivitImageProcessor, BertTokenizer
from CMU_MOSI import CMU_MOSIDataset
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
from   train_test import train_test_contrastive, train_test_downstream_regressive
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    def __init__(self):
        self.TASK_NAME = "Emotion"
        self.ROOT = "/var/data/student_home/agnelli/VATE"
        self.DATASET = CMU_MOSIDataset
        self.DATASET_NAME = "CMU_MOSI"
        self.OUTPUT_DIR = f"{self.ROOT}/output/{self.DATASET_NAME}"

        self.DATASET_ARGS = {
            "dataset_name": self.DATASET_NAME,
            "data_path": "/var/data/CMU_MOSI/processed", #to be updated
            "store": True,
            "shuffle": True,
            "pkl_fname": f"{self.DATASET_NAME}_data_frame.pkl",
        }
        self.MEDIA = {
            "Text": {"class": Text, "ext": "txt", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_text_media_frame.pkl"},
            "Video": {"class": Video, "ext": "mp4", "store_info": False, "store": True, "pkl_fname": f"{self.DATASET_NAME}_video_media_frame.pkl"},    
}
        
class LoaderDataset(Dataset):
    def __init__(self, x_fold_video, x_fold_text, y_fold):
        self.x_fold_video = x_fold_video
        self.x_fold_text = x_fold_text
        self.y_fold = y_fold

    def __len__(self):
        return len(self.x_fold_video)

    def __getitem__(self, idx):
        item_video = self.x_fold_video[idx]
        item_text = self.x_fold_text[idx]
        label = self.y_fold[idx]
        return item_video, item_text, label

class  main():
    def __init__(self, config, num_epochs, batch_size, learning_rate, store):
        self.config = config
        self.store = store
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def training(self):
        image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_path = os.path.join(self.config.ROOT, "output/VATE/best_model_contrastive.pt")
        model = Downstream_model_finetuned_regressive(200, model_path)
        # model = Downstream_model_regressive(200)

        model_video = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
        model_text = BertModel.from_pretrained("bert-base-uncased")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        CMU_MOSI = CMU_MOSIDataset(self.config, ext = 'mp4', verbose=1)
        video_media = Video(
                    self.config, dataset=CMU_MOSI, filename=self.config.MEDIA["Video"]["pkl_fname"], store=self.config.MEDIA["Video"]["store"], store_info=config.MEDIA["Video"]["store_info"], verbose=1
                )
        CMU_MOSI = CMU_MOSIDataset(config, ext = 'txt', verbose=1)
        text_media = Text(
                    self.config, dataset=CMU_MOSI, filename=self.config.MEDIA["Text"]["pkl_fname"], store=self.config.MEDIA["Text"]["store"], store_info=config.MEDIA["Text"]["store_info"], verbose=1
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

        loader_video = []
        loader_text = []

        if self.store:
            for i in trange(CMU_MOSI.size()):
                        # if i<4:
                            video_media.load_video_frames(index = i)
                            item = video_media.frames
                            item = frame_resampling_np(np.array(item), 32)
                            item = image_processor(list(item), return_tensors="pt")
                            with torch.no_grad():
                                item["pixel_values"] = item["pixel_values"].squeeze(1)
                                item = model_video(**item).logits.squeeze(0)
                            loader_video.append(item)

                            text_media.load_text(index = i)
                            item_txt = torch.tensor([text_tokenizer.encode(text_media.text)])
                            with torch.no_grad():
                                item_txt = model_text(item_txt).pooler_output.squeeze(0)
                            loader_text.append(item_txt)
                            
            pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "loader_video.pkl")
            print("Storing loader video...")
            with open(pathout, 'wb') as handle:
                pickle.dump(loader_video, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Loader video stored into file: " + pathout)
            pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "loader_text.pkl")
            print("Storing loader text...")
            with open(pathout, 'wb') as handle:
                pickle.dump(loader_text, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Loader audio stored into file: " + pathout)
        else:
            print("Loading loader video...")
            pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "loader_video.pkl")
            with open(pathout, 'rb') as handle:
                loader_video = pickle.load(handle)
            pathout = os.path.join(self.config.ROOT, self.config.OUTPUT_DIR, "loader_text.pkl")
            with open(pathout, 'rb') as handle:
                loader_text = pickle.load(handle)

        for j in range(5):
            x_train_loader, x_test_loader, y_train_loader, y_test_loader =  CMU_MOSI.train_test_split(five_fold=j)

            max_frame = 200
            train_loader_video = []
            train_loader_audio = []

            for i in trange(len(x_train_loader)):
                train_loader_video.append(loader_video[x_train_loader[i]])
                train_loader_audio.append(loader_text[x_train_loader[i]])

            test_loader_video = []
            test_loader_audio = []
            
            for i in trange(len(x_test_loader)):
                test_loader_video.append(loader_video[x_test_loader[i]])
                test_loader_audio.append(loader_text[x_test_loader[i]])

            train_loader = LoaderDataset(train_loader_video, train_loader_audio, y_train_loader)
            test_loader = LoaderDataset(test_loader_video, test_loader_audio, y_test_loader)

            train_loader = DataLoader(train_loader, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_loader, batch_size=self.batch_size, shuffle=True)

            Training = train_test_downstream_regressive(optimizer, criterion, device, self.config)
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

        output_path = config.ROOT + "results/downstream_finetuned_CMU-MOSI.json"
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