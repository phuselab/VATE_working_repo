import torch
import numpy as np
import json
import os
import datetime

class train_test_downstream():
    def __init__(self, optimizer, criterion, device, config):
         pass
         self.optimizer = optimizer
         self.criterion = criterion
         self.device = device
         self.config = config

    def train(self, model, loader):
        model.to(self.device)
        model.train()
        total_loss = 0.0
        total_correct = 0
        i = 0
        for data_video, data_audio, label in loader:
            i += 1
            self.optimizer.zero_grad()
            logits = model(data_video.to(self.device), data_audio.to(self.device))

            pred = logits.argmax(dim=-1)
            true = label.to(self.device)

            correct = (pred == true).sum().item()
            total_correct += correct

            loss = self.criterion(logits, true)

            loss.backward()
            # for name, param in model.named_parameters():
            #      print(name)
            #      print(torch.norm(param.grad))
            self.optimizer.step()

            total_loss += loss.item()

            # Clear GPU memory
            del logits, pred, true
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / len(loader.dataset)
        return model, epoch_loss, epoch_acc

    @torch.no_grad()
    def test(self, model, loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        float_acc = 0.0
        logits_all = []
        for data_video, data_audio, label in loader:
            logits = model(data_video.to(self.device), data_audio.to(self.device))

            pred = logits.argmax(dim=-1)
            true = label.to(self.device)

            correct = (pred == true).sum().item()
            total_correct += correct

            loss = self.criterion(logits, true)

            total_loss += loss.item()

            # Clear GPU memory
            del pred, logits, true
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / len(loader.dataset)
        return epoch_loss, epoch_acc
    
    def training(self, epochs, train_loader, test_loader, model):

        valid_epoch_acc = 0
        valid_epoch_loss = np.inf
        best_train_acc = 0
        best_model = None

        print("number of parameters in the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
                
        for epoch in range(1, epochs):
            current_model, train_epoch_loss, train_epoch_acc = self.train(model, train_loader)
            test_epoch_loss, test_epoch_acc = self.test(current_model, test_loader)

            if train_epoch_acc >= best_train_acc:
                best_model = current_model
                best_epoch = epoch
                best_train_loss = train_epoch_loss
                best_train_acc = train_epoch_acc
                valid_epoch_acc = test_epoch_acc
                valid_epoch_loss = test_epoch_loss

            progress_bar(epoch, epochs, train_epoch_loss, train_epoch_acc)
        
        torch.save(best_model.state_dict(), self.config.OUTPUT_DIR + "/best_model.pt")
        print(f"CURRENT ACCURACY: {valid_epoch_acc:.4f} - LOSS: {valid_epoch_loss:.4f}")
        del model, train_loader, test_loader
        torch.cuda.empty_cache()

        return valid_epoch_acc, valid_epoch_loss


class train_test_downstream_regressive():
    def __init__(self, optimizer, criterion, device, config):
         pass
         self.optimizer = optimizer
         self.criterion = criterion
         self.device = device
         self.config = config

    def train(self, model, loader):
        model.to(self.device)
        model.train()
        total_loss = 0.0
        total_correct = 0
        i = 0
        for data_video, data_text, label in loader:
            i += 1
            self.optimizer.zero_grad()
            logits = model(data_video.to(self.device), data_text.to(self.device)).squeeze(-1)

            pred = logits
            true = label.to(self.device).to(torch.float32)

            correct = (pred == true).sum().item()
            total_correct += correct

            loss = self.criterion(logits, true)

            loss.backward()
            # for name, param in model.named_parameters():
            #      print(name)
            #      print(torch.norm(param.grad))
            self.optimizer.step()

            total_loss += loss.item()

            # Clear GPU memory
            del logits, pred, true
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / len(loader.dataset)
        return model, epoch_loss, epoch_acc

    @torch.no_grad()
    def test(self, model, loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        float_acc = 0.0
        logits_all = []
        for data_video, data_audio, label in loader:
            logits = model(data_video.to(self.device), data_audio.to(self.device)).squeeze(-1)

            pred = logits.argmax(dim=-1)
            true = label.to(self.device).to(torch.float32)

            correct = (pred == true).sum().item()
            total_correct += correct

            loss = self.criterion(logits, true)

            total_loss += loss.item()

            # Clear GPU memory
            del pred, logits, true
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / len(loader.dataset)
        return epoch_loss, epoch_acc
    
    def training(self, epochs, train_loader, test_loader, model):

        valid_epoch_acc = 0
        valid_epoch_loss = np.inf
        best_train_acc = 0
        best_model = None

        print("number of parameters in the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
                
        for epoch in range(1, epochs):
            current_model, train_epoch_loss, train_epoch_acc = self.train(model, train_loader)
            test_epoch_loss, test_epoch_acc = self.test(current_model, test_loader)

            if train_epoch_acc >= best_train_acc:
                best_model = current_model
                best_epoch = epoch
                best_train_loss = train_epoch_loss
                best_train_acc = train_epoch_acc
                valid_epoch_acc = test_epoch_acc
                valid_epoch_loss = test_epoch_loss

            progress_bar(epoch, epochs, train_epoch_loss, train_epoch_acc)
        
        torch.save(best_model.state_dict(), self.config.OUTPUT_DIR + "/best_model.pt")
        print(f"CURRENT ACCURACY: {valid_epoch_acc:.4f} - LOSS: {valid_epoch_loss:.4f}")
        del model, train_loader, test_loader
        torch.cuda.empty_cache()

        return valid_epoch_acc, valid_epoch_loss



class train_test_contrastive():
    """
    Trains and tests a machine learning model using a contrastive learning approach.
    
    The `train_test_contrastive` class provides methods for training and testing a model on a dataset. The `train` method trains the model on a training dataset, while the `test` method evaluates the model on a test dataset. The `training` method orchestrates the training and testing process, keeping track of the best model and reporting the final accuracy and loss.
    
    The class takes an optimizer and a device (e.g. CPU or GPU) as input, and uses them to train and test the model. The training process involves iterating over the training dataset, computing the loss, backpropagating the gradients, and updating the model parameters. The testing process involves evaluating the model on the test dataset and computing the loss and accuracy.
    
    The `progress_bar` function is used to display the training progress during the training process.
    """
    def __init__(self, optimizer, device, scheduler):
         pass
         self.optimizer = optimizer
         self.device = device
         self.scheduler = scheduler

    def train(self, model, loader):
        model.to(self.device)
        model.train()
        total_loss = 0.0
        total_correct = 0
        for data_video, data_audio, data_text in loader:
            self.optimizer.zero_grad()
            _, _, _, loss = model(data_video.to(self.device), data_audio.to(self.device), data_text.to(self.device))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Clear GPU memory
            del loss
            torch.cuda.empty_cache()

        self.scheduler.step(total_loss)
        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / len(loader.dataset)
        return model, epoch_loss, epoch_acc

    @torch.no_grad()
    def test(self, model, loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        for data_video, data_audio, data_text, label in loader:
            self.optimizer.zero_grad()
            _, _, _, loss = model(data_video.to(self.device), data_audio.to(self.device), data_text.to(self.device))
            total_loss += loss.item()

            # Clear GPU memory
            del loss
            torch.cuda.empty_cache()

        epoch_loss = total_loss / len(loader)
        epoch_acc = total_correct / len(loader.dataset)
        return epoch_loss, epoch_acc
    
    def training(self, epochs, train_loader, model):
        
        metrics = {
            "Train": {
                "Loss": [],
                "Accuracy": [],
            },
        }

        best_train_loss = np.inf
        best_train_acc = 0
        best_model = None

        print("number of parameters in the model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
                
        for epoch in range(1, epochs):
            current_model, train_epoch_loss, train_epoch_acc = self.train(model, train_loader)

            if train_epoch_loss <= best_train_loss:
                best_model = current_model
                best_epoch = epoch
                best_train_loss = train_epoch_loss
                best_train_acc = train_epoch_acc

            progress_bar(epoch, epochs, train_epoch_loss, train_epoch_acc)

            # valid_epoch_loss, valid_epoch_acc, valid_epoch_float_acc, batch_logits = test(best_model, test_loader, device, criterion, target_attribute)
            # print(
            #     f"TESTING {len(embed.keys[test_index])} SUBJECTS -> EPOCH: {best_epoch}, Loss: {valid_epoch_loss:.3f}, Accuracy: {valid_epoch_acc:.3f}, Float-acc: {valid_epoch_float_acc:.4f}"
            # )
        

        print(f"CURRENT ACCURACY: {best_train_acc:.4f} - LOSS: {best_train_loss:.4f}")
        torch.save(best_model.state_dict(), "/var/data/student_home/agnelli/new_dataset/results/best_model_contrastive.pt")

        metrics["Train"]["Loss"].append(best_train_loss)
        metrics["Train"]["Accuracy"].append(best_train_acc)
        del model, train_loader
        torch.cuda.empty_cache()
        if not os.path.exists(f"/var/data/student_home/agnelli/new_dataset/results"):
            os.makedirs(f"/var/data/student_home/agnelli/new_dataset/results")

        with open(f"/var/data/student_home/agnelli/new_dataset/results/_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json", "w") as f:
            json.dump(metrics, f, indent=4)


def progress_bar(current, total, loss, accuracy, bar_length=20):
    progress = current / total
    arrow_length = int(round(progress * bar_length))
    arrow = "=" * arrow_length + ">"
    spaces = " " * (bar_length - len(arrow))
    percentage = progress * 100
    print(f"[{arrow}{spaces}] {percentage:.2f}% - Epoch: {current:02d}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}", end="\r", flush=True)
