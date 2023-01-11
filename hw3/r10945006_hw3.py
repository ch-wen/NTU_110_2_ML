# Import necessary packages.
from asyncio import selector_events
import numpy as np
import pandas as pd
import torch
import os
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.models as models

# This is for the progress bar.
from tqdm.auto import tqdm
import random
from random import shuffle
import ttach as tta
import argparse


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

batch_size = 240
_exp_name = "train60k"
_dataset_dir = "./food11"
myseed = 6666  # set a random seed for reproducibility
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# The number of training epochs and patience.
n_epochs = 100000
patience = 20 # If no improvement in 'patience' epochs, early stop
augment = 60000
split_ratio = 0.8
fold = 0


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# Image preprocessing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    normalize
])


train_tfm = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomCrop((128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.12, saturation=0.12, contrast=0.12),
    transforms.RandomAffine(degrees=18, translate=(0.15, 0.15), scale=(0.8, 1.2)),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    normalize
])



def preprocess(path, fold=fold):
    labels = {}
    groups = [[] for _ in range(11)]
    val_filenames = []
    train_filenames = []
    files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
    for idx, file in enumerate(files):
        label = int(file.split("\\")[-1].split("_")[0])
        if label not in labels:
            labels[label] = 0
        if idx % int(augment//(augment*(1-split_ratio))) == fold:
            val_filenames.append([file, file])
        else:
            groups[label].append([file, file])
            labels[label] += 1
    print('\n==> Raw data')
    print(labels)
    # print('val length: ', len(val_filenames))

    print('\n==> Training data augmentation')
    for groupid in tqdm(range(11)):
        if labels[groupid] < augment:
            left = augment-labels[groupid]
            mix = []
            while len(mix) < left:
                selected = random.sample(groups[groupid], 2)
                if selected not in mix:
                    mix.append([selected[0][0], selected[1][0]])
            for item in mix:
                groups[groupid].append(item)

    newLabels = {}
    for group in groups:
        for item in group:
            label = int(item[0].split("\\")[-1].split("_")[0])
            if label not in newLabels:
                newLabels[label] = 0
            newLabels[label] += 1
    print('After balance')
    print(newLabels)

    for group in groups:
        for idx, item in enumerate(group):
            train_filenames.append(item)
    shuffle(train_filenames)
    shuffle(val_filenames)
    return train_filenames, val_filenames


class FoodDataset(Dataset):

    def __init__(self,filenames,tfm=None):
        super(FoodDataset).__init__()
        self.filenames = filenames
        self.transform = tfm
  
    def __len__(self):
        return len(self.filenames)
  
    def __getitem__(self,idx):
        fname = self.filenames[idx]
        if fname[0] == fname[1]:
            im = Image.open(fname[0])
        else:
            im1 = Image.open(fname[0])
            im1 = im1.resize((128, 128))
            im2 = Image.open(fname[1])
            im2 = im2.resize((128, 128))
            mix = random.random()
            im = Image.blend(im1, im2, alpha=mix)

        if self.transform is not None:
            im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname[0].split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im, label

class TestDataset(Dataset):

    def __init__(self, path, tfm=None, files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        if self.transform is not None:
            im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()

    if args.mode == 'train':
        train_filenames, val_filenames = preprocess(os.path.join(_dataset_dir,"training_all"))
        
        # Construct datasets.
        # The argument "loader" tells how torchvision reads the data.
        train_set = FoodDataset(train_filenames, tfm=train_tfm)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        valid_set = FoodDataset(val_filenames, tfm=test_tfm)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        print(f'\n==> Total:\nTraining data: {len(train_set)} items')
        print(f'Validate data: {len(valid_set)} items')

        # Initialize a model, and put it on the device specified.
        model = models.resnext50_32x4d(pretrained=False, num_classes=11)
        
        model = model.to(device)

        # For the classification task, we use cross-entropy as the measurement of performance.
        criterion = nn.CrossEntropyLoss()

        # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Initialize trackers, these are not parameters and should not be changed
        stale = 0
        best_acc = 0

        for epoch in range(n_epochs):

            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()


            # These are used to record information in training.
            train_loss = []
            train_accs = []

            for batch in tqdm(train_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()
                #print(imgs.shape,labels.shape)

                # Forward the data. (Make sure data and model are on the same device.)
                logits = model(imgs.to(device))

                # Calculate the cross-entropy loss.
                # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                loss = criterion(logits, labels.to(device))
                

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer.zero_grad()

                # Compute the gradients for parameters.
                loss.backward()

                # Clip the gradient norms for stable training.
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

                # Update the parameters with computed gradients.
                optimizer.step()

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                train_loss.append(loss.item())
                train_accs.append(acc)
                
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            scheduler.step()

            # Print the information.
            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            model_eval = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(96, 96))
            model_eval.eval()


            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):

                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                #imgs = imgs.half()

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model_eval(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                #break

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # Print the information.
            # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


            # update logs
            if valid_acc > best_acc:
                with open(f"./{_exp_name}_log.txt","a"):
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
            else:
                with open(f"./{_exp_name}_log.txt","a"):
                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


            # save models
            if valid_acc > best_acc:
                print(f"Best model found at epoch {epoch+1}, saving model")
                torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
                best_acc = valid_acc
                stale = 0
            else:
                stale += 1
                print(f"Early stop encounter {stale}/{patience}")
                if stale >= patience:
                    print(f"No improvment {patience} consecutive epochs, early stopping")
                    print(f"Best model performance: {best_acc:.5f}")
                    torch.save(model.state_dict(), f"{_exp_name}_final.ckpt")
                    break

    if args.mode == 'test':
        test_set = TestDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
        test_loader = DataLoader(test_set, batch_size=batch_size//2, shuffle=False, num_workers=4, pin_memory=True)
        print('test', len(test_set))

        # model_best = models.resnext50_32x4d()
        # num_ftrs = model.fc.in_features
        # model_best.fc = nn.Linear(num_ftrs, 11)

        """
        Good models: 
        train50k: 0.895
        train60k: 0.933
        """
        model_best = models.resnext50_32x4d(pretrained=False, num_classes=11)
        # model_best = Classifier(backbone).to(device)
        # model = MODEL( num_classes = 11 , senet154_weight = None, multi_scale = True, learn_region=True)
        model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
        model_best = tta.ClassificationTTAWrapper(model_best, tta.aliases.five_crop_transform(96, 96))
        model_best.eval()

        model_best.to(device)

        prediction = []
        with torch.no_grad():
            for data,_ in tqdm(test_loader):
                test_pred = model_best(data.to(device))
                test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
                prediction += test_label.squeeze().tolist()

        #create test csv
        def pad4(i):
            return "0"*(4-len(str(i)))+str(i)
        df = pd.DataFrame()
        df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
        df["Category"] = prediction
        df.to_csv("submission.csv",index = False)