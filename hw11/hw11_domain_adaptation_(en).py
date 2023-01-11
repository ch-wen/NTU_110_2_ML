
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
 
source_transform = transforms.Compose([
    # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
    transforms.Grayscale(),
    # cv2 do not support skimage.Image, so we transform it to np.array, 
    # and then adopt cv2.Canny algorithm.
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # Transform np.array back to the skimage.Image.
    transforms.ToPILImage(),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # Turn RGB to grayscale.
    transforms.Grayscale(),
    # Resize: size of source data is 32x32, thus we need to 
    #  enlarge the size of target data from 28x28 to 32x32。
    transforms.Resize((32, 32)),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])
 
source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data_5000', transform=target_transform)
 
source_dataloader = DataLoader(source_dataset, batch_size=1, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=1, shuffle=True)

"""# Model

Feature Extractor: Classic VGG-like architecture

Label Predictor / Domain Classifier: Linear models.
"""

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x
feature_extractor = FeatureExtractor().cuda()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

"""## Step1: Load checkpoint and evaluate to get extracted features"""

# Hints:
# Set features_extractor to eval mode
# Start evaluation and collect features and labels
#feature_extractor.eval()
#X = feature_extractor(source)
"""## Step2: Apply t-SNE and normalize"""


feature_extractor.eval()
feature_extractor.load_state_dict(torch.load("extractor_model_1.bin"))

for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
    source_data = source_data.cuda()
    #source_label = source_label.cuda()
    target_data = target_data[:len(source_data)].cuda()
    
    mix_data = torch.cat([source_data, target_data], dim=0)
    domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
    domain_label[:source_data.shape[0]] = 1
    
    if(i==0):
        domain_feature = feature_extractor(mix_data)
        #source_feature = feature_extractor(source_data)
        d_label = domain_label
        #s_label = source_label
    else:
        d_feature = feature_extractor(mix_data)
        domain_feature = torch.cat([domain_feature, d_feature], dim=0)
        #s_feature = feature_extractor(source_data)
        #source_feature = torch.cat([source_feature, s_feature], dim=0)
        #t_feature = feature_extractor(target_data)
        
        d_label = torch.cat([d_label, domain_label], dim=0)
        #s_label = torch.cat([s_label, source_label], dim=0)
        
# process extracted features with t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(domain_feature.cpu().detach())

# Normalization the processed features 
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

"""## Step3: Visualization with matplotlib"""
plt.figure()
for i in range(X_norm.shape[0]):
    if(d_label[i]==1):
        color='r'
    if(d_label[i]==0):
        color='b'
    '''if(s_label[i]==0):
        color="r"
    if(s_label[i]==1):
        color="b"
    if(s_label[i]==2):
        color="g"
    if(s_label[i]==3):
        color="c"
    if(s_label[i]==4):
        color="y"
    if(s_label[i]==5):
        color="m"
    if(s_label[i]==6):
        color="k"
    if(s_label[i]==7):
        color="orange"
    if(s_label[i]==8):
        color="tan"
    if(s_label[i]==9):
        color="cyan"
'''
    plt.scatter(X_norm[i,0], X_norm[i,1], s=5, color=color)
plt.title('Q2 Early stage')
plt.xticks([0,1]), plt.yticks([0,1])
    
# Data Visualization
# Use matplotlib to plot the distribution
#plt.plot(X_norm)
#plt.show()
# The shape of X_norm is (N,2)
"""# Training Statistics

- Number of parameters:
  - Feature Extractor: 2, 142, 336
  - Label Predictor: 530, 442
  - Domain Classifier: 1, 055, 233

- Simple
 - Training time on colab: ~ 1 hr
- Medium
 - Training time on colab: 2 ~ 4 hr
- Strong
 - Training time on colab: 5 ~ 6 hrs
- Boss
 - **Unmeasurable**

# Learning Curve (Strong Baseline)
* This method is slightly different from colab.

![Loss Curve](https://i.imgur.com/vIujQyo.png)

# Accuracy Curve (Strong Baseline)
* Note that you cannot access testing accuracy. But this plot tells you that even though the model overfits the training data, the testing accuracy is still improving, and that's why you need to train more epochs.

![Acc Curve](https://i.imgur.com/4W1otXG.png)

# Q&A

If there is any problem related to Domain Adaptation, please email to b08901058@ntu.edu.tw / mlta-2022-spring@googlegroups.com。
"""

