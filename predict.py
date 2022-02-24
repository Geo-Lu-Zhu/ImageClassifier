from os import listdir
import json
import argparse
import matplotlib.pyplot as plt
import torch 
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import pandas as pd
import PIL
import seaborn as sb

# Initiate variables with default values
checkpoint = 'my_checkpoint.pth'
filepath = 'cat_to_name.json'    
arch=''
image_path = 'flowers/test/100/image_07896.jpg'
topk = 5

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint', action='store',type=str, help='Name of trained model to be loaded and used for predictions.')
parser.add_argument('-i','--image_path',action='store',type=str, help='Location of image to predict e.g. flowers/test/class/image')
parser.add_argument('-k', '--topk', action='store',type=int, help='Select number of classes you wish to see in descending order.')
parser.add_argument('-j', '--json', action='store',type=str, help='Define name of json file holding class names.')
parser.add_argument('-g','--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

def load_model(checkpoint_path):
        # Load the saved file
    checkpoint = torch.load(checkpoint_path)      
    # Download pretrained model
    model = models.vgg16(pretrained=True);    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])    
    return model


def process_image(image):
    test_image = PIL.Image.open(image)
    # Get original dimensions
    orig_width, orig_height = test_image.size
    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]        
    test_image.thumbnail(size=resize_size)
    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))
    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!
    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)    
    return np_image

def predict(image_path, model, top_k=5):    
     # No need for GPU on this part (just causes problems)
    #model.to("cpu")    
    # Set model to evaluate
    model.eval();
    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to("cpu")
    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)
    # Convert to linear scale
    linear_probs = torch.exp(log_probs)
    # Find the top results
    top_probs, top_labels = linear_probs.topk(top_k)    
    # Detatch all of the details
    top_probs = top_probs.detach().numpy().tolist()
    top_labels = top_labels.tolist()  
    # Convert to classes
    labels = pd.DataFrame({'class':pd.Series(model.class_to_idx),'flower_name':pd.Series(cat_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_labels[0]]
    labels['predictions'] = top_probs[0]   
    return labels
    
    

model = load_model(checkpoint) 

print('-' * 40)
print('This is the model used for the prediction: \n')
print(model)
print('-' * 40)
input("When you are ready - press Enter to continue to the prediction.")
labels = predict(image_path,model,topk)
print('-' * 40)
print(labels)
print('-' * 40)