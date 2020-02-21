import numpy as np
import pandas as pd
import sys
import torch
from torch import optim
from torch import nn
import torch.nn.functional as func
import torchvision
import torchvision.models as models
from torchvision import models,transforms
import matplotlib.pyplot as plt 
from PIL import Image
from collections import OrderedDict
import os
import json

from arguments import get_input_args
from get_transforms import get_transforms

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    model = checkpoint['arch']
    
    if (model == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print("Architecture shoud be vgg16")
        
    model.classifier = checkpoint['classifier']
    
    model.load_state_dict(checkpoint['state_dict'],strict = False)
    
    optimizer = checkpoint['optimizer_dict']
    
    epochs = checkpoint['epoch']
    
    model.class_to_idx = checkpoint['class_to_idx']

    for param in model.parameters():
        param.requires_grad = False   
        
    return model, checkpoint['class_to_idx']

def process_image(path_to_image):
    
    pil_img = Image.open(path_to_image)
    
    width, height = pil_img.size 
    reduce = 224
    left = (width - reduce)/2
    top = (height - reduce)/2
    right = left + reduce
    bottom = top + reduce
    pil_img = pil_img.crop ((left, top, right, bottom))
    npimage = np.array (pil_img)/255 
    # Normalization of values
    npimage -= np.array ([0.485, 0.456, 0.406])
    npimage /= np.array ([0.229, 0.224, 0.225])
    npimage= npimage.transpose ((2,0,1))
    return npimage
    
    
    
    
   
    
def predict(path_to_image, model, device, top_k, class_to_idx):
    # predicting the class of image
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    imag = process_image(path_to_image)
    img_torch = torch.from_numpy(imag).type(torch.cuda.FloatTensor)
    img_torch=img_torch.unsqueeze_(dim=0)
    img_torch = img_torch.float()

    
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
          
    
    probability = func.softmax(output.data,dim=1)
    
    return probability.topk(top_k)



def main():
    
    # get all argument values
    
    args_values = get_input_args()
    
    #retreiving argument values
    
    data_dir = args_values.data_dir
    device = args_values.device
    path_to_image = args_values.path_to_image
    top_k = args_values.top_k
    arch = args_values.arch
    save_dir = args_values.checkpoint_dir
    category_names = args_values.category_names

    # path of image
    
    path_of_image = 'flowers/test/'+path_to_image
    
    print("The data path: \t", data_dir)
    print("Saved Directory: \t", save_dir)
    print("Device: \t", device)
    print("top_k: \t", top_k)
    print("categories list: \t", category_names)
    print("path to image file: \t", path_to_image)
   
    # checking for checkpoint
    if os.path.exists(save_dir):
        print("Checkpoint is available")
        
        # Category to name
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
            
        # load saved checkpoint
        loaded_model, class_to_idx = load_checkpoint(save_dir)
        print("Model loaded")
        probability = predict(path_of_image, loaded_model, device, top_k, class_to_idx)

        
        img_number = path_of_image.split('/')[-2]
        
        actual_image_name = cat_to_name[str(img_number)]
        print("Actual Image Name:", actual_image_name)
        print("Predict Completed")
        probs = np.array(probability[0][0])
        classes = np.array(probability[1][0])
        
        #for prob in probs:            
            #print("Probs:", prob)

        #for cla in classes:
         #   print("classes:", cla)
         

        a = np.array(probability[1][0])
        b = [cat_to_name[str(index+1)] for index in a]
      
        print("Top ",top_k, " classes are as listed:", b)
        print("--------------------Prediction completed--------------------")      
    else:
        
        print("No checkpoint file found")
        
        
        
if __name__ == '__main__':
    main()
