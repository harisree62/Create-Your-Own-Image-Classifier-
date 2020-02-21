import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
import torch.nn.functional as func
import torchvision
import torchvision.models as models
from torchvision import models
import matplotlib.pyplot as plt 
from PIL import Image
from collections import OrderedDict
import json

from arguments import get_input_args
from get_transforms import get_transforms




def training_model(args, device):
    
    arch = args.arch
    data_dir = args.data_dir
    epochs = args.epochs
    save_dir = args.save_dir
    
    # getting data path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
        
    # Device CPU or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # architectures list
    if (arch == 'vgg16' ): 
        model = models.vgg16(pretrained=True)
    else:
        print("please enter vgg16 ")
    
    for param in model.parameters():
        param.requires_grad = False    
            
    # define classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,4096)),
        ('relu1',nn.ReLU()),
        ('FC2',nn.Linear(4096,2048)),
        ('relu2',nn.ReLU()),
        ('fc3',nn.Linear(2048,102)),  #102 images outputs
        ('output',nn.LogSoftmax(dim=1))
        ]))
    
        
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)
    print("Training started.")
    
    # Start Classifier training    
    image_datasets, dataloaders = get_transforms(train_dir, valid_dir, test_dir)
    
    print("loaded datasets successfully")      
    
    steps = 0
    run_loss = 0
    print_all = 20
    print("Started trainer.......")
    for epoch in range(epochs):
        for inputs, labels in dataloaders['trainingloader']:
            steps += 1
            #print("1")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            #print("2")
            # loss and accuracy        

            if steps % print_all == 0:
                test_loss = 0
                accuracy = 0
                #evaluate the model
                model.eval()
                #print("3")

                with torch.no_grad():
                    for inputs2, labels2 in dataloaders['validationloader']:
                        inputs2, labels2 = inputs2.to(device), labels2.to(device)
                        output = model.forward(inputs2)
                        group_loss = criterion(output, labels2)
                        test_loss += group_loss.item()
                        #print("4")

                        p = torch.exp(output)
                        top_p, top_class = p.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                
                print("Epoch: {}/{}\t".format(epoch+1,epochs),
                 "Loss: {:.3f}\t".format(run_loss/print_all),
                 "Validated loss: {:.3f}\t".format(test_loss/len(dataloaders['validationloader'])),
                 "Validated Accuracy: {:.3f}%\t".format(accuracy/len(dataloaders['validationloader'])*100))
                
                run_loss = 0
                model.train()           
    print("Training completed successfully")
        
    return model,optimizer,classifier            
 
def testing(model, testingloader, device):
    test_loss = 0
    accuracy = 0
    model.eval()    
    model.to(device)
    
    with torch.no_grad():
        
        for inputs3, labels3 in testingloader:
            #steps_test +=1
            inputs3, labels3 = inputs3.to(device), labels3.to(device)
            logps = model(inputs3)
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels3.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
   print(f"Test accuracy: {accuracy/len(testingloader):.3f}")
    
def save_checkpoint(training_data, epochs, trainingloader, model, optimizer, classifier, arch):
    model.class_to_idx = training_data.class_to_idx
    model.epochs = epochs
    checkpoint = {'input_size': [3, 224, 224],
                'batch_size': trainingloader.batch_size,
                'output_size': 102,
                'state_dict': model.state_dict(),
                'optimizer_dict':optimizer.state_dict(),
                'class_to_idx': model.class_to_idx,
                'epoch': model.epochs,
                'classifier': classifier, 
                'arch': arch}
    torch.save(checkpoint, 'image_checkpoint.pth')
    return checkpoint
    
def main():
    
    
    # Get arguments
    args_values = get_input_args()
    arch = args_values.arch
    data_dir = args_values.data_dir
    learning_rate = args_values.learning_rate
    epochs = args_values.epochs
    save_dir = args_values.save_dir
    device = args_values.device
    
    #print arguments received
    print("Data Directory1:", data_dir)
    print("epochs:", epochs)
    print("save_dir:", save_dir)
    print("arch:", arch)
    print("device:", device)
    print("learning_rate:", learning_rate)

    # set data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # transforms
    image_datasets, dataloaders = get_transforms(train_dir, valid_dir, test_dir)
    print("Datasets transforms completed")
    
    # Cat_to_name
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
      
   #print(cat_to_name)
        
    print("Cat_to_name completed")
    
      
    model, optimizer, classifier = training_model(args_values, device)
    
    print("started Testing")
    
    testing(model, dataloaders['testingloader'], device)
    
    print("Testing Completed")
    print("started saving checkpoint")

    checkpoint = save_checkpoint(image_datasets['training_data'],epochs,dataloaders['trainingloader'],model,optimizer, classifier, arch)
    
    print ("Checkpoint saved")
    
        
if __name__ == '__main__':
    main()
