from torchvision import datasets,transforms
import torch
from torch import optim
from torch import nn
import torch.nn.functional as func

def get_transforms(train_dir, valid_dir, test_dir): 
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])


    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # datasets
    image_datasets = dict()
    image_datasets['training_data'] = datasets.ImageFolder(train_dir, transform=training_transforms)
    image_datasets['testing_data'] = datasets.ImageFolder(test_dir, transform=testing_transforms)
    image_datasets['validation_data'] = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    # dataloaders 
    dataloaders = dict()
    dataloaders['trainingloader'] = torch.utils.data.DataLoader(image_datasets['training_data'], batch_size=32, shuffle=True)
    dataloaders['testingloader'] = torch.utils.data.DataLoader(image_datasets['testing_data'], batch_size=32, shuffle=True)
    dataloaders['validationloader'] = torch.utils.data.DataLoader(image_datasets['validation_data'], batch_size=32, shuffle=True)
    
    return image_datasets, dataloaders