import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,random_split


def process_mnist_dataset(batch_size,validation_split):

    transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))
])

    #The .ToTensor method turns the range of the pixel intensity values from [0,255] -> [0,1]
    #The .Normalize turns this range from [0,1] -> [-1,1]
    
    train_dataset = datasets.MNIST(
        root='./data',  
        train=True,     
        transform=transform,  
        download=True   
    )

    test_dataset = datasets.MNIST(
        root='./data',  
        train=False,    
        transform=transform,
        download=True   
    )
    
    train_size = int((1 - validation_split) * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    
    train_subset, validation_subset = random_split(train_dataset, [train_size, validation_size])
    
    return train_subset,validation_subset,test_dataset