import matplotlib.pyplot as plt
import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import PIL
import seaborn as sb
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


def train_transformer(train_dir):
   traindata_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                     transforms.RandomResizedCrop(224), 
                                    transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
   trainimage_datasets = datasets.ImageFolder(train_dir, transform = traindata_transforms)
   return trainimage_datasets

def test_transformer(test_dir):
    testdata_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    testimage_datasets = datasets.ImageFolder(test_dir, transform = testdata_transforms)
    return testimage_datasets

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size = 32,shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size = 32)
    return loader


def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

def create_model():

    model = models.vgg16(pretrained = True);
    model.name = "vgg16"

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, 4096), 
        nn.ReLU(), 
        nn.Linear(4096, 1024), 
        nn.ReLU(), 
        nn.Dropout(p = 0.5), 
        nn.Linear(1024, 102), 
        nn.LogSoftmax(dim = 1))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.0001)    
    
    return model, criterion, optimizer

def trainer(model, trainloader, testloader, device, criterion, optimizer):
    
    epochs = 2
    steps = 0
    running_loss = 0
    print_every = 8
    print("Training process initializing .....")
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    # Check on the test set
            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in testloader:

                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model(inputs)
                        loss = criterion(logps, labels)
                        test_loss += loss.item()

                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim = 1)
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))

                    print(f"Epoch {epoch+1}/{epochs}.. "
                         f"Train loss: {running_loss/print_every:.3f}.. "
                         f"Test loss: {test_loss/len(testloader):.3f}.. "
                         f"Test accuracy: {accuracy/len(testloader):.3f}")

                    running_loss = 0
                    model.train()  
    return model


def validation(model, testloader, criterion, device):
    print("Validation process initializing .....\n")
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)

            logps = model(inputs)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim = 1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor))

        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
            f"Test accuracy: {accuracy/len(testloader):.3f}")

    return 

def initial_checkpoint(model, train_data):
    
    model.class_to_idx = train_data.class_to_idx
    # Create checkpoint dictionary
    checkpoint = {'architecture': model.name,
                 'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                 'state_dict': model.state_dict()}
    # Save checkpoint
    torch.save(checkpoint, 'my_checkpoint.pth')
    print("Image classifier is saved! \n")
    return
    
    
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = test_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    model, criterion, optimizer = create_model()
    
    device = check_gpu(gpu_arg=args.gpu);
    model.to(device);
    
    trained_model = trainer(model, trainloader, testloader, device, criterion, optimizer)
    
    print("\nTraining process is completed!!")
    
    validation(model, testloader, criterion, device)
   
    initial_checkpoint(trained_model, train_data)
if __name__ == '__main__': main()