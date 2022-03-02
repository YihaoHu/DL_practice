from cgi import test
from random import shuffle
from ImageData import CatsDogsDataset
from model import Resnet50
import os
import torch
from torchvision import transforms
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from PIL import Image

if __name__ == "__main__":
    #data preprocess
    data_trans = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    img_list = os.listdir('train/')
    dataset = CatsDogsDataset(img_list=img_list, transform=data_trans)

    kfold = KFold(n_splits=5, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Resnet50(Pretrained=True).model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 5

    for epoch in range(1, num_epochs + 1):
        print(f'epoch: {epoch}')
        print('-'*10)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            print(f'Fold {fold}')
            print('-'*5)

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=test_subsampler)
            dataloaders = {'train': trainloader, 'test': testloader}

            #train mode
            model.train()

            running_loss = 0
            accuracy = 0

            for inputs, labels in dataloaders['train']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0) #batch size
                accuracy += torch.sum(preds == labels)
            
            lr_scheduler.step()

            epoch_loss = running_loss / len(train_ids)
            epoch_acc = accuracy / len(train_ids)

            print('Training,  Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            #test mode
            model.eval()
            
            running_loss = 0.0
            accuracy = 0
            with torch.no_grad():
                for inputs, labels in dataloaders['test']:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0) #batch size
                    accuracy += torch.sum(preds == labels)

            epoch_loss = running_loss / len(test_ids)
            epoch_acc = accuracy / len(test_ids)

            print('Testing,  Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    #save model 
    torch.save(model, 'model.pt')

    #submission output
    test_imgs = os.listdir('test/')
    labels = {}
    with torch.no_grad():
        for test_img in test_imgs:
            img = Image.open('test/'+test_img)
            img = data_trans(img)
            img = img.unsqueeze(0)
            img = img.to(device)
            
            model.eval()
            output = model(img)
            pred = torch.nn.functional.softmax(output, dim=1)[:, 1].tolist()
            labels[int(test_img.split('.')[0])] = pred[0]

    res = []
    for i in sorted(labels.keys()):
        res.append(labels[i])
                
    df = {'id':sorted(labels.keys()), 'label':res}
    df = pd.DataFrame(df)
    df.to_csv('submit.csv', index=False)






