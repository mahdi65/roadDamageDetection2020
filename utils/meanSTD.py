import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# train_set = datasets.CIFAR10(root='dataset/',train=True,transform = transforms.ToTensor(),download=True)
# train_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
# custom dataset 
# custom_train_set = datasets.CIFAR10(root='dataset/',train=True,transform = transforms.ToTensor(),download=True)
# custom_train_loader = DataLoader(dataset=custom_train_set,batch_size=64,shuffle=True)
custom_train_set = datasets.CocoDetection(root = "./data/train",
                                annFile = "./data/annotations/instances_train.json",
                        transform=transforms.ToTensor())
custom_train_loader = DataLoader(dataset=custom_train_set,batch_size=1,shuffle=True)
custom_val_set = datasets.CocoDetection(root = "./data/val",
                                annFile = "./data/annotations/instances_val.json",
                        transform=transforms.ToTensor())
custom_val_loader = DataLoader(dataset=custom_val_set,batch_size=1,shuffle=True)


# custom_test_set = datasets.CocoDetection(root = "../data/test1/",
#                                 annFile = "../data/annotations/instances_val.json",
#                         transform=transforms.ToTensor())
# custom_test_set = datasets.ImageFolder(
#         root="../data/test1",
#         transform=transforms.ToTensor()
#     )
# custom_test_loader = DataLoader(custom_test_set,batch_size=1,shuffle=True)


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0,0,0

    for data, _ in tqdm(loader) :
        channels_sum += torch.mean(data,dim=[0,2,3]) 
        channels_sqrd_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1 
    
    
    mean = (channels_sum/num_batches)
    std = (channels_sqrd_sum/num_batches - mean**2)**0.5

    return mean,std

m , s = get_mean_std(custom_train_loader)
print(m)
print(s)
m , s = get_mean_std(custom_val_loader)
print(m)
print(s)

# m , s = get_mean_std(custom_test_loader)
# print(m)
# print(s)
