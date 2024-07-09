import os
import copy
import random
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import timm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from prettytable import PrettyTable

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

    def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.strip().split(',')
                    imagePath = os.path.join(images_path, lineItems[0])
                    imageLabel = int(lineItems[1])  # Read a single integer label
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

        indexes = np.arange(len(self.img_list))
        if annotation_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]
            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []
            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.tensor(self.img_label[index], dtype=torch.long)  # Ensure label is a tensor of type long
        if self.augment is not None:
            imageData = self.augment(imageData)
        return imageData, imageLabel

    def __len__(self):
        return len(self.img_list)

# Define data transforms with more augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Paths to dataset
images_path = '/mnt/dfs/jpang12/datasets/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png'
train_file_path = './train.txt'
test_file_path = './test.txt'

# Prepare the dataset
train_dataset = ShenzhenCXR(images_path, train_file_path, transform)
test_dataset = ShenzhenCXR(images_path, test_file_path, transform)

# Define the model
def create_model():
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=2)
    return model

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Training loop
def train(rank, world_size, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    start_time = time.time()
    
    # Data loading time
    load_time = 0
    push_time = 0
    train_time = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        load_start = time.time()
        data, target = data.to(device), target.to(device)
        load_end = time.time()
        
        push_start = load_end
        optimizer.zero_grad()
        output = model(data)
        push_end = time.time()
        
        train_start = push_end
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_end = time.time()
        
        load_time += (load_end - load_start)
        push_time += (push_end - push_start)
        train_time += (train_end - train_start)
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % 10 == 0 and rank == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    
    end_time = time.time()
    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    if rank == 0:
        print(f'Training set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.0f}%)')
    
    return load_time, push_time, train_time, end_time - start_time

# Test loop
def test(rank, world_size, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    
    # Data loading time
    load_time = 0
    push_time = 0
    test_time = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            load_start = time.time()
            data, target = data.to(device), target.to(device)
            load_end = time.time()
            
            push_start = load_end
            output = model(data)
            push_end = time.time()
            
            test_start = push_end
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_end = time.time()
            
            load_time += (load_end - load_start)
            push_time += (push_end - push_start)
            test_time += (test_end - test_start)
    
    end_time = time.time()
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    if rank == 0:
        print(f'\nValidation set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    
    return load_time, push_time, test_time, end_time - start_time

# Distributed training function
def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Prepare the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=16, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16, sampler=test_sampler)
    
    model = create_model().to(rank)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    epoch_data = {
        "Train Data Loading Time": [],
        "Push Train Data to GPU Time": [],
        "Training Time": [],
        "Total Train Time": [],
        "Test Data Loading Time": [],
        "Push Test Data to GPU Time": [],
        "Testing Time": [],
        "Total Test Time": []
    }
    
    for epoch in range(1, 11):
        train_sampler.set_epoch(epoch)
        train_load_time, train_push_time, train_time, total_train_time = train(rank, world_size, model, rank, train_loader, optimizer, criterion, epoch)
        test_load_time, test_push_time, test_time, total_test_time = test(rank, world_size, model, rank, test_loader, criterion)
        scheduler.step()
        
        if rank == 0:
            epoch_data["Train Data Loading Time"].append(train_load_time)
            epoch_data["Push Train Data to GPU Time"].append(train_push_time)
            epoch_data["Training Time"].append(train_time)
            epoch_data["Total Train Time"].append(total_train_time)
            epoch_data["Test Data Loading Time"].append(test_load_time)
            epoch_data["Push Test Data to GPU Time"].append(test_push_time)
            epoch_data["Testing Time"].append(test_time)
            epoch_data["Total Test Time"].append(total_test_time)
    
    if rank == 0:
        results_table = PrettyTable()
        results_table.field_names = ["Step"] + [f"Epoch {i+1}" for i in range(10)]
        for step in epoch_data.keys():
            results_table.add_row([step] + epoch_data[step])
        print(results_table)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)