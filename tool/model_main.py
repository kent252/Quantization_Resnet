
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import os
import sys
import time
import streamlit as st
sys.path.append(os.path.abspath('./model'))

from model import CLSModel
from size_model import print_size_of_model

def train_test_split(data_path):
    # Tạo đối tượng ImageFolder và chia dữ liệu thành tập huấn luyện, tập kiểm tra và tập xác thực
    # Định nghĩa phép biến đổi dữ liệu
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về kích thước 224x224 pixel
        transforms.ToTensor(),  # Chuyển ảnh sang tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Chuẩn hóa ảnh về khoảng [-1, 1]
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=(10, 70)),
        transforms.RandomCrop((200, 200))
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
    ])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về kích thước 224x224 pixel
        transforms.ToTensor(),  # Chuyển ảnh sang tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Chuẩn hóa ảnh về khoảng [-1, 1]
    ])
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    test_dir = os.path.join(data_path, "test")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    return train_dataset, val_dataset, test_dataset


def load_data(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, val_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            st.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Define the validation function
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * correct / len(val_loader.dataset)
    st.write('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, correct, len(val_loader.dataset), val_acc))
    return val_loss, val_acc


def test_model(model, test_loader, device,mode):
    model.to(device)
    # Test model
    test_loss = 0
    correct = 0
    t1 = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    st.write('Test set of' +mode+' : Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
          .format(test_loss, correct, len(test_loader.dataset), accuracy))
    st.write('Test time : ',time.time()-t1)


# Define the main function for training the model
def train_model(pretrain,num_class = 120,num_epochs = 10,data_path = "./data/dog_breed"):
    # Set the device
    st.write("Model start training !!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(device)

    # Load the data
    train_dataset, val_dataset, test_dataset = train_test_split(data_path)

    train_loader, val_loader, test_loader = load_data(train_dataset, val_dataset, test_dataset)

    # Define the model, optimizer and loss function
    # Our initial baseline model which is FP32
    float_model = CLSModel(pretrain,num_class).to(device)
    model_fp32 = float_model
    model_fp32.train()

    # Sets the backend for x86
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepares the model for the next step i.e. calibration.
    model = torch.quantization.prepare_qat(model_fp32, inplace=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Define the training parameters
    best_val_loss = float('inf')
    best_val_acc = 0
    checkpoint_path = './result/checkpoint/last_checkpoint.pt'
    best_checkpoint_path = './result/checkpoint/best_checkpoint.pt'

    # Train from epoch
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path,map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_loss = checkpoint['loss']
        for epoch in range(cur_epoch + 1, num_epochs + 1):
            t1 = time.time()
            with st.empty():
                train(model, device, train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = validate(model, device, val_loader, criterion)
            st.write("EST time per epoch:", time.time() - t1)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
            }
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state, best_checkpoint_path)
            torch.save(state, checkpoint_path)

    # Train from scratch
    else:
        for epoch in range(1, num_epochs + 1):
            t1 = time.time()
            with st.empty():
                train(model, device, train_loader, optimizer, criterion, epoch)
            val_loss, val_acc = validate(model, device, val_loader, criterion)
            st.write("EST time per epoch:", time.time() - t1)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
            }
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state, best_checkpoint_path)
            torch.save(state, checkpoint_path)


    st.write("Training completed!")
    test_model(model, test_loader, device,mode = 'fp32_model')

def inference_quant(pretrain,num_class = 120,data_path = "./data/dog_breed"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(device)

    # Load the data
    train_dataset, val_dataset, test_dataset = train_test_split(data_path)

    train_loader, val_loader, test_loader = load_data(train_dataset, val_dataset, test_dataset)

    # Define the model, optimizer and loss function
    # Our initial baseline model which is FP32
    float_model = CLSModel(pretrain,num_class).to(device)
    checkpoint = torch.load("./result/checkpoint/best_checkpoint.pt",map_location=torch.device(device))
    float_model.load_state_dict(checkpoint['state_dict'], strict=False)
    model_fp32 = float_model
    model_fp32.train()

    # Sets the backend for x86
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepares the model for the next step i.e. calibration.
    # Inserts observers in the model that will observe the activation tensors during calibration
    model = torch.quantization.prepare_qat(model_fp32, inplace=False)

    st.write('Size of fp32 model:')
    print_size_of_model(model)
    num_params_fp32 = sum(p.numel() for p in model.parameters())
    st.write("Number of parameters in the fp32 model:", num_params_fp32)

    #Calibrate process
    batch_images, batch_labels = next(iter(val_loader))
    model(batch_images.to(device))
    model.to('cpu')
    model.eval()

    # Converts the model to a quantized model(int8)
    model_quantized = torch.quantization.convert(model)  # Quantize the model
    num_params = sum(p.numel() for p in model_quantized.parameters())

    # Print the number of parameters in the quantized model
    st.write("Number of parameters in the 8int model:", num_params)
    st.write('Size of fp32 model:')
    print_size_of_model(model_quantized)

    #Test on 8INT model
    test_model(model_quantized, test_loader, 'cpu',mode = 'quant_model' )