"""
Train model on the poisoned training set
Author: Zhen Xiang
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import json
import sys
from tqdm import tqdm
import torch
import torch.nn as nn

from data_utils import load_data, change_label
from attack_utils import poison
from model_zoo.resnet import ResNet18
from model_zoo.lenet5 import LeNet5
from model_zoo.vgg import VGG11


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Load raw data and keep only two classes
trainset, testset = load_data(config)

# Change the labels to 0 or 1
trainset = change_label(trainset, config)
testset = change_label(testset, config)

# Load in attack data
if not os.path.isdir('attacks'):
    print('Attack images not found, please craft attack images first!')
    sys.exit(0)
train_attacks = torch.load('./attacks/train_attacks')
train_images_attacks = train_attacks['image']
train_labels_attacks = train_attacks['label']
test_attacks = torch.load('./attacks/test_attacks')
test_images_attacks = test_attacks['image']
test_labels_attacks = test_attacks['label']

# Normalize backdoor test images
testset_attacks = torch.utils.data.TensorDataset(test_images_attacks, test_labels_attacks)

# Poison the training set
ind_train = torch.load('./attacks/ind_train')
trainset = poison(trainset, train_images_attacks, train_labels_attacks, ind_train, config)

# Load in the datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)
attackloader = torch.utils.data.DataLoader(testset_attacks, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=8)

# Model
if config['MODEL_TYPE'] == 'resnet18':
    net = ResNet18(num_classes=2)
elif config['MODEL_TYPE'] == 'vgg11':
    net = VGG11(num_classes=2, in_channels=1)
elif config['MODEL_TYPE'] == 'lenet5':
    net = LeNet5(num_classes=2)
else:
    sys.exit("Unknown model_type!")     # Please specify other model types in advance
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=config['LR'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader, 0), total=len(trainloader), smoothing=0.9):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    scheduler.step()

    acc = 100. * correct / total
    print('Train ACC: %.3f' % acc)

    return net


# Test
def eval_clean():
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader, 0), total=len(testloader), smoothing=0.9):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    print('Test ACC: %.3f' % acc)

    return acc


# Test ASR
def eval_attack(attackloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(attackloader, 0), total=len(attackloader), smoothing=0.9):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    asr = 100. * correct / total
    print('ASR: %.3f' % asr)

    return asr


for epoch in range(config['EPOCH']):
    model_contam = train(epoch)
    acc = eval_clean()
    asr = eval_attack(attackloader)

print('Clean test accuracy: %.3f' % acc)
print('Attack success rate: %.3f' % asr)

# Save model
torch.save(model_contam.state_dict(), './model.pth')
