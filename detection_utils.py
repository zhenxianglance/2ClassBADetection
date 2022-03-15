import torch
import torch.nn as nn

import random
import copy
import numpy as np

random.seed()


# Re-implementation of the perturbation estimation algorithm in
# "Detection of backdoors in trained classifiers without access to the training set", Xiang et. al, IEEE TNNLS 2020
def pert_est(images, model, config, t, pi=0.9, device='cuda'):

    # Control parameters
    BATCHSIZE = max(int(np.round(len(images) * 1.0 / 1.0)), 1)
    NSTEP = 1000
    LR = 1e-2   # Smaller learning rate (e.g. 1e-3) is suggested for grey-scale images (e.g. FMNIST, MNIST)
    criterion = nn.CrossEntropyLoss()

    # Create a detection loader
    if config['DATASET'] in ['cifar10', 'cifar100']:
        images = np.transpose(images, [0, 3, 1, 2])
    elif config['DATASET'] in ['fmnist', 'mnist']:
        images = np.expand_dims(images, axis=1)
    images = torch.tensor(images) / 255.
    labels = t * torch.ones((len(images)))
    labels = labels.type(torch.LongTensor)
    sourceset = torch.utils.data.TensorDataset(images, labels)
    sourcesetloader_all = torch.utils.data.DataLoader(sourceset, batch_size=len(labels), shuffle=True, num_workers=1)

    # Perform perturbation estimation for target class
    pert = torch.normal(mean=0, std=1e-3, size=sourceset.__getitem__(0)[0].size())
    pert = pert.to(device)

    for iter_idx in range(NSTEP):
        # Optimizer
        lr_noisy = LR*(1+np.random.normal(loc=0, scale=0.1))
        optimizer = torch.optim.SGD([pert], lr=lr_noisy, momentum=0.0)

        # Require gradient
        pert.requires_grad = True

        # Get a mini-batch of images from s
        sourcesetloader = torch.utils.data.DataLoader(sourceset, batch_size=BATCHSIZE, shuffle=True, num_workers=1)
        batch_idx, (images, labels) = list(enumerate(sourcesetloader))[0]
        images, labels = images.to(device), labels.to(device)
        images_with_bd = torch.clamp(images + pert, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss = criterion(outputs, labels)

        # Update the perturbation (for 1 step)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        # Get the misclassification count for class s
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(sourcesetloader_all):
                images, labels = images.to(device), labels.to(device)
                # Embed the pattern
                images_with_bd = torch.clamp(images + pert, min=0, max=1)
                outputs = model(images_with_bd)
                _, predicted = outputs.max(1)
                misclassification += predicted.eq(labels).sum().item()
                total += len(labels)
        rho = misclassification / total

        # Stopping criteria
        if rho >= pi:
            break

        pert = pert.detach()
    pert_best = copy.deepcopy(pert)

    if rho < pi:
        print('PI-misclassification not achieved, may need to improve the pattern reverse-engineering algorithm.')

    return pert_best.detach().cpu(), rho


def get_MF_pert(pert, images, model, config, t, device='cuda'):
    # Create a data loader
    if config['DATASET'] in ['cifar10', 'cifar100']:
        images = np.transpose(images, [0, 3, 1, 2])
    elif config['DATASET'] in ['fmnist', 'mnist']:
        images = np.expand_dims(images, axis=1)
    images = torch.tensor(images) / 255.
    labels = t * torch.ones((len(images)))
    labels = labels.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(labels), shuffle=True, num_workers=1)

    # Compute the misclassification fraction
    misclassification = 0.0
    total = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # Embed the pattern
            images_with_bd = torch.clamp(images + pert.to(device), min=0, max=1)
            outputs = model(images_with_bd)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            total += len(labels)
    rho = misclassification / total

    return rho


# Re-implement of the patch reverse-engineering algorithm in
# "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks", Wang et. al, IEEE S&P 2019
def pm_est(images, model, config, t, pattern_init, mask_init, pi=0.9, device='cuda'):

    # Control parameters
    BATCHSIZE = max(int(np.round(len(images) * 1.0 / 1.0)), 1)
    COST_INIT = 1e-1  # Initial cost parameter of the L1 norm
    COST_MAX = 1.
    NSTEP1 = 3000  # Maximum number of steps for reaching PI misclassification without L1-constraint
    NSTEP2 = int(1e4)  # Maximum number of steps for pattern estimation after achieving PI misclassification
    PATIENCE_UP = 5
    PATIENCE_DOWN = 5
    PATIENCE_STAGE1 = 1
    PATIENCE_CONVERGENCE = 50
    COST_UP_MULTIPLIER = 1.5
    COST_DOWN_MULTIPLIER = 1.5
    LR1 = 1e-2  # Learning rate for the first stage
    LR2 = 1e-1
    criterion = nn.CrossEntropyLoss()

    # Create a detection loader
    if config['DATASET'] in ['cifar10', 'cifar100']:
        images = np.transpose(images, [0, 3, 1, 2])
    elif config['DATASET'] in ['fmnist', 'mnist']:
        images = np.expand_dims(images, axis=1)
    images = torch.tensor(images) / 255.
    labels = t * torch.ones((len(images)))
    labels = labels.type(torch.LongTensor)
    sourceset = torch.utils.data.TensorDataset(images, labels)
    sourcesetloader_all = torch.utils.data.DataLoader(sourceset, batch_size=len(labels), shuffle=True, num_workers=1)

    # Perform pattern-mask estimation for target class
    im_size = sourceset.__getitem__(0)[0].size()
    if pattern_init is None:
        pattern_raw = torch.ones(im_size) * random.uniform(-1.5, 1.5)
        mask_raw = torch.zeros((1, im_size[1], im_size[2]))
    else:
        pattern_raw = torch.arctanh(pattern_init)
        mask_raw = torch.arctanh(mask_init[0, :, :])
    noise = torch.normal(0, 1e-2, size=pattern_raw.size())
    pattern_raw, mask_raw = (pattern_raw + noise).to(device), (mask_raw + noise[0, :, :]).to(device)

    mask_norm_best = float("inf")
    associated_rho = 0.0

    # First stage, achieve PI-level misclassification
    stopping_count = 0

    for iter_idx in range(NSTEP1):

        if mask_init is not None:
            pattern = (torch.tanh(pattern_raw) + 1) / 2
            mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
            rho = 1.
            break

        # Optimizer
        lr_noisy = LR1 * (1 + np.random.normal(loc=0, scale=0.1))
        optimizer = torch.optim.SGD([pattern_raw, mask_raw], lr=lr_noisy, momentum=0.5)

        # Require gradient
        pattern_raw.requires_grad = True
        mask_raw.requires_grad = True

        # Get a mini-batch of images from s
        sourcesetloader = torch.utils.data.DataLoader(sourceset, batch_size=BATCHSIZE, shuffle=True, num_workers=1)
        batch_idx, (images, labels) = list(enumerate(sourcesetloader))[0]
        images, labels = images.to(device), labels.to(device)

        # Embed the backdoor pattern
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss = criterion(outputs, labels)

        # Update the pattern and mask (for 1 step)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Clip pattern_raw and mask_raw to avoid saturation
        pattern_raw, mask_raw = pattern_raw.detach(), mask_raw.detach()
        pattern_raw.clamp(min=-10., max=10.)
        mask_raw.clamp(min=-10., max=10.)

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        # Get the misclassification count for class s
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(sourcesetloader_all):
                images, labels = images.to(device), labels.to(device)
                # Embed the backdoor pattern
                images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

                outputs = model(images_with_bd)
                _, predicted = outputs.max(1)
                misclassification += predicted.eq(labels).sum().item()
                total += len(labels)
        rho = misclassification / total

        # Stopping criteria
        if rho >= pi:
            stopping_count += 1
        else:
            stopping_count = 0

        if stopping_count >= PATIENCE_STAGE1:
            break

    mask_best = copy.deepcopy(mask)
    pattern_best = copy.deepcopy(pattern)

    if rho < pi:
        print('PI-misclassification not achieved in phase 1.')

    # Second State, jointly optimize pattern and mask with the L1 constraint
    stopping_count = 0

    # Set the cost manipulation parameters
    cost = COST_INIT  # Initialize the cost of L1 constraint
    cost_up_counter = 0
    cost_down_counter = 0

    for iter_idx in range(NSTEP2):
        # Optimizer
        optimizer = torch.optim.SGD([pattern_raw, mask_raw], lr=LR2, momentum=0.0)

        # Require gradient
        pattern_raw.requires_grad = True
        mask_raw.requires_grad = True

        # Get the loss
        sourcesetloader = torch.utils.data.DataLoader(sourceset, batch_size=BATCHSIZE, shuffle=True, num_workers=1)
        batch_idx, (images, labels) = list(enumerate(sourcesetloader))[0]
        images, labels = images.to(device), labels.to(device)
        # Embed the backdoor pattern
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

        # Feed the image with the backdoor into the classifier, and get the loss
        outputs = model(images_with_bd)
        loss = criterion(outputs, labels)

        # Add the loss corresponding to the L1 constraint
        loss += cost * torch.sum(torch.abs(mask))

        # Update the pattern & mask (for 1 step)
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Compute the misclassification fraction
        misclassification = 0.0
        total = 0.0
        pattern = (torch.tanh(pattern_raw) + 1) / 2
        mask = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
        # Get the misclassification count for class s
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(sourcesetloader_all):
                images, labels = images.to(device), labels.to(device)
                # Embed the pattern
                images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)

                outputs = model(images_with_bd)
                _, predicted = outputs.max(1)
                misclassification += predicted.eq(labels).sum().item()
                total += len(labels)
        rho = misclassification / total

        # Modify the cost
        # Check if the current loss causes the misclassification fraction to be smaller than PI
        if rho >= pi:
            cost_up_counter += 1
            cost_down_counter = 0
        else:
            cost_up_counter = 0
            cost_down_counter += 1
        # If the misclassification fraction to be smaller than PI for more than PATIENCE iterations, reduce the cost;
        # else, increase the cost
        if cost_up_counter >= PATIENCE_UP and cost <= COST_MAX:
            cost_up_counter = 0
            cost *= COST_UP_MULTIPLIER
        elif cost_down_counter >= PATIENCE_DOWN:
            cost_down_counter = 0
            cost /= COST_DOWN_MULTIPLIER

        # Stopping criteria
        if rho >= pi and torch.sum(torch.abs(mask)) < mask_norm_best * 0.99:
            mask_norm_best = torch.sum(torch.abs(mask))
            pattern_best = (torch.tanh(pattern_raw) + 1) / 2
            mask_best = (torch.tanh(mask_raw.repeat(im_size[0], 1, 1)) + 1) / 2
            associated_rho = rho
            stopping_count = 0
        else:
            stopping_count += 1

        if stopping_count >= PATIENCE_CONVERGENCE:
            break

    return pattern_best.detach().cpu(), mask_best.detach().cpu(), associated_rho


def get_MF_patch(pattern, mask, images, model, config, t, device='cuda'):
    # Create a data loader
    if config['DATASET'] in ['cifar10', 'cifar100']:
        images = np.transpose(images, [0, 3, 1, 2])
    elif config['DATASET'] in ['fmnist', 'mnist']:
        images = np.expand_dims(images, axis=1)
    images = torch.tensor(images) / 255.
    labels = t * torch.ones((len(images)))
    labels = labels.type(torch.LongTensor)
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(labels), shuffle=True, num_workers=1)

    # Compute the misclassification fraction
    misclassification = 0.0
    total = 0.0
    pattern, mask = pattern.to(device), mask.to(device)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            # Embed the pattern
            images_with_bd = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)
            outputs = model(images_with_bd)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            total += len(labels)
    rho = misclassification / total

    return rho
