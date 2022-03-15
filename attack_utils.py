import sys
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt


def create_pattern(im_size, config):

    if config['PATTERN_TYPE'] == "perturbation":
        pert_size = config['PERTURBATION_SIZE']
        pert_shape = config['PERTURBATION_SHAPE']
        if pert_shape == 'chessboard':
            pert = torch.zeros(im_size)
            for i in range(im_size[1]):
                for j in range(im_size[2]):
                    if (i + j) % 2 == 0:
                        pert[:, i, j] = torch.ones(im_size[0])
            pert *= pert_size
        elif pert_shape == 'static':
            pert = torch.zeros(im_size)
            for i in range(im_size[1]):
                for j in range(im_size[2]):
                    if (i % 2 == 0) and (j % 2 == 0):
                        pert[:, i, j] = torch.ones(im_size[0])
            pert *= pert_size
        elif pert_shape == 'lshape':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            for c in range(im_size[0]):
                pert[c, cx, cy + 1] = pert_size
                pert[c, cx - 1, cy] = pert_size
                pert[c, cx - 2, cy] = pert_size
                pert[c, cx, cy] = pert_size
        elif pert_shape == 'cross':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            for c in range(im_size[0]):
                pert[c, cx, cy - 1] = pert_size
                pert[c, cx, cy + 1] = pert_size
                pert[c, cx - 1, cy] = pert_size
                pert[c, cx + 1, cy] = pert_size
                pert[c, cx, cy] = pert_size
        elif pert_shape == 'X':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            ch = torch.randint(low=0, high=im_size[0], size=(1,))
            pert[ch, cx - 1, cy - 1] = pert_size
            pert[ch, cx - 1, cy + 1] = pert_size
            pert[ch, cx + 1, cy - 1] = pert_size
            pert[ch, cx + 1, cy + 1] = pert_size
            pert[ch, cx, cy] = pert_size
        elif pert_shape == 'pixel':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            ch = torch.randint(low=0, high=im_size[0], size=(1,))
            sgn = torch.randint(low=0, high=2, size=(1,)) * 2 - 1
            pert[ch, cx, cy] += sgn * pert_size * (1 + 0.2 * random.random())
        elif pert_shape == 'square':
            pert = torch.zeros(im_size)
            cx = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            cy = torch.randint(low=3, high=im_size[-1] - 3, size=(1,))
            ch = torch.randint(low=0, high=im_size[0], size=(1,))
            pert[ch, cx, cy] = pert_size
            pert[ch, cx, cy + 1] = pert_size
            pert[ch, cx + 1, cy] = pert_size
            pert[ch, cx + 1, cy + 1] = pert_size
        else:
            sys.exit("Perturbation shape is unrecognized!")
        return pert
    elif config['PATTERN_TYPE'] == "patch":
        mask_size = config['MASK_SIZE']
        margin = config['MARGIN']
        patch_type = config['PATCH_TYPE']
        if margin * 2 + mask_size >= im_size[1] or margin * 2 + mask_size >= im_size[2]:
            sys.exit("Decrease margin or mask size!")
        # Pick a random location
        x_candidate = torch.from_numpy(np.concatenate([np.arange(0, margin),
                                                       np.arange(int(im_size[1] - margin - mask_size + 1),
                                                                 int(im_size[1] - mask_size + 1))]))
        y_candidate = torch.from_numpy(np.concatenate([np.arange(0, margin),
                                                       np.arange(int(im_size[2] - margin - mask_size + 1),
                                                                 int(im_size[2] - mask_size + 1))]))
        x = x_candidate[torch.randperm(len(x_candidate))[0]].item()
        y = y_candidate[torch.randperm(len(y_candidate))[0]].item()
        # Create mask and pattern
        mask = torch.zeros(im_size)
        mask[:, x:x + mask_size, y:y + mask_size] = 1
        if patch_type == 'noise':
            patch = torch.randint(0, 255, size=(im_size[0], mask_size, mask_size)) / 255
        elif patch_type == 'uniform':
            color = torch.randint(50, 200, size=(im_size[0], 1, 1)) / 255
            patch = torch.ones((im_size[0], mask_size, mask_size)) * color.repeat(1, mask_size, mask_size)
        pattern = torch.zeros(im_size)
        pattern[:, x:x + mask_size, y:y + mask_size] = patch
        pattern = (pattern, mask)
        return pattern
    else:
        sys.exit("Pattern type is unrecognized!")

    pass


def pattern_save(pattern, config, path):

    if config['PATTERN_TYPE'] == "perturbation":
        pattern = pattern.numpy()
        pattern = np.transpose(pattern, [1, 2, 0])
        if config['DATASET'] in ['cifar10', 'cifar100', 'stl10']:
            plt.imshow(pattern)
        else:
            plt.imshow(pattern[:, :, 0], cmap='gray', vmin=0., vmax=1.)
        plt.savefig(os.path.join(path, 'backdoor_pattern.png'))
    elif config['PATTERN_TYPE'] == "patch":
        pattern = pattern[0].numpy()
        pattern = np.transpose(pattern, [1, 2, 0])
        if config['DATASET'] in ['cifar10', 'cifar100', 'stl10']:
            plt.imshow(pattern)
        else:
            plt.imshow(pattern[:, :, 0], cmap='gray', vmin=0., vmax=1.)
        plt.savefig(os.path.join(path, 'backdoor_pattern.png'))
    else:
        sys.exit("Pattern type is unrecognized!")

    pass


def backdoor_embedding(image, pattern, config):

    if config['PATTERN_TYPE'] == "perturbation":
        image += pattern
        image *= 255
        image = image.round()
        image /= 255
        image = image.clamp(0, 1)
    elif config['PATTERN_TYPE'] == "patch":
        image = image * (1 - pattern[1]) + pattern[0] * pattern[1]
    else:
        sys.exit("Pattern type is unrecognized!")

    return image


def poison(trainset, images, labels, ind, config):

    dataset = config['DATASET']
    image_dtype = trainset.data.dtype
    if dataset in ['cifar10', 'cifar100']:
        images = np.rint(np.transpose(images.numpy() * 255, [0, 2, 3, 1])).astype(image_dtype)
        trainset.data = np.concatenate((trainset.data, images))
        trainset.targets = np.concatenate((trainset.targets, labels))
    if dataset in ['mnist', 'fmnist']:
        images *= 255
        images = images.type(image_dtype)
        images = torch.squeeze(images, dim=1)
        trainset.data = torch.cat((trainset.data, images))
        trainset.targets = torch.cat((trainset.targets, labels))
    if dataset == 'stl10':
        images = np.rint(images.numpy() * 255).astype(image_dtype)
        trainset.data = np.concatenate((trainset.data, images))
        trainset.labels = np.concatenate((trainset.labels, labels))
    if dataset in ['cifar10', 'mnist', 'fmnist']:
        trainset.data = np.delete(trainset.data, ind, axis=0)
        trainset.targets = np.delete(trainset.targets, ind, axis=0)

    return trainset

