import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid

import argparse
import os
from tqdm import tqdm
from pprint import pprint
import numpy as np
from copy import deepcopy 

from utils import set_seed, CIFAR10Poisoned, AverageMeter, accuracy_top1, transform_test, make_and_restore_model
from attacks.step import LinfStep, L2Step
from utils import show_image_row
from poison_loader import folder_load

from datasets import CIFAR10

STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}



def label_project(labels):
    projection = [2,9,4,5,2,3,3,5,0,1]
    new_labels = deepcopy(labels)
    for i in range(len(labels)):
        new_labels[i] = projection[labels[i]]
    return new_labels


def gradient_poison(args, model, dirty_model, writer):
    poisoned_input = torch.empty(50000, args.input_channel, args.input_size, args.input_size)
    clean_target = torch.empty(50000).long()

    train_set = CIFAR10('../data/', train=True, transform=transform_test)
    test_set = CIFAR10('../data/', train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, drop_last=False, shuffle=True)

    poison_delta = torch.empty(50000, args.input_channel, args.input_size, args.input_size).uniform_(
                -args.eps, args.eps
            )
    poison_delta = torch.nn.Parameter(poison_delta.clone().detach().requires_grad_(True).cuda(non_blocking=True))

    
    att_optimizer = torch.optim.Adam([poison_delta], lr=0.1, weight_decay=0)
    # att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)

    if args.scheduling:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[args.attackiter // 2.667, args.attackiter // 1.6,
                                                                                    args.attackiter // 1.142], gamma=0.1)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for step in range(args.attackiter):
        att_optimizer.zero_grad()
        poison_delta.grad = torch.zeros_like(poison_delta)

        for batch, batch_sample in enumerate(train_loader):
            inputs, labels, ids = batch_sample # __getitem__ return img, target, index

            delta_slice = poison_delta[ids].detach().cuda(non_blocking=True)
            delta_slice.requires_grad_()

            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # dirty_label = torch.remainder(labels+1, 10)
            dirty_label = label_project(labels)
            poison_input = torch.clamp(inputs+delta_slice, 0, 1)

            output = model(poison_input)
            ce_loss = criterion(output, dirty_label)

            inp_grad, _ = gradient(model, poison_input, labels, criterion, re_graph=True)
            dir_inp_grad, _ = gradient(model, inputs, dirty_label, criterion, re_graph=True)
            loss = grad_loss(inp_grad, dir_inp_grad) + ce_loss

            print('Step: {:d} loss: {:.4f}'.format(step, loss))


            loss.backward()
            # poison_delta.grad.sign_() # FGSM-style optimization with sign

            poison_delta.grad[ids] = delta_slice.grad.detach()

            att_optimizer.step()
            if args.scheduling:
                scheduler.step()

            with torch.no_grad():
                # Projection Step
                poison_delta.data = torch.clamp(poison_delta, min=-args.eps, max=args.eps)
                poison_imgs = inputs.detach()+poison_delta[ids].detach()
                poison_imgs = torch.clamp(poison_imgs, min=0, max=1)
                delta_slice.data = poison_imgs.detach() - inputs.detach()
                poison_delta[ids] = delta_slice.detach()
        print('----- ', poison_delta[0][0][0])

    for batch, batch_sample in enumerate(train_loader):
        inputs, labels, ids = batch_sample # __getitem__ return img, target, index

        delta_slice = poison_delta[ids].detach().cuda(non_blocking=True)

        inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        poison_input = torch.clamp(inputs+delta_slice, 0, 1)
        
        poisoned_input[ids] = poison_input.detach().cpu()
        clean_target[ids] = labels.detach().cpu()


    return poisoned_input, clean_target


def get_poison_ids(img_ids, img2poi):
    return [img2poi[int(key)] for key in img_ids]


def gradient(model, images, labels, criterion=None, re_graph=False):
    """Compute the gradient of criterion(model) w.r.t to given data."""
    loss = criterion(model(images), labels)
    if re_graph:
        gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    else:
        gradients = torch.autograd.grad(loss, model.parameters(), only_inputs=True)
    grad_norm = 0
    for grad in gradients:
        grad_norm += grad.detach().pow(2).sum()
    grad_norm = grad_norm.sqrt()
    return gradients, grad_norm


def grad_loss(poison_grad, target_grad, loss_type='MSE'):
    loss = 0

    for i in range(len(poison_grad)):
        if loss_type == 'scalar_product':
            loss -= (target_grad[i] * poison_grad[i]).sum()
        elif loss_type == 'cosine1':
            loss -= torch.nn.functional.cosine_similarity(target_grad[i].flatten(), poison_grad[i].flatten(), dim=0)
        elif loss_type == 'SE':
            loss += 0.5 * (target_grad[i] - poison_grad[i]).pow(2).sum()
        elif loss_type == 'MSE':
            loss += torch.nn.functional.mse_loss(target_grad[i], poison_grad[i])
        elif loss_type == 'SUM':
            loss += poison_grad[i].sum().abs()

    return loss


def poisoning(args, loader, model, writer, dirty_model=None):
    set_seed(args.seed)

    if args.poison_type == 'P1':
        poisoned_data = poison_p1(args, loader, model, writer)
    elif args.poison_type == 'P2':
        poisoned_data = poison_p2(args, loader, model, writer)
    elif args.poison_type == 'P3':
        poisoned_data = poison_p3(args, loader, model, writer)
    elif args.poison_type == 'P4':
        poisoned_data = poison_p4(args, loader, model, writer)
    elif args.poison_type == 'P5':
        poisoned_data = poison_p5(args, loader, model, writer)
    elif args.poison_type == 'P6': # adversarial + dirty label model
        poisoned_data = poison_p6(args, loader, model, dirty_model, writer)
    elif args.poison_type == 'grad': # adversarial + dirty label model
        poisoned_data = gradient_poison(args, model, dirty_model, writer)
    torch.save(poisoned_data, args.poison_file_path)


def visualization(args, writer):
    clean_set = datasets.CIFAR10(args.clean_data_path, train=True, transform=transform_test)
    poison_set = CIFAR10Poisoned(args.poison_data_path, args.constraint, args.poison_type, 1.0, transform=transform_test)

    clean_loader = DataLoader(clean_set, batch_size=5, shuffle=False, num_workers=8)
    poison_loader = DataLoader(poison_set, batch_size=5, shuffle=False, num_workers=8)

    clean_iterator = iter(clean_loader)
    poison_iterator = iter(poison_loader)
    for i in range(3):
        clean_inp, label = next(clean_iterator)
        poison_inp, label = next(poison_iterator)

        imgs = torch.cat([clean_inp, poison_inp], dim=0)
        vis = make_grid(imgs, nrow=5, normalize=False, scale_each=False)
        writer.add_image('poisoned_examples', vis, global_step=i, dataformats='CHW')

        ylist = None
        # ylist = ['$\mathcal{D}$', '$\widehat{\mathcal{D}}_{\mathsf{P5}}$']
        # clean_set.classes[1] = 'car'

        show_image_row([clean_inp, poison_inp],
                ylist=ylist,
                tlist=[[clean_set.classes[int(t)] for t in l] for l in [label, label]],
                fontsize=20,
                filename=os.path.join(os.path.join(args.out_dir, args.exp_name), 'poisoned_examples_{}.png'.format(i)))

# def main(args):
#     visualization(args, None)

def main(args):

    if os.path.isfile(args.poison_file_path):
        print('Poison [{}] already exists.'.format(args.poison_file_path))
        return
    
    data_set = datasets.CIFAR10(args.clean_data_path, train=True, download=True, transform=transform_test)
    # data_set = folder_load(path='../data/TAP/', T=transform_test, poison_rate=1.0)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False)

    model = make_and_restore_model(args.arch, resume_path=args.model_path)
    # dirty_model = make_and_restore_model(args.arch, resume_path=args.dirty_model_path)
    model.eval()
    writer = SummaryWriter(args.tensorboard_path)

    poisoning(args, data_loader, model, writer)
    
    # visualization(args, writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate poisoned dataset for CIFAR10')
    
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--input_channel', default=3, type=int)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')

    parser.add_argument('--poison_rate', default=0.1, type=float)

    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--eps', default=0.5, type=float)
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)

    parser.add_argument('--arch', default='ResNet18', type=str, choices=['VGG16', 'ResNet18'])
    parser.add_argument('--model_path', default='results/ResNet18-STonC-lr0.1-bs128-wd0.0005-pr0.0-seed0-dirtylabel/checkpoint.pth', type=str)
    parser.add_argument('--dirty_model_path', default='results/ResNet18-STonTAP(L2)-lr0.1-bs128-wd0.0005-pr1.0-seed0-TAP/checkpoint.pth', type=str)

    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--out_dir', default='results/', type=str)
    parser.add_argument('--clean_data_path', default='../data', type=str)
    parser.add_argument('--poison_data_path', default='../data/CIFAR10Poison', type=str)
    parser.add_argument('--poison_type', default='C', choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'grad'])

    parser.add_argument('--gpuid', default=0, type=int)

    args = parser.parse_args()

    args.exp_name = '{}-{}-{}-eps{:.5f}'.format(args.arch, args.poison_type, args.constraint, args.eps)
    args.tensorboard_path = os.path.join(os.path.join(args.out_dir, args.exp_name), 'tensorboard')
    args.batch_size = 256
    args.num_classes = 10
    args.data_shape = (3, 32, 32)
    args.num_steps = 300
    if args.constraint == 'Linf':
        args.eps /= 255
    args.step_size = args.eps / 100

    args.poison_data_path = os.path.expanduser(args.poison_data_path)
    if not os.path.exists(args.poison_data_path):
        os.makedirs(args.poison_data_path)
    args.poison_file_path = os.path.join(args.poison_data_path, '{}.{}.{:.1f}'.format(args.constraint, args.poison_type.lower(), args.eps*255))

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
