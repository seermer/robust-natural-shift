import argparse
import torch
import sys
import logging
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch import nn, optim

import utils
import arg_process
from custom_transforms import cutout, autoaugment
from dataset import dataset_util
from model.resnet import get_resnet_model


def get_loaders(batch_size: int, **kwargs):
    arg1, arg2 = {}, {}
    for key in kwargs.keys():
        arg = kwargs[key]
        if isinstance(arg, list) or isinstance(arg, tuple):
            if len(arg) == 1:
                arg1[key] = arg[0]
                arg2[key] = arg[0]
            elif len(arg) == 2:
                arg1[key] = arg[0]
                arg2[key] = arg[1]
            else:
                raise NotImplementedError('list or tuple args passed to get_loader must be length one or two')
        else:
            arg1[key] = arg
            arg2[key] = arg

    labeled_dataset = dataset_util.get_dataset(**arg1)
    unlabeled_dataset = dataset_util.get_dataset(**arg2)
    loader1 = DataLoader(dataset=labeled_dataset[0], batch_size=batch_size // 2, shuffle=True)
    if labeled_dataset[1] is None:
        loader1_val = None
    else:
        loader1_val = DataLoader(dataset=labeled_dataset[1], batch_size=batch_size // 2, shuffle=True)
    unlabeled_bs = batch_size - batch_size // 2
    loader2 = DataLoader(dataset=unlabeled_dataset[0], batch_size=unlabeled_bs, shuffle=True)
    if unlabeled_dataset[1] is None:
        loader2_val = None
    else:
        loader2_val = DataLoader(dataset=unlabeled_dataset[1], batch_size=unlabeled_bs, shuffle=True)
    return (loader1, loader2), (loader1_val, loader2_val)


def train(model, loaders, val_loaders, epochs, criterion, optimizer, val_freq, fake_label, device):
    assert len(loaders) == len(val_loaders) == 2, 'two loaders must be given'
    val = (val_loaders[0] is not None and val_loaders[1] is not None)
    metrics = utils.MetricAccumulator([True, True], 'loss', 'accuracy')
    model = model.to(device)
    y2 = torch.full(size=(loaders[1].batch_size,), fill_value=fake_label,
                    dtype=torch.long if isinstance(fake_label, int) else torch.float32)
    for i in range(1, epochs + 1):
        model.train()
        for j, ((x1, y1), (x2, y2_temp)) in enumerate(zip(*loaders)):
            if len(y2_temp) != len(y2):
                y2 = torch.full(size=(len(y2_temp),), fill_value=fake_label,
                                dtype=y2.dtype)
            x = torch.vstack((x1, x2)).to(device)
            y = torch.cat((y1, y2)).to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            with torch.no_grad():
                acc = torch.sum(torch.as_tensor(torch.argmax(y_pred, dim=-1) == y)) / len(y)
                metrics.add('loss', loss.item())
                metrics.add('accuracy', acc.item())
            loss.backward()
            optimizer.step()
            step = j + 1
            if step % 100 == 0:
                logging.info('epoch {}/{} step {}/{}: {}'.format(i, epochs, step, len(loaders[0]), metrics.info(step)))
        if val_freq > 0 and i % val_freq == 0 and val:
            validation(model, val_loaders, criterion, fake_label, device)
        metrics.reset()
    if val_freq > 0 and epochs % val_freq != 0 and val:
        validation(model, val_loaders, criterion, fake_label, device)


@torch.no_grad()
def validation(model, loaders, criterion, fake_label, device):
    metrics = utils.MetricAccumulator([True, True], 'val_loss', 'val_accuracy')
    model = model.to(device)
    model.eval()
    y2 = torch.full(size=(loaders[1].batch_size,), fill_value=fake_label,
                    dtype=torch.long if isinstance(fake_label, int) else torch.float32)
    assert len(loaders) == 2, 'two loaders must be given'
    for j, ((x1, y1), (x2, _)) in enumerate(zip(*loaders)):
        x = torch.vstack((x1, x2)).to(device)
        y = torch.cat((y1, y2)).to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = torch.sum(torch.as_tensor(torch.argmax(y_pred, dim=-1) == y)) / len(y)
        metrics.add('val_loss', loss.item())
        metrics.add('val_accuracy', acc.item())
    logging.info(metrics.info())


def main():
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(message)s',
                        handlers=[logging.StreamHandler()])
    parser = argparse.ArgumentParser(description='Pytorch training detector for robustness against natural shift')
    checker = utils.ArgumentChecker(sys.argv)
    arg_process.default_args(parser, checker)
    arg_process.detector_args(parser, checker)
    args = parser.parse_args()
    train_transform = []
    if args.other_transforms:
        train_transform += [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip()]
    allowed_autoaug = {'cifar10': autoaugment.CIFAR10Policy(),
                       'imagenet': autoaugment.ImageNetPolicy(),
                       'svhn': autoaugment.SVHNPolicy(),
                       'svhn_extra': autoaugment.SVHNPolicy()}
    if args.autoaug and args.labeled_dataset in allowed_autoaug.keys():
        train_transform.append(allowed_autoaug[args.labeled_dataset])
    train_transform.append(transforms.ToTensor())
    if args.cutout:
        train_transform.append(cutout.Cutout(n_holes=1, length=16))
    if args.normalize_input:
        train_transform.append(transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)))
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.ToTensor()

    custom_train_path = [[args.labeled_train]]
    if args.unlabeled_train is None:
        custom_train_path.append(None)
    else:
        custom_train_path.append([args.unlabeled_train])
    if args.labeled_train_y is not None:
        custom_train_path[0].append(args.labeled_train_y)
    custom_val_path = [[args.labeled_val]]
    if args.unlabeled_val is None:
        custom_val_path.append(None)
    else:
        custom_val_path.append([args.unlabeled_val])
    if args.labeled_val_y is not None:
        custom_val_path[0].append(args.labeled_val_y)
    loader_args = {
        'batch_size': args.batch_size,
        'name': [args.labeled_dataset, args.unlabeled_dataset],
        'transform': train_transform,
        'val_transform': val_transform,
        'custom_train_path': custom_train_path,
        'custom_val_path': custom_val_path,
        'custom_mode': [args.labeled_mode, args.unlabeled_mode]
    }
    loaders, val_loaders = get_loaders(**loader_args)
    model_arg = args.model.lower()
    model_config = {
        'width_factor': args.model_width,
        'num_classes': args.num_classes + 1,
        'wide_resnet': 'wide' in model_arg,
        'configs': args.model_config,
        'preact': 'preact' in model_arg,
        'bottleneck': 'bottleneck' in model_arg,
        'reduced_stem': len(args.model_config) < 4,
        'activation': utils.activation(args.activation)
    }
    # TODO: try nn.DistributedDataParallel
    model = nn.parallel.DataParallel(get_resnet_model(**model_config)).to('cuda')

    # TODO: add support to more loss
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    train(model, loaders, val_loaders, args.epochs, criterion, optimizer, args.val_freq, args.num_classes, 'cuda')


if __name__ == '__main__':
    main()
