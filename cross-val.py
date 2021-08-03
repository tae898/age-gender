"""
This script is specifically made for the Adience dataset, since the evaluation is 
done with five-fold cross validation.
"""
import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, update_lr_scheduler, read_json, write_json
from tqdm import tqdm
import os
from datetime import datetime
from torch.cuda.amp import autocast # for float16 mixed point precision
from pprint import pprint


def train(config: ConfigParser):
    logger = config.get_logger('train')

    config['data_loader']['args']['training'] = True

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    config = update_lr_scheduler(config, len(data_loader))

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    if config['checkpoint']:
        logger.info(f'Loading checkpoint: {config["checkpoint"]} ...')
        checkpoint = torch.load(config["checkpoint"])
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj(
        'lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      amp=config['amp'])

    trainer.train()


def train_to_dump(config: ConfigParser, checkpoint: str) -> dict:
    """Save train / val results."""
    logger = config.get_logger('train_to_dump')

    config['data_loader']['args']['training'] = True

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info(f'Loading checkpoint: {checkpoint} ...')
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    if config['n_gpu'] > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    logs = {}
    for split in ['train', 'val']:
        if split == 'train':
            dl = data_loader
        else:
            dl = valid_data_loader
        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        with torch.no_grad():
            for i, (data, target) in tqdm(enumerate(dl)):
                data, target = data.to(device), target.to(device)

                if config['amp']:
                    with autocast():
                        output = model(data)
                        loss = loss_fn(output, target)
                else:
                    output = model(data)
                    loss = loss_fn(output, target)

                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(dl.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(f"{split}: {log}")
        logs[split] = log

    return logs


def test(config: ConfigParser, checkpoint: str) -> dict:
    """Save test results"""
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        data_dir=config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        num_workers=config['data_loader']['args']['num_workers'],
        dataset=config['data_loader']['args']['dataset'],
        num_classes=config['data_loader']['args']['num_classes'],
        test_cross_val=config['data_loader']['args']['test_cross_val'],
        training=False,
    )
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info(f'Loading checkpoint: {checkpoint} ...')
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    if config['n_gpu'] > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(data_loader)):
            data, target = data.to(device), target.to(device)

            if config['amp']:
                with autocast():
                    output = model(data)
                    loss = loss_fn(output, target)
            else:
                output = model(data)
                loss = loss_fn(output, target)

            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(f"test: {log}")

    return log


def main(config_path: str):
    config_dict = read_json(config_path)
    num_cross_val = config_dict['num_cross_val']
    SEEDS = config_dict['seeds']

    to_dump = {'config': config_dict}
    to_dump['stats'] = {}

    for SEED in tqdm(SEEDS):
        # fix random seeds for reproducibility
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)

        to_dump['stats'][SEED] = {split: {}
                                  for split in ['train', 'val', 'test']}

        for i in range(num_cross_val):
            config_dict['data_loader']['args']['test_cross_val'] = i
            config = ConfigParser(config_dict)
            train(config)
            logs = train_to_dump(config, checkpoint=os.path.join(
                config.save_dir, 'model_best.pth'))
            to_dump['stats'][SEED]['train'].update({i: logs['train']})
            to_dump['stats'][SEED]['val'].update({i: logs['val']})

            log = test(config, checkpoint=os.path.join(
                config.save_dir, 'model_best.pth'))

            to_dump['stats'][SEED]['test'].update({i: log})

            pprint(to_dump['stats'][SEED])

    for split in ['train', 'val', 'test']:
        for metric in ['loss'] + config['metrics']:
            to_dump['stats'][f'{split}_{metric}_mean'] = np.nanmean(
                [to_dump['stats'][SEED][split][i][metric] for SEED in SEEDS for i in range(num_cross_val)])

            to_dump['stats'][f'{split}_{metric}_std'] = np.nanstd(
                [to_dump['stats'][SEED][split][i][metric] for SEED in SEEDS for i in range(num_cross_val)])
                
    filepath = os.path.join(config_dict['trainer']['save_dir'],
                            datetime.now().strftime(r'%m%d_%H%M%S') + '_cross-val-results.json')
    write_json(to_dump, filepath)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='cross validation on adience dataset')
    args.add_argument('-c', '--config', default="cross-val.json", type=str,
                      help='config file path (default: None)')

    config_path = args.parse_args().config
    main(config_path)
