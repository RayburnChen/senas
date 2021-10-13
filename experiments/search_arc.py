import os
import sys
import time

import yaml
import datetime
import shutil
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn

sys.path.append('..')
from utils.loss.loss import SegmentationLosses, MultiSegmentationLosses
from utils.datasets import get_dataset
from utils.utils import get_logger, save_checkpoint, gpu_memory
from utils.utils import calc_time
from utils.utils import get_gpus_memory_info, calc_parameters_count
from utils.optimizers import get_optimizer
from utils.metrics import *
from search.senas_search import NAS, Architecture

from tensorboardX import SummaryWriter


class SearchNetwork(object):

    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._check_resume()

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')
        parser.add_argument('--config', nargs='?', type=str, default='../configs/senas/senas_promise12.yml',
                            help='Configuration file to use')
        parser.add_argument('--batch_size', nargs='?', type=int, default=-1, help='Batch size')
        parser.add_argument('--meta_node_num', nargs='?', type=int, default=-1, help='Batch size')

        self.args = parser.parse_args()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp, Loader=yaml.FullLoader)
            print('load configure file at {}'.format(self.args.config))

    def _init_logger(self):
        log_dir = '../logs/senas/search' + '/{}'.format(self.cfg['data']['dataset']) + \
                  '/search-{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        self.logger = get_logger(log_dir)
        self.logger.info('RUNDIR: {}'.format(log_dir))
        shutil.copy(self.args.config, log_dir)
        self.logger.info('Nas-Search')
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def _init_device(self):
        self.device = torch.device("cuda" if self.cfg['searching']['gpu'] else "cpu")
        self.logger.info('seed is {}'.format(self.cfg.get('seed', 0)))
        np.random.seed(self.cfg.get('seed', 0))
        torch.manual_seed(self.cfg.get('seed', 0))
        if self.cfg['searching']['gpu'] and torch.cuda.is_available():
            self.device_id, _ = get_gpus_memory_info()
            self.device = torch.device('cuda:{}'.format(0 if self.cfg['searching']['multi_gpus'] else self.device_id))
            torch.cuda.manual_seed(self.cfg.get('seed', 0))
            torch.cuda.set_device(self.device)
            cudnn.enabled = True
            cudnn.benchmark = True
        else:
            self.logger.info('No gpu devices available!, we will use cpu')
            self.device = torch.device('cpu')
            self.device_id = 0

    def _init_dataset(self):
        trainset = get_dataset(self.cfg['data']['dataset'], split='train', mode='train')

        num_train = len(trainset)
        indices = list(range(num_train))
        split = int(np.floor(self.cfg['searching']['train_portion'] * num_train))
        self.n_classes = trainset.num_class
        self.in_channels = trainset.in_channels
        self.batch_size = self.args.batch_size if self.args.batch_size > 0 else self.cfg['searching']['batch_size']
        kwargs = {'num_workers': self.cfg['searching']['n_workers'], 'pin_memory': True}
        self.train_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices[:split]), **kwargs)

        self.valid_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices[split:num_train]), **kwargs)

    def _init_model(self):

        # Read the configure
        init_channel = self.cfg['searching']['init_channels']
        depth = self.cfg['searching']['depth']
        supervision = self.cfg['searching']['deep_supervision']
        meta_node_num = self.args.meta_node_num if self.args.meta_node_num > 0 else self.cfg['searching'][
            'meta_node_num']
        loss_name = self.cfg['searching']['loss']['name']

        # Setup loss function
        self.criterion = MultiSegmentationLosses(loss_name, depth) if supervision else SegmentationLosses(loss_name)
        self.logger.info("Using loss {}".format(loss_name))

        # Setup Model
        model = NAS(self.in_channels, init_channel, self.n_classes, depth,
                    meta_node_num=meta_node_num, use_sharing=self.cfg['searching']['sharing_normal'],
                    double_down_channel=self.cfg['searching']['double_down_channel'],
                    supervision=supervision,
                    multi_gpus=self.cfg['searching']['multi_gpus'],
                    device=self.device)

        if self.device.type == 'cuda':
            if torch.cuda.device_count() > 1 and self.cfg['searching']['multi_gpus']:
                self.logger.info('use: %d gpus', torch.cuda.device_count())
                self.model = nn.DataParallel(model)
            elif torch.cuda.is_available():
                self.logger.info('gpu device = %d' % self.device_id)
                torch.cuda.set_device(self.device)

        # stat_info(model, (1, 256, 256))
        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

        # Setup optimizer, lr_scheduler and loss function for model
        optimizer_cls1 = get_optimizer(self.cfg, phase='searching', optimizer_type='model_optimizer')
        optimizer_params1 = {k: v for k, v in self.cfg['searching']['model_optimizer'].items()
                             if k != 'name'}

        self.model_optimizer = optimizer_cls1(self.model.parameters(), **optimizer_params1)
        self.logger.info("Using model optimizer {}".format(self.model_optimizer))

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer,
                                                                    self.cfg['searching']['epoch'])

        # Setup optimizer, lr_scheduler and loss function for architecture
        optimizer_cls2 = get_optimizer(self.cfg, phase='searching', optimizer_type='arch_optimizer')
        optimizer_params2 = {k: v for k, v in self.cfg['searching']['arch_optimizer'].items()
                             if k != 'name'}

        self.arch_optimizer = optimizer_cls2(self.model.arch_parameters(), **optimizer_params2)

        self.architect = Architecture(self.model, arch_optimizer=self.arch_optimizer, criterion=self.criterion)

    def _check_resume(self):
        self.dur_time = 0
        self.start_epoch = 0
        self.patience = 0
        self.geno_type = ''
        # optionally resume from a checkpoint for model
        if self.cfg['searching']['resume'] is not None:
            if os.path.isfile(self.cfg['searching']['resume']):
                self.logger.info(
                    "Loading model and optimizer from checkpoint '{}'".format(
                        self.cfg['searching']['resume']
                    )
                )
                checkpoint = torch.load(self.cfg['searching']['resume'], map_location=self.device)
                self.start_epoch = checkpoint['epoch']
                self.dur_time = checkpoint['dur_time']
                self.patience = 0
                self.geno_type = checkpoint['geno_type']
                self.architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.model.load_state_dict(checkpoint['model_state'])
                self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
                self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
                self.model.load_params(checkpoint['alphas_dict'], checkpoint['betas_dict'])
            else:
                self.logger.info("No checkpoint found at '{}'".format(self.cfg['searching']['resume']))

    def run(self):
        self.logger.info('args = {}'.format(self.cfg))
        # Setup Metrics
        self.metric_train = SegmentationMetric(self.n_classes)
        self.metric_val = SegmentationMetric(self.n_classes)
        self.train_loss_meter = AverageMeter()
        self.val_loss_meter = AverageMeter()
        self.geno_type = None
        run_start = time.time()

        for epoch in range(self.start_epoch, self.cfg['searching']['epoch']):
            self.epoch = epoch
            self.logger.info('Epoch %d / %d lr %e', self.epoch,
                             self.cfg['searching']['epoch'], self.scheduler.get_last_lr()[-1])

            # get genotype
            genotype = self.model.genotype()
            self.logger.info('genotype = %s', genotype)
            print('alpha down normal:', F.softmax(self.model.alphas_dn_nm, dim=-1))
            print('alpha down:', F.softmax(self.model.alphas_dn, dim=-1))
            print('alpha up normal:', F.softmax(self.model.alphas_up_nm, dim=-1))
            print('alpha up:', F.softmax(self.model.alphas_up, dim=-1))
            print('betas down:', F.softmax(self.model.betas_dn, dim=-1))
            print('betas up:', F.softmax(self.model.betas_up, dim=-1))
            print('gamma:', F.softmax(self.model.gamma, dim=-1))

            # the performance may be unstable, before train in a degree
            if self.epoch >= self.cfg['searching']['alpha_begin']:
                # check whether the genotype has changed
                if self.geno_type == genotype:
                    self.patience += 1
                else:
                    self.patience = 0
                    self.geno_type = genotype

                self.logger.info('Current patience :{}'.format(self.patience))

                if self.patience >= self.cfg['searching']['max_patience']:
                    self.logger.info('Reach the max patience! \n best genotype {}'.format(self.geno_type))
                    break

            # train and search the model
            self.train()

            # valid the model
            self.infer()

            if self.epoch % self.cfg['searching']['report_freq'] == 0:
                self.logger.info('GPU memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))

            save_checkpoint({
                'epoch': epoch + 1,
                'dur_time': self.dur_time + time.time() - run_start,
                'cur_patience': self.patience,
                'geno_type': self.geno_type,
                'model_state': self.model.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'model_optimizer': self.model_optimizer.state_dict(),
                'alphas_dict': self.model.alphas_dict(),
                'betas_dict': self.model.betas_dict(),
                'scheduler': self.scheduler.state_dict()
            }, False, self.save_path)
            self.logger.info('save checkpoint (epoch %d) in %s  dur_time: %s'
                             , epoch, self.save_path, calc_time(self.dur_time + time.time() - run_start))

            self.metric_train.reset()
            self.metric_val.reset()
            self.val_loss_meter.reset()
            self.train_loss_meter.reset()

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.save_tbx_log + "/all_scalars.json")
        self.writer.close()
        self.logger.info('End! \n best genotype {}'.format(self.geno_type))

    def train(self):
        self.model.train()
        tbar = tqdm(self.train_queue)
        for step, (input, target) in enumerate(tbar):

            input = input.to(self.device)
            target = target.to(self.device)

            # Get a random mini-batch from the search queue
            input_valid, target_valid = next(iter(self.valid_queue))
            input_valid = input_valid.to(self.device)
            target_valid = target_valid.to(self.device)

            # Update the architecture parameters first!
            # Update the architecture parameters when the model weights
            # trained in a degree
            if self.epoch >= self.cfg['searching']['alpha_begin']:
                self.architect.step(input_valid, target_valid)

            self.model_optimizer.zero_grad()
            predicts = self.model(input)

            # self.logger.info('GPU memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))

            train_loss = self.criterion(predicts, target)

            self.train_loss_meter.update(train_loss.item())
            self.metric_train.update(target, predicts[-1])

            train_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.cfg['searching']['grad_clip'])

            if step % self.cfg['searching']['report_freq'] == 0:
                pixAcc, mIoU, dice = self.metric_train.get()
                self.logger.info('Train %03d %e | epoch[%d]/[%d]', step + 1,
                                 self.train_loss_meter.avg, self.epoch, self.cfg['searching']['epoch'])
                tbar.set_description('Train loss: %.3f' % (self.train_loss_meter.avg))
                self.logger.info('Train pixAcc: %.3f; mIoU: %.5f; dice: %.5f' % (pixAcc, mIoU, dice))

            # Update the network parameters
            self.model_optimizer.step()

        # update scheduler
        self.scheduler.step()
        _, _, dice = self.metric_train.get()
        self.writer.add_scalar('Train/Loss', self.train_loss_meter.avg, self.epoch)
        self.writer.add_scalar('Train/dice', dice, self.epoch)

    def infer(self):

        self.model.eval()
        tbar = tqdm(self.valid_queue)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.to(self.device)
                target = target.to(self.device)
                predicts = self.model(input)

                val_loss = self.criterion(predicts, target)
                self.val_loss_meter.update(val_loss.item())

                self.metric_val.update(target, predicts[-1])
                if step % self.cfg['searching']['report_freq'] == 0:
                    pixAcc, mIoU, dice = self.metric_val.get()
                    loss_v = self.val_loss_meter.avg
                    self.logger.info(
                        'Val loss: %.6f; pixAcc: %.3f; mIoU: %.5f; dice: %.5f' % (loss_v, pixAcc, mIoU, dice))
                    tbar.set_description(
                        'Val loss: %.6f; pixAcc: %.3f; mIoU: %.5f; dice: %.5f' % (loss_v, pixAcc, mIoU, dice))

        pixAcc, mIoU, dice = self.metric_val.get()
        cur_loss = self.val_loss_meter.mloss()
        self.logger.info(
            'Epoch {} Val loss: {}, pixAcc: {}, mIoU: {}, dice: {}'.format(self.epoch, cur_loss, pixAcc, mIoU, dice))
        self.writer.add_scalar('Val/pixAcc', pixAcc, self.epoch)
        self.writer.add_scalar('Val/mIoU', mIoU, self.epoch)
        self.writer.add_scalar('Val/dice', dice, self.epoch)
        self.writer.add_scalar('Val/loss', cur_loss, self.epoch)


if __name__ == '__main__':
    search_network = SearchNetwork()
    search_network.run()
