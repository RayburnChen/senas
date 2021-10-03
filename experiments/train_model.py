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
from util.loss.loss import SegmentationLosses, MultiSegmentationLosses
from util.datasets import get_dataset
from util.utils import get_logger, save_checkpoint, calc_time, store_images, gpu_memory, complexity_info, stat_info
from util.utils import weights_init
from util.utils import get_gpus_memory_info, calc_parameters_count
from util.schedulers import get_scheduler
from util.optimizers import get_optimizer
from util.challenge.promise12.store_test_seg import predict_test
from util.metrics import *
from models import get_segmentation_model
import models.geno_searched as geno_types
from util import genotype
from util.gpu_memory_log import gpu_memory_log
from tensorboardX import SummaryWriter


class Network(object):

    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._check_resume()

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')

        # Add default argument
        parser.add_argument('--config', nargs='?', type=str, default='../configs/senas/senas_chaos.yml',
                            help='Configuration file to use')
        parser.add_argument('--model', nargs='?', type=str, default='senas', help='Model to train and evaluation')
        parser.add_argument('--ft', action='store_true', default=False, help='fine tuning on a different dataset')
        parser.add_argument('--warm', nargs='?', type=int, default=0, help='warm up from pre epoch')
        parser.add_argument('--genotype', nargs='?', type=str, default='', help='Model architecture')
        parser.add_argument('--loss', nargs='?', type=str, default='', help='Loss function')
        parser.add_argument('--depth', nargs='?', type=int, default=-1, help='Loss function')
        parser.add_argument('--batch_size', nargs='?', type=int, default=-1, help='Batch size')

        self.args = parser.parse_args()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp, Loader=yaml.FullLoader)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        print('Usage model :{}'.format(self.model_name))

    def _init_logger(self):
        log_dir = '../logs/' + self.model_name + '/train' + '/{}'.format(self.cfg['data']['dataset']) \
                  + '/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        self.logger = get_logger(log_dir)
        self.logger.info('RUNDIR: {}'.format(log_dir))
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.save_image_path = os.path.join(self.save_path, 'saved_val_images')
        self.writer = SummaryWriter(self.save_tbx_log)
        shutil.copy(self.args.config, self.save_path)

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)

        self.logger.info('seed is {}'.format(self.cfg.get('seed', 0)))
        np.random.seed(self.cfg.get('seed', 0))
        torch.manual_seed(self.cfg.get('seed', 0))
        torch.cuda.manual_seed(self.cfg.get('seed', 0))
        cudnn.enabled = True
        cudnn.benchmark = True
        self.device_id, self.gpus_info = get_gpus_memory_info()
        self.device = torch.device('cuda:{}'.format(0 if self.cfg['training']['multi_gpus'] else self.device_id))

    def _init_dataset(self):
        trainset = get_dataset(self.cfg['data']['dataset'], split='train', mode='train')
        valset = get_dataset(self.cfg['data']['dataset'], split='val', mode='val')
        # testset = get_dataset(self.cfg['data']['dataset'], split='test', mode='test')
        self.nweight = trainset.class_weight
        self.logger.info('dataset weights: {}'.format(self.nweight))
        self.n_classes = trainset.num_class
        self.batch_size = self.args.batch_size if self.args.batch_size > 0 else self.cfg['training']['batch_size']
        kwargs = {'num_workers': self.cfg['training']['n_workers'], 'pin_memory': True}

        # Split val dataset
        if self.cfg['data']['dataset'] in ['bladder', 'chaos', 'ultrasound_nerve']:
            num_train = len(trainset)
            indices = list(range(num_train))
            split = int(np.floor(0.8 * num_train))
            self.logger.info('split training data : 0.8')
            self.train_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                               **kwargs)

            self.valid_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   indices[split:num_train]),
                                               **kwargs)
        else:
            self.train_queue = data.DataLoader(trainset, batch_size=self.batch_size,
                                               drop_last=True, shuffle=True, **kwargs)

            self.valid_queue = data.DataLoader(valset, batch_size=self.batch_size,
                                               drop_last=False, shuffle=False, **kwargs)

    def _init_model(self):

        # Setup loss function
        depth = self.args.depth if self.args.depth > 0 else self.cfg['training']['depth']
        supervision = self.cfg['training']['deep_supervision']
        loss_name = self.args.loss if len(self.args.loss) > 0 else self.cfg['training']['loss']['name']
        criterion = MultiSegmentationLosses(loss_name, depth) if supervision else SegmentationLosses(loss_name)
        self.criterion = criterion.to(self.device)
        self.logger.info("Using loss {}".format(loss_name))

        self.show_dice_coeff = False
        if self.cfg['data']['dataset'] in ['bladder', 'chaos', 'ultrasound_nerve', 'promise12']:
            self.show_dice_coeff = True

        # Setup Model
        init_channels = self.cfg['training']['init_channels']
        if len(self.args.genotype) > 0:
            self.genotype = eval('genotype.%s' % self.args.genotype)
        else:
            self.genotype = eval('geno_types.%s' % self.cfg['training']['geno_type'])
        self.logger.info('Using genotype: {}'.format(self.genotype))
        model = get_segmentation_model(self.model_name,
                                       dataset=self.cfg['data']['dataset'],
                                       backbone=self.cfg['training']['backbone'],
                                       c=init_channels,
                                       depth=depth,
                                       supervision=supervision,
                                       genotype=self.genotype,
                                       double_down_channel=self.cfg['training']['double_down_channel']
                                       )

        # init weight using hekming methods
        model.apply(weights_init)
        self.logger.info('Initialize the model weights: kaiming_uniform')

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)

        # stat_info(model, (1, 256, 256))
        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

        # Setup optimizer, lr_scheduler for model
        optimizer_cls = get_optimizer(self.cfg, phase='training', optimizer_type='model_optimizer')
        optimizer_params = {k: v for k, v in self.cfg['training']['model_optimizer'].items()
                            if k != 'name'}

        self.model_optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)
        self.logger.info("Using model optimizer {}".format(self.model_optimizer))
        # self.logger.info('Computational complexity:{}, Number of parameters:{}'.format(*complexity_info(self.model, (1, 256, 256))))

    def _check_resume(self):
        self.dur_time = 0
        self.start_epoch = 0
        self.best_mIoU, self.best_loss, self.best_pixAcc, self.best_dice = 0, 1.0, 0, 0
        # optionally resume from a checkpoint for model
        resume = self.cfg['training']['resume'] if self.cfg['training']['resume'] is not None else None
        if resume is not None:
            if os.path.isfile(resume):
                self.logger.info("Loading model and optimizer from checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=self.device)
                if not self.args.ft:  # no fine-tuning
                    self.start_epoch = checkpoint['epoch']
                    self.dur_time = checkpoint['dur_time']
                    self.best_mIoU = checkpoint['best_mIoU']
                    self.best_pixAcc = checkpoint['best_pixAcc']
                    self.best_loss = checkpoint['best_loss']
                    self.best_dice = checkpoint['best_dice_coeff']
                    self.model_optimizer.load_state_dict(checkpoint['model_optimizer'])
                self.model.load_state_dict(checkpoint['model_state'])
            else:
                self.logger.info("No checkpoint found at '{}'".format(resume))

        # init LR_scheduler
        scheduler_params = {k: v for k, v in self.cfg['training']['lr_schedule'].items()}
        if 'max_iter' in self.cfg['training']['lr_schedule']:
            scheduler_params['max_iter'] = self.cfg['training']['epoch']
            # Note: For step in train epoch !!!!  must use the value below
            # scheduler_params['max_iter'] = len(self.train_queue) * self.cfg['training']['epoch'] \
            #                                // self.cfg['training']['batch_size']
        if 'T_max' in self.cfg['training']['lr_schedule']:
            scheduler_params['T_max'] = self.cfg['training']['epoch']
            # Note: For step in train epoch !!!!  must use the value below
            # scheduler_params['T_max'] = len(self.train_queue) * self.cfg['training']['epoch'] \
            #                                // self.cfg['training']['batch_size']

        scheduler_params['last_epoch'] = -1 if self.start_epoch == 0 else self.start_epoch
        self.scheduler = get_scheduler(self.model_optimizer, scheduler_params)

    def run(self):
        self.logger.info('args = %s', self.cfg)
        # Setup Metrics
        self.metric_train = SegmentationMetric(self.n_classes)
        self.metric_val = SegmentationMetric(self.n_classes)
        self.metric_test = SegmentationMetric(self.n_classes)
        self.val_loss_meter = AverageMeter()
        self.test_loss_meter = AverageMeter()
        self.train_loss_meter = AverageMeter()
        self.patience = 0
        self.save_best = True
        run_start = time.time()

        # Set up results folder
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)

        for epoch in range(self.start_epoch, self.cfg['training']['epoch']):
            self.epoch = epoch

            self.scheduler.step()

            self.logger.info('=> Epoch {}, lr {}'.format(self.epoch, self.scheduler.get_last_lr()[-1]))

            # train and search the model
            self.train()

            # valid the model
            self.val()

            self.logger.info('Best loss {}, pixAcc {}, mIoU {}, dice {}'.format(
                self.best_loss, self.best_pixAcc, self.best_mIoU, self.best_dice
            ))

            if self.epoch % self.cfg['training']['report_freq'] == 0:
                self.logger.info('GPU memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))

            if self.save_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'dur_time': self.dur_time + time.time() - run_start,
                    'model_state': self.model.state_dict(),
                    'model_optimizer': self.model_optimizer.state_dict(),
                    'best_pixAcc': self.best_pixAcc,
                    'best_mIoU': self.best_mIoU,
                    'best_dice_coeff': self.best_dice,
                    'best_loss': self.best_loss,
                }, True, self.save_path)
                self.logger.info('save checkpoint (epoch %d) in %s  dur_time: %s',
                                 epoch, self.save_path, calc_time(self.dur_time + time.time() - run_start))
                self.save_best = False

            if self.patience == self.cfg['training']['max_patience'] or epoch == self.cfg['training']['epoch'] - 1:
                # load best model weights
                # self._check_resume(os.path.join(self.save_path, 'checkpint.pth.tar'))
                # # Test
                # if len(self.test_queue) > 0:
                #     self.logger.info('Training ends \n Test')
                #     self.test()
                # else:
                #     self.logger.info('Training ends!')
                self.logger.info('Early stopping')
                break
            else:
                self.logger.info('Current patience :{}'.format(self.patience))

            self.val_loss_meter.reset()
            self.train_loss_meter.reset()
            self.metric_train.reset()
            self.metric_val.reset()
            self.logger.info('Cost time: {}'.format(calc_time(self.dur_time + time.time() - run_start)))

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.save_tbx_log + "/all_scalars.json")
        self.writer.close()
        self.logger.info('Best loss {}, pixAcc {}, mIoU {}, dice {}'.format(
                self.best_loss, self.best_pixAcc, self.best_mIoU, self.best_dice
            ))
        self.logger.info('Cost time: {}'.format(calc_time(self.dur_time + time.time() - run_start)))
        self.logger.info('Log dir in : {}'.format(self.save_path))

    def train(self):
        self.model.train()
        tbar = tqdm(self.train_queue)
        for step, (input, target) in enumerate(tbar):

            self.model_optimizer.zero_grad()

            input = input.cuda(self.device)
            target = target.cuda(self.device)

            predicts = self.model(input)

            self.logger.info('GPU memory total:{}, reserved:{}, allocated:{}, waiting:{}'.format(*gpu_memory()))

            train_loss = self.criterion(predicts, target)

            self.train_loss_meter.update(train_loss.item())

            self.metric_train.update(target, predicts[-1])

            train_loss.backward()

            if self.cfg['training']['grad_clip']:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg['training']['grad_clip'])

            if step % self.cfg['training']['report_freq'] == 0:
                self.logger.info('Train loss %03d %e | epoch [%d] / [%d]', step,
                                 self.train_loss_meter.mloss(), self.epoch, self.cfg['training']['epoch'])
                pixAcc, mIoU, dice = self.metric_train.get()
                self.logger.info('Train pixAcc: {}, mIoU: {}, dice: {}'.format(pixAcc, mIoU, dice))
                tbar.set_description('train loss: %.6f; pixAcc: %.3f; mIoU %.6f; dice %.6f;' % (
                    self.train_loss_meter.mloss(), pixAcc, mIoU, dice))

            # Update the network parameters
            self.model_optimizer.step()

        # save in tensorboard scalars
        _, _, dice = self.metric_train.get()
        self.writer.add_scalar('Train/loss', self.train_loss_meter.mloss(), self.epoch)
        self.writer.add_scalar('Train/dice', dice, self.epoch)

    def val(self):
        self.model.eval()
        tbar = tqdm(self.valid_queue)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda(self.device)
                target = target.cuda(self.device)
                predicts = self.model(input)

                val_loss = self.criterion(predicts, target)

                self.val_loss_meter.update(val_loss.item())

                self.metric_val.update(target, predicts[-1])

                if step % self.cfg['training']['report_freq'] == 0:
                    self.logger.info('Val loss %03d %e | epoch [%d] / [%d]', step,
                                 self.val_loss_meter.mloss(), self.epoch, self.cfg['training']['epoch'])
                    pixAcc, mIoU, dice = self.metric_val.get()

                    self.logger.info('Val pixAcc: {}, mIoU: {}, dice: {}'.format(pixAcc, mIoU, dice))
                    tbar.set_description('val loss: %.6f, pixAcc: %.3f, mIoU: %.6f, dice: %.6f'
                                         % (self.val_loss_meter.mloss(), pixAcc, mIoU, dice))

        grid_image = store_images(input, predicts[-1], target)
        self.writer.add_image('Val', grid_image, self.epoch)

        # save in tensorboard scalars
        pixAcc, mIoU, dice = self.metric_val.get()
        cur_loss = self.val_loss_meter.mloss()
        self.logger.info('Val pixAcc: {}, mIoU: {}, dice: {}'.format(pixAcc, mIoU, dice))
        self.writer.add_scalar('Val/Acc', pixAcc, self.epoch)
        self.writer.add_scalar('Val/mIoU', mIoU, self.epoch)
        self.writer.add_scalar('Val/dice', dice, self.epoch)
        self.writer.add_scalar('Val/loss', cur_loss, self.epoch)

        # for early-stopping
        if self.best_dice < dice or self.best_mIoU < mIoU:
            # Store best score
            self.patience = 0
            self.best_mIoU = mIoU
            self.best_dice = dice
            self.best_pixAcc = pixAcc
            self.best_loss = cur_loss
            self.save_best = True
        else:
            self.patience += 1

    def test(self):
        self.model.eval()
        predict_list = []
        tbar = tqdm(self.test_queue)
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda(self.device)
                if not isinstance(target, list) and not isinstance(target, str):
                    target = target.cuda(self.device)
                elif isinstance(target, str):
                    target = target.split('.')[0] + '_mask.tiff'

                predicts = self.model(input)

                test_loss = self.criterion(predicts, target)
                self.test_loss_meter.update(test_loss.item())
                self.metric_test.update(target, predicts[-1])

        # save images
        if not isinstance(target, list):
            grid_image = store_images(input, predicts[-1], target)
            self.writer.add_image('Test', grid_image, self.epoch)
            pixAcc, mIoU, dice = self.metric_test.get()
            self.logger.info('Test/loss: {}, pixAcc: {}, mIoU: {}, dice: {}'.format(
                self.test_loss_meter.mloss(), pixAcc, mIoU, dice))
        else:
            predict_test(predict_list, target, self.save_path + '/test_rst')


if __name__ == '__main__':
    train_network = Network()
    train_network.run()


