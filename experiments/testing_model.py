import argparse
import datetime
import os
import sys
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from torch.utils import data
from tqdm import tqdm

sys.path.append('..')
from utils.loss.loss import SegmentationLosses, MultiSegmentationLosses
from utils.datasets import get_dataset
from utils.utils import get_logger, store_images
from utils.utils import get_gpus_memory_info, calc_parameters_count, create_exp_dir
from models import get_segmentation_model
from utils.metrics import *
from torchvision import transforms
import models.geno_searched as geno_types
from utils import genotype
from PIL import Image


class RunNetwork(object):
    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        if not self._check_resume():
            self.logger.error('The pre-trained model not exist!!!')
            exit(-1)

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='config')

        # Add default argument
        parser.add_argument('--config', nargs='?', type=str, default='../configs/senas/senas_chaos.yml',
                            help='Configuration file to use')
        parser.add_argument('--model', nargs='?', type=str, default='senas', help='Model to testing')
        parser.add_argument('--genotype', nargs='?', type=str,
                            default="Genotype(down=[('se_conv_3', 1), ('avg_pool', 0), ('dil_3_conv_5', 2), ('dep_sep_conv_5', 1), ('dil_3_conv_5', 2), ('avg_pool', 0), ('avg_pool', 1), ('dil_3_conv_5', 3)], down_concat=range(2, 6), up=[('up_sample', 1), ('dil_3_conv_5', 0), ('dil_3_conv_5', 0), ('dil_2_conv_5', 2), ('dil_3_conv_5', 1), ('dil_2_conv_5', 2), ('dep_sep_conv_3', 0), ('dil_2_conv_5', 4)], up_concat=range(2, 6), gamma=[0, 0, 0, 1, 1, 1])",
                            help='Model architecture')
        parser.add_argument('--loss', nargs='?', type=str, default='', help='Loss function')
        parser.add_argument('--depth', nargs='?', type=int, default=-1, help='Loss function')
        parser.add_argument('--batch_size', nargs='?', type=int, default=6, help='Batch size')
        parser.add_argument('--resume', nargs='?', type=str,
                            default='../logs/senas/train/chaos/20211017-170505-615151/model_best.pth.tar',
                            help='Resume path')

        self.args = parser.parse_args()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp, Loader=yaml.FullLoader)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        print('Usage model :{}'.format(self.model_name))

    def _init_logger(self):
        log_dir = '../logs/' + self.model_name + '/testing' + '/{}'.format(self.cfg['data']['dataset']) \
                  + '/{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_image_path = os.path.join(self.save_path, 'saved_val_images')

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)

        np.random.seed(self.cfg.get('seed', 0))
        torch.manual_seed(self.cfg.get('seed', 0))
        torch.cuda.manual_seed(self.cfg.get('seed', 0))
        cudnn.enabled = True
        cudnn.benchmark = True
        self.device_id, self.gpus_info = get_gpus_memory_info()
        self.device = torch.device('cuda:{}'.format(0 if self.cfg['training']['multi_gpus'] else self.device_id))

    def _init_dataset(self):
        self.trainset = get_dataset(self.cfg['data']['dataset'], split='train', mode='train')
        self.valset = get_dataset(self.cfg['data']['dataset'], split='val', mode='val')
        self.testingset = get_dataset(self.cfg['data']['dataset'], split='testing', mode='testing')
        self.nweight = self.trainset.class_weight
        self.n_classes = self.trainset.num_class
        self.batch_size = self.args.batch_size if self.args.batch_size > 0 else self.cfg['training']['batch_size']
        kwargs = {'num_workers': self.cfg['training']['n_workers'], 'pin_memory': True}

        self.train_queue = data.DataLoader(self.trainset, batch_size=self.batch_size, drop_last=False, shuffle=False,
                                           **kwargs)

        self.valid_queue = data.DataLoader(self.valset, batch_size=self.batch_size, drop_last=False, shuffle=False,
                                           **kwargs)

        self.testing_queue = data.DataLoader(self.testingset, batch_size=self.batch_size, drop_last=False,
                                             shuffle=False,
                                             **kwargs)

    def _init_model(self):
        # Setup loss function
        depth = self.args.depth if self.args.depth > 0 else self.cfg['training']['depth']
        supervision = self.cfg['training']['deep_supervision']
        loss_name = self.args.loss if len(self.args.loss) > 0 else self.cfg['training']['loss']['name']
        criterion = MultiSegmentationLosses(loss_name, depth) if supervision else SegmentationLosses(loss_name)
        self.criterion = criterion.to(self.device)
        self.logger.info("Using loss {}".format(loss_name))

        # Setup Model
        init_channels = self.cfg['training']['init_channels']
        if len(self.args.genotype) > 0:
            self.genotype = eval('genotype.%s' % self.args.genotype)
        else:
            self.genotype = eval('geno_types.%s' % self.cfg['training']['geno_type'])
        self.logger.info('Using genotype: {}'.format(self.genotype))

        model = get_segmentation_model(self.model_name,
                                       dataset=self.cfg['data']['dataset'],
                                       c=init_channels,
                                       depth=depth,
                                       supervision=False,
                                       genotype=self.genotype,
                                       double_down_channel=self.cfg['training']['double_down_channel']
                                       )
        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

    def _check_resume(self):
        resume = self.args.resume
        if resume is not None:
            if os.path.isfile(resume):
                self.logger.info("Loading model and optimizer from checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state'])
                return True
            else:
                self.logger.info("No checkpoint found at '{}'".format(resume))
                return False
        return False

    def testing(self, img_queue, split='val', path=''):
        self.model.eval()
        tbar = tqdm(img_queue)
        print('Save prediction image on : {}'.format(path))
        with torch.no_grad():
            for step, (input, target) in enumerate(tbar):
                input = input.cuda(self.device)
                if not isinstance(target, list):
                    target = target.cuda(self.device)

                predicts = self.model(input)

                if not isinstance(target, list):
                    testing_loss = self.criterion(predicts, target)
                    self.loss_meter.update(testing_loss.item())
                    self.metric.update(target, predicts[-1])
                    if step % self.cfg['training']['report_freq'] == 0:
                        pixAcc, mIoU, dice = self.metric.get()
                        self.logger.info('{} loss: {}, pixAcc: {}, mIoU: {}, dice: {}'.format(
                            split, self.loss_meter.mloss(), pixAcc, mIoU, dice))
                        tbar.set_description('loss: %.6f, pixAcc: %.3f, mIoU: %.6f, dice: %.6f'
                                             % (self.loss_meter.mloss(), pixAcc, mIoU, dice))

                    # N = predicts[-1].shape[0]
                    # for i in range(N):
                    #     img = Image.fromarray(
                    #         (torch.argmax(predicts[-1].cpu(), 1)[i] * 255).numpy().astype(np.uint8))
                    #     file_name = str(step) + '_' + str(i) + '_mask.png'
                    #     img.save(os.path.join(path, file_name), format="png")

                pixAcc, mIoU, dice = self.metric.get()
                print('==> dice: {}'.format(dice))

                # save grid_image
                if not isinstance(target, list) and not isinstance(target, str):  #
                    grid_image = store_images(input, predicts[-1], target)
                    im = transforms.ToPILImage()(grid_image).convert("RGB")
                    im.save(os.path.join(path, 'grid_' + str(step)+'.png'), format="png")
                    pixAcc, mIoU, dice = self.metric.get()
                    self.logger.info('{}/loss: {}, pixAcc: {}, mIoU: {}, dice: {}'.format(
                        split, self.loss_meter.mloss(), pixAcc, mIoU, dice))

    def run(self):
        self.logger.info('args = %s', self.cfg)
        # Setup Metrics
        self.metric = SegmentationMetric(self.n_classes)
        self.loss_meter = AverageMeter()
        run_start = time.time()
        # Set up results folder
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)

        if len(self.train_queue) != 0:
            self.logger.info('Begin train set evaluation')
            self.testing(self.train_queue, split='train', path=self.save_image_path)
        # if len(self.valid_queue) != 0:
        #     self.logger.info('Begin valid set evaluation')
        #     self.testing(self.valid_queue, split='val', path=self.save_image_path)
        # if len(self.testing_queue) != 0:
        #     self.logger.info('Begin testing set evaluation')
        #     self.testing(self.testing_queue, split='testing', path=self.save_image_path)
        self.logger.info('Evaluation done!')


if __name__ == '__main__':
    testing_network = RunNetwork()
    testing_network.run()
