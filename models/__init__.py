from segmentation_models_pytorch import UnetPlusPlus, FPN, Linknet, MAnet, PSPNet, Unet, PAN, DeepLabV3Plus
from .nasunet.nas_unet import NasUnet
from .senas_model import *




def unet(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    decod = (256, 128, 64, 32, 16, 8, 4, 2)[:depth]
    model = Unet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth, decoder_channels=decod)
    return model


def unet_plus_plus(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    decod = (256, 128, 64, 32, 16, 8, 4, 2)[:depth]
    model = UnetPlusPlus(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth, decoder_channels=decod)
    return model


def senas(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    model = SenasModel(datasets[dataset.lower()].NUM_CLASS, datasets[dataset.lower()].IN_CHANNELS,
                       **kwargs)
    return model


def nasunet(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    model = NasUnet(nclass=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, depth=depth)
    return model


def deeplab_v3_plus(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    model = DeepLabV3Plus(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10')
    return model


def fpn(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    upsampling = 2**(depth-3)
    model = FPN(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth, upsampling=upsampling)
    return model


def manet(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    decod = (256, 128, 64, 32, 16, 8, 4, 2)[:depth]
    model = MAnet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth, decoder_channels=decod)
    return model


def linknet(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    decod = (256, 128, 64, 32, 16, 8, 4, 2)[:depth]
    model = Linknet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth, decoder_channels=decod)
    return model


def pspnet(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    model = PSPNet(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth)
    return model


def pan(dataset, **kwargs):
    # infer number of classes
    from utils.datasets import datasets
    depth = kwargs['depth']
    upsampling = 2**(depth-3)
    model = PAN(classes=datasets[dataset.lower()].NUM_CLASS, in_channels=datasets[dataset.lower()].IN_CHANNELS, encoder_weights=None, encoder_name='resnet10', encoder_depth=depth, upsampling=upsampling)
    return model


def get_segmentation_model(name, **kwargs):
    models = {
        'senas': senas,
        'unet': unet,
        'unet_plus_plus': unet_plus_plus,
        'deeplab_v3_plus': deeplab_v3_plus,
        'fpn': fpn,
        'linknet': linknet,
        'manet': manet,
        'pspnet': pspnet,
        'pan': pan,
        'nasunet': nasunet,
    }
    return models[name.lower()](**kwargs)
