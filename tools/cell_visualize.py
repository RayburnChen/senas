import datetime
import sys
import os
import platform

from util import visualize
from util.genotype import Genotype


def main(format):
    if 'Windows' in platform.platform():
        os.environ['PATH'] += os.pathsep + '../3rd_tools/graphviz-2.38/bin/'
    try:
        genotype = Genotype(down=[('dep_sep_conv_3', 0), ('dep_sep_conv_3', 1), ('avg_pool', 0), ('dil_5_conv_5', 2), ('dil_5_conv_5', 3), ('dil_5_conv_5', 2)], down_concat=range(2, 5), up=[('up_sample', 1), ('dil_5_conv_5', 0), ('dil_5_conv_5', 0), ('dil_5_conv_5', 2), ('dil_5_conv_5', 2), ('dil_5_conv_5', 3)], up_concat=range(2, 5), gamma=[1, 0, 0, 1, 1, 0])


    except AttributeError:
        print('{} is not specified in genotype.py'.format(genotype))
        sys.exit(1)

    down_cell_name = '{}-{}'.format("DownC", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    up_cell_name = '{}-{}'.format("UpC", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    visualize.plot(genotype.down, down_cell_name, format=format, directory="./cell_visualize")
    visualize.plot(genotype.up, up_cell_name, format=format, directory="./cell_visualize")


if __name__ == '__main__':
    # support {'jpeg', 'png', 'pdf', 'tiff', 'svg', 'bmp', 'tif', 'tiff'}
    main(format='pdf')
