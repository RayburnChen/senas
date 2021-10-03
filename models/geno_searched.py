from util.genotype import Genotype

senas_node_2 = Genotype(down=[('down_dil_conv', 0), ('down_dil_conv', 1), ('down_cweight', 1), ('dil_conv', 2)], down_concat=range(2, 4), up=[('dil_conv', 0), ('up_dil_conv', 1), ('dil_conv', 0), ('dil_conv', 2)], up_concat=range(2, 4), gamma=[0, 0, 0, 0, 0, 0])

senas_node_3 = Genotype(down=[('down_dil_conv', 0), ('down_dil_conv', 1), ('down_dil_conv', 0), ('dil_conv', 2), ('dil_conv', 2), ('dil_conv', 3)], down_concat=range(2, 5), up=[('dil_conv', 0), ('up_dil_conv', 1), ('up_dil_conv', 1), ('dil_conv', 2), ('up_dil_conv', 1), ('dil_conv', 3)], up_concat=range(2, 5), gamma=[1, 1, 1, 0, 1, 1])

senas = senas_node_3

