from utils.genotype import Genotype

senas_node_2 = Genotype(down=[('dil_2_conv_5', 1), ('dil_2_conv_5', 0), ('dil_3_conv_5', 2), ('dil_3_conv_5', 0)], down_concat=range(2, 4), up=[('dil_3_conv_5', 1), ('dil_3_conv_5', 0), ('dil_2_conv_5', 0), ('dil_3_conv_5', 2)], up_concat=range(2, 4), gamma=[0, 0, 1, 1, 1, 1])

senas_node_3 = Genotype(down=[('se_conv_3', 1), ('dil_2_conv_5', 0), ('dil_3_conv_5', 0), ('dil_2_conv_5', 2), ('dil_3_conv_5', 0), ('dil_2_conv_5', 3)], down_concat=range(2, 5), up=[('up_sample', 1), ('dil_3_conv_5', 0), ('up_sample', 1), ('dil_3_conv_5', 2), ('up_sample', 1), ('dep_sep_conv_3', 3)], up_concat=range(2, 5), gamma=[1, 0, 1, 0, 1, 1])

senas_node_4 = Genotype(down=[('se_conv_3', 1), ('avg_pool', 0), ('dil_3_conv_5', 2), ('dep_sep_conv_5', 1), ('dil_3_conv_5', 2), ('avg_pool', 0), ('avg_pool', 1), ('dil_3_conv_5', 3)], down_concat=range(2, 6), up=[('up_sample', 1), ('dil_3_conv_5', 0), ('dil_3_conv_5', 0), ('dil_2_conv_5', 2), ('dil_3_conv_5', 1), ('dil_2_conv_5', 2), ('dep_sep_conv_3', 0), ('dil_2_conv_5', 4)], up_concat=range(2, 6), gamma=[0, 0, 0, 1, 1, 1])


senas = senas_node_3

