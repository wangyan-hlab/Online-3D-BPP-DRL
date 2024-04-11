import numpy as np
import torch
from sympy.utilities.iterables import multiset_permutations

# real full height post boxes
# lwh
# BOX_FH_SIZES = {
#     "1":[530, 290, 370], "2":[530, 230, 290], "3":[430, 210, 270],
#     "4":[350, 190, 230], "5":[290, 170, 190], "6":[260, 150, 180],
#     "7":[230, 130, 160], "8":[210, 110, 140], "9":[195, 105, 135],
#     "10":[175, 95, 115], "11":[145, 85, 105], "12":[130, 80, 90]
#     }
# lhw
BOX_FH_SIZES = {
    "1":[530, 370, 290], "2":[530, 290, 230], "2.5":[410, 300, 300], "3":[430, 270, 210],
    "4":[350, 230, 190], "5":[290, 190, 170], "6":[260, 180, 150],
    "7":[230, 160, 130], "8":[210, 140, 110], "9":[195, 135, 105],
    "10":[175, 115, 95], "11":[145, 105, 85], "12":[130, 90, 80]
    }

# real half-height post boxes
# lwh
BOX_HH_SIZES = {
    "3.5":[430, 210, 270], "4.5":[350, 190, 115], "5.5":[290, 170, 95],
    "6.5":[260, 150, 90], "7.5":[230, 130, 80], "8.5":[210, 110, 70],
    "9.5":[195, 105, 67], "10.5":[175, 95, 57], "11.5":[145, 85, 53], 
    "12.5":[130, 80, 45]
    }


def vanilla_boxgen(box_seq_len, box_real_sizes, cases, bin_real_size=1000, 
                   grid_num=25, box_type_num_range=[0, 12], permute=False, save=False):
    """
        Given a dict of actual box types and sizes:
        1. parameterizing the sizes into grid value
        2. randomly generate several box sequences

        Args:
            box_seq_len [int]: length of each box sequence
            box_real_sizes [dict]: a dict including items of {box_type: box_real_size}
            cases [int]: number of box sequences to generate
            bin_real_size: actual length and width of the bin
            grid_num: number of grid along length and width direction
            box_type_num_range: range of box types to use to generate box sequences
            permute [boolean]: if true, permute l, w, and h of each box
    """
    box_sets = []
    grid_size = np.ceil(bin_real_size / grid_num)
    print("grid size: {}".format(grid_size))

    # extract the box sizes from the dict into a list
    size_list = []
    for k, v in box_real_sizes.items():
        size_list.append(v)
    size_list = np.array(size_list).flatten()

    box_grid_sizes = {}
    for k, v in box_real_sizes.items():
        box_grid_sizes[k] = np.hstack((np.ceil((np.array(v)/grid_size)).astype(int),
                                      100*(np.ceil((np.array(v)/grid_size)[-1]) - v[-1]/grid_size))).tolist()
    print("box grid sizes:\n{}".format(box_grid_sizes))
    
    sel_box_grid_sizes = []
    box_grid_sizes = dict(list(box_grid_sizes.items())[box_type_num_range[0]:box_type_num_range[-1]])
    for k, v in box_grid_sizes.items():
        if permute:
            v = multiset_permutations(v)
            for box_grid_size in v:
                sel_box_grid_sizes.append(box_grid_size)
        else:
            sel_box_grid_sizes.append(v)
    print("length of sel_box_grid_sizes:\n{}".format(len(sel_box_grid_sizes)))
    print("sel_box_grid_sizes:\n{}".format(sel_box_grid_sizes))
    box_grid_sizes = sel_box_grid_sizes

    for i in range(cases):
        box_indices = np.random.randint(low=0, high=len(box_grid_sizes), size=box_seq_len)
        box_set = [[int(box_grid_sizes[box_index][0]), 
                    int(box_grid_sizes[box_index][1]), 
                    int(box_grid_sizes[box_index][2]), 
                    int(box_grid_sizes[box_index][3])] for box_index in box_indices]
        box_sets.append(box_set)
        if i % 100 == 0:
            print("\n>>> Box set {}, length {}:\n{}".format(i, len(box_set), box_set))

    return box_sets


if __name__ == "__main__":

    box_seq_len = 50
    box_real_sizes = BOX_FH_SIZES
    cases = 3000
    bin_real_size=1120
    grid_num=28
    box_type_num_range = [0,5]
    permute = False
    save = True

    box_sequences = vanilla_boxgen(box_seq_len=box_seq_len,
                                   box_real_sizes=box_real_sizes, 
                                   cases=cases, 
                                   bin_real_size=bin_real_size, 
                                   grid_num=grid_num,
                                   box_type_num_range=box_type_num_range,
                                   permute=permute,
                                   save=save
                                )
    # print("box_sequences:\n", box_sequences)

    if len(box_sequences) == cases:
        torch.save(box_sequences, 'real_boxgen.pt')