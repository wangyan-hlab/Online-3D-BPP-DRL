# Test the heuristic BPH algorithm for ONLINE 3D-BPP
# ips: item packing sequence
# ems: empty maximal space
# kb: number of arrived items (# number of lookahead items)
# ke: number of EMSs to select from the EMSs list of each container
# p: place combination = (item, rotation, ems)
# ps: packing solution
# rotation: 0 - no rotation, 1 - rotate 90deg around item's z-axis
import torch
import argparse
import numpy as np
import copy


def item_fit_ems(item, rotation, ems):
    """
    Checking if an item can be totally inscribed by EMS with a given rotation
    """
    if rotation == 1:
        item = [item[1], item[0], item[2]]
    for d in range(3):
        if item[d] > ems[1][d] - ems[0][d]:
            return False
    return True


def item_is_blocked(box, rotation, selected_EMS, ps, container_size):
    """
    Checking if an item is blocked by others
    """
    if rotation == 1:
        box = [box[1], box[0], box[2]]
    ems =[selected_EMS[0], list(np.array(selected_EMS[0]) + np.array(box))]
    # print("Space of the box {}: {}".format(box, ems))
    x3, y3, z3 = ems[0]
    x4, y4, z4 = ems[1]
    upstretch_EMSs = [
        [[x4, y3, z3], [container_size[0], y4, z4]],    # x+ direction
        [[x3, y4, z3], [x4, container_size[1], z4]],    # y+ direction
        [[x3, y3, z4], [x4, y4, container_size[2]]]     # z+ direction
    ]
    # print("Upstretch EMSs:", upstretch_EMSs)
    if ps:
        for p in ps:
            p_occ_space = [p[2][0], list(np.array(p[2][0]) + np.array(p[0]))]
            overlaps, inscribes = [], []
            for upstretch_ems in upstretch_EMSs:
                overlaps.append(overlapped(upstretch_ems, p_occ_space))
                inscribes.append(inscribed(upstretch_ems, p_occ_space))
            # print("overlaps: {}, inscribes: {}".format(overlaps, inscribes))
            # print("Overlapped: {}\nInscribed: {}".format(any(overlaps), any(inscribes)))
            if any(overlaps) or any(inscribes):
                return True
        return False
    else:
        return False


def overlapped(EMS_1, EMS_2):
    """
        Checking if EMS_1 overlaps with EMS_2
    """
    corner1_ovl = all(ai > bi for ai, bi in zip(EMS_1[1], EMS_2[0]))
    corner2_ovl = all(ai < bi for ai, bi in zip(EMS_1[0], EMS_2[1]))
    if corner1_ovl and corner2_ovl:
        # print("{} is overlapped with {}".format(EMS_1, EMS_2))
        return True
    return False


def inscribed(EMS_1, EMS_2):
    """
        Checking if EMS_1 is inscribed by EMS_2
    """
    corner1_ins = all(ai >= bi for ai, bi in zip(EMS_1[0], EMS_2[0]))
    corner2_ins = all(ai <= bi for ai, bi in zip(EMS_1[1], EMS_2[1]))
    if corner1_ins and corner2_ins:
        return True
    return False


def update_ips(ips, placed_item_idx):
    """
        Updating the item packing sequence
    """
    ips.pop(placed_item_idx)
    return ips


def update_ems_list(box, rotation, ips, selected_EMS, existing_EMSs):
    """
        Updating the EMSs 
    """
    print(">>> Updating EMS list")
    # 1. compute maximal-space for box with selected EMS
    if rotation == 1:
        box = [box[1], box[0], box[2]]
    ems = [selected_EMS[0], list(np.array(selected_EMS[0]) + np.array(box))]

    # 2. Generate new EMSs resulting from the intersection of the box 
    all_candidate_EMSs = []
    existing_EMSs_copy = copy.deepcopy(existing_EMSs)
    for id, EMS in enumerate(existing_EMSs_copy):
        print("Checking relationship with {}th existing EMS {}".format(id, EMS))
        if overlapped(ems, EMS):
            # eliminate overlapped EMS
            existing_EMSs.remove(EMS)

            # three new EMSs in 3D
            x1, y1, z1 = EMS[0]
            x2, y2, z2 = EMS[1]
            x3, y3, z3 = ems[0]
            x4, y4, z4 = ems[1]
            new_EMSs = [
                [[x1, y1, z1], [x3, y2, z2]],
                [[x4, y1, z1], [x2, y2, z2]],
                [[x1, y1, z1], [x2, y3, z2]],
                [[x1, y4, z1], [x2, y2, z2]],
                [[x1, y1, z1], [x2, y2, z3]],
                [[x1, y1, z4], [x2, y2, z2]]
            ]
            for new_EMS in new_EMSs:
                all_candidate_EMSs.append(new_EMS)
        print(">>> All candidate EMSs:", all_candidate_EMSs)

        idx_to_del = []
        for i, new_EMS in enumerate(all_candidate_EMSs):
            # print("Checking candidate EMS {}".format(new_EMS))
            isValid = True
            temp = copy.deepcopy(all_candidate_EMSs)
            temp.pop(i)
            # 3. Eliminate new EMSs which are totally inscribed by other EMSs
            for other_EMS in temp:
                if inscribed(new_EMS, other_EMS):
                    # print("Candidate EMS {} is inscribed by {}\n".format(new_EMS, other_EMS))
                    isValid = False
            
            # 4. Remove _EMSs_ which are smaller than existing boxes to be placed
            if ips:
                new_box = np.array(new_EMS[1]) - np.array(new_EMS[0])
                ## (1) new EMS smaller than the volume of remaining boxes
                min_vol = min([x * y * z for x, y, z in ips])
                if np.prod(new_box) < min_vol:
                    isValid = False
                ## (2) new EMS having smaller dimension of the smallest dimension of remaining boxes
                x_values, y_values, z_values = zip(*ips)
                min_dim = min(min(x_values), min(y_values), min(z_values))
                if np.min(new_box) < min_dim:
                    isValid = False

            # add new EMS if valid
            if isValid:
                # print("Adding valid candidate EMS {}".format(new_EMS))
                pass
            else:
                idx_to_del.append(i)

    # 使用列表解析删除指定索引的元素
    new_all_candidate_EMSs = [element for index, element in enumerate(all_candidate_EMSs) if index not in idx_to_del]

    for new_EMS in new_all_candidate_EMSs:
        existing_EMSs.append(new_EMS)

    # Sort the list using the custom sorting function
    sorted_existing_EMSs = sorted(existing_EMSs, key=custom_sort)
        
    return sorted_existing_EMSs


def custom_sort(elem):
    x1, y1, z1 = elem[0]
    x2, y2, z2 = elem[1]
    
    # 先比较z值，值小的排在前面
    if z1 != z2:
        return z1 - z2
    # 如果z值相同，比较x值
    elif x1 != x2:
        return x1 - x2
    # 如果x值相同，比较y值
    else:
        return y1 - y2


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='pt file path')
    parser.add_argument('--index', type=int, default=0, help='chose traj index')
    parser.add_argument('--plot', action='store_true', help='plot the cubes')
    parser.add_argument('--cases', type=int, default=20, help='cases number')
    parser.add_argument('--preview', type=int, default=1, help='preview number')
    parser.add_argument('--ems_number', type=int, default=10, help='EMS number to consider')
    args = parser.parse_args()

    box_sequences = torch.load(args.file_path)
    # box_sequences = [[[3, 4, 5], [5, 6, 7], [2, 3, 4], [9, 10, 11]]]
    
    space_ratios = []
    packed_box_numbers = []
    for case_id, box_sequence in enumerate(box_sequences[1:args.cases]): #TODO: recovery to range [:args.cases]

        container_size = [25, 25, 25]
        ps  = []
        ips = box_sequence
        ems_list = [[[0,0,0], container_size]]

        while ips:
            print("-"*20)
            p = []
            kb = args.preview
            ke = args.ems_number
            placed_item_idx = []
            item_placed = 0
            
            j = 0
            while j < ke and j < len(ems_list):
                kb = min(kb, len(ips))
                for i in range(kb):
                    for rotation in [0, 1]:
                        print("***")
                        print("ips:", ips)
                        print("ems_list:", ems_list)
                        fit_ems = item_fit_ems(ips[i], rotation, ems_list[j])
                        is_blocked = item_is_blocked(ips[i], rotation, ems_list[j], ps, container_size)
                        print("fit_ems:", fit_ems)
                        print("is_blocked:", is_blocked)
                        if fit_ems and not is_blocked:
                            p.append([ips[i], rotation, ems_list[j]]) 
                            placed_item_idx.append(i)
                j += 1
                
            
            print("--- Feasible Packing Solution ---")
            print("p:", p)
            if p:
                # if len(p) >= 2:
                #     if p[0][-1] == p[1][-1]: # two feasible rotation in one EMS
                #         chosen_box = p[0][0]
                #         chosen_ems = p[0][-1]
                #         chosen_box_x, chosen_box_y, chosen_box_z = chosen_box
                #         chosen_ems_x, chosen_ems_y, chosen_ems_z = chosen_ems[1]
                #         margin_rot0_x = chosen_ems_x - chosen_box_x
                #         margin_rot0_y = chosen_ems_y - chosen_box_y
                #         margin_rot1_x = chosen_ems_x - chosen_box_y
                #         margin_rot1_y = chosen_ems_y - chosen_box_x
                #         min_value = min(margin_rot0_x, margin_rot0_y, margin_rot1_x, margin_rot1_y)
                #         if margin_rot0_x == min_value or margin_rot0_y == min_value:
                #             ps.append(p[0])
                #         else:
                #             ps.append(p[1])
                ps.append(p[0])
                ips = update_ips(ips, placed_item_idx[0]) # delete placed item from ips
                ems_list = update_ems_list(box=p[0][0], rotation=p[0][1], ips=ips, selected_EMS=p[0][2], existing_EMSs=ems_list)
                item_placed = 1
                print("An item placed!")
                print("Packing_solution: {}\n".format(ps))

            if item_placed == 0:
                print("No item placed!")
                break

        print("\n--- Packing solution ---\n", ps)

        total_vol = 0
        counter = 0
        sequence = []
        for p in ps:
            vol = np.prod(p[0])
            total_vol += vol
            counter += 1
            if p[1] == 1:
                p[0] = [p[0][1], p[0][0], p[0][2]]
            sequence.append(p[0]+p[2][0])
        space_ratio = total_vol/np.prod(container_size)
        print("space ratio:", space_ratio)
        print("packed box number:", counter)
        print("packing sequence:", sequence)
        space_ratios.append(space_ratio)
        packed_box_numbers.append(counter)

        if args.plot:
            color_list = ['red', 'orange','yellow', 'green', 'blue', 'purple']
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i, box in enumerate(sequence):
                # Define cube properties
                lx, ly, lz = box[3], box[4], box[5]
                x, y, z = box[0], box[1], box[2]
                color = color_list[i % len(color_list)]
                # Plot the cube
                ax.bar3d(lx, ly, lz, x, y, z, color=color)

            # Set axis labels
            ax.set_xlim3d([0, container_size[0]])
            ax.set_ylim3d([0, container_size[1]])
            ax.set_zlim3d([0, container_size[2]])
            ax.set_xticks(np.linspace(0, container_size[0], container_size[0]+1))
            ax.set_yticks(np.linspace(0, container_size[1], container_size[1]+1))
            ax.set_zticks(np.linspace(0, container_size[2], container_size[2]+1))
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_aspect('equal')

            # plt.show()
            plt.savefig(f'/home/ubuntu/Pictures/bpp_bph_real_{case_id+1}_{args.preview}.png')
    
    print("\n=== For {} Cases ===".format(args.cases))
    print("space_ratios:", space_ratios)
    print("mean space ratio:", np.mean(space_ratios))
    print("mean packed box number:", np.mean(packed_box_numbers))