# We'll use this script to test inference and visualize the result.

from matplotlib import pyplot as plt
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
from time import perf_counter
from functools import reduce
from math import gcd
import numpy as np
import copy
import gym
from unified_test import registration_envs


class Box(object):
    """
        Box class
    """
    def __init__(self, x, y, z, lx, ly, lz):
        self.x = x
        self.y = y
        self.z = z
        self.lx = lx
        self.ly = ly
        self.lz = lz


class BppAgent(object):
    """
        This tester class aims to simulate the inference process of the online-3dbpp agent
    """
    def __init__(self, args):
        """
            Bin state
        """
        self.args = args
        self.model = nnModel(args.load_dir + args.load_name, args)
        data_url = './dataset/' + args.data_name #TODO: use custom data
        self.env = gym.make(args.env_name,
                            box_set=args.box_size_set,
                            container_size=args.container_size,
                            test=True, 
                            data_name=data_url,
                            enable_rotation=args.enable_rotation,
                            data_type=args.data_type)
        print('Env name: ', args.env_name)
        print('Data url: ', data_url)
        print('Model url: ', args.load_dir + args.load_name)
        print('Case number: ', args.cases)
        print('Known item number: ', args.preview)

        self.bin_size = args.bin_size
        self.grid = np.zeros(shape=(self.bin_size[0], self.bin_size[1]), dtype=np.int32) # height map
        self.height = self.bin_size[2]  # height of boxes in the bin
        self.boxes = []     # boxes in the bin
        self._flag_sequences = []     # record rotation information
        self._sequences = []

    @property
    def sequences(self):
        return self._sequences
    
    @property
    def flag_sequences(self):
        return self._flag_sequences

    def run_sequence(self, nmodel, raw_env, preview_num, **kwargs):
        env = copy.deepcopy(raw_env)
        obs = env.cur_observation
        default_counter = 0
        box_counter = 0
        start = perf_counter()
        sequence = []
        box_list = []
        while True:
            box_list = env.box_creator.preview(preview_num)
            tree = ReorderTree(nmodel, box_list, env, times=100)
            act, val, default = tree.reorder_search()
            obs, _, done, info = env.step([act])

            if done:
                end = perf_counter()
                self.sequences.append(sequence)
                self.flag_sequences.append(env.space.flags)
                # print(len(self.sequences), len(self.flag_sequences), 
                #       "rotation_flags:", self.flag_sequences)
                assert len(self.sequences) == len(self.flag_sequences)
                self.grid = np.zeros(shape=(self.bin_size[0], self.bin_size[1]), dtype=np.int32)
                print('Time cost:', end-start)
                print('Time cost per item:', (end-start)/info['counter'])
                print('Ratio:', info['ratio'])
                print('Item number:', info['counter'])

                return info['ratio'], info['counter'], end-start, default_counter/box_counter
            else:
                next_box = env.space.boxes[-1]
                sequence.append([next_box.x, next_box.y, next_box.z,
                                next_box.lx, next_box.ly, next_box.lz])
            box_counter += 1
            default_counter += int(default)

    def test_sim(self):
        model = self.model
        env = self.env
        args = self.args

        times = args.cases
        ratios = []
        avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
        for i in range(times):
            if i % 10 == 0:
                print('case', i+1)
            env.reset()
            env.box_creator.preview(500)
            ratio, counter, time, depen_rate = self.run_sequence(model, env, args.preview, real=False)
            avg_ratio += ratio
            ratios.append(ratio)
            avg_counter += counter
            avg_time += time
            avg_drate += depen_rate

        print()
        print('All cases have been done!')
        print('----------------------------------------------')
        print('average space utilization: %.4f'%(avg_ratio/times))
        print('average put item number: %.4f'%(avg_counter/times))
        print('average sequence time: %.4f'%(avg_time/times))
        print('average time per item: %.4f'%(avg_time/avg_counter))
        print('----------------------------------------------')

    def test_real(self):
        model = self.model
        env = self.env
        env.reset()
        # env.box_creator.preview(500)
        ratio, counter, time, depen_rate = self.run_sequence(model, env, args.preview, real=True)

        print()
        print('case has been done!')
        print('----------------------------------------------')
        print('space utilization: %.4f'%(ratio))
        print('put item number: %.4f'%(counter))
        print('sequence time: %.4f'%(time))
        print('time per item: %.4f'%(time/counter))
        print('----------------------------------------------')


if __name__ == "__main__":

    from acktr.arguments import get_args
    import time

    registration_envs()
    args = get_args()
    tester = BppAgent(args=args)
    bin_size = tester.bin_size

    if not args.real:
        tester.test_sim()
    else:
        tester.test_real()

    sequences = tester.sequences

    # for case_id, sequence in enumerate(sequences):
    #     print(sequence)
    #     grid_size = 40
    #     sequence = np.multiply(sequence, grid_size)
    #     print(">>>", sequence)
    #     for box in sequence:
    #         box[3] += box[0]/2
    #         box[4] += box[1]/2
    #         box[5] += box[2]
    #     np.savetxt(f'fake_box_data_{case_id}.txt', sequence, fmt='%5.1f')

    # plot the cubes
    color_list = ['red','tomato','orange','yellow', 'green','cyan','blue','magenta','purple']

    for case_id, sequence in enumerate(sequences):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        volumes = 0
        counter = 0
        for i, box in enumerate(sequence):
            # Define cube properties
            lx, ly, lz = box[3], box[4], box[5]
            x, y, z = box[0], box[1], box[2]
            color = color_list[i % len(color_list)]
            # Plot the cube
            volume = x * y * z
            volumes += volume
            counter += 1
            ax.bar3d(lx, ly, lz, x, y, z, color=color)

        print("----- CASE {} -----".format(case_id))
        print("box_sequence:", sequence)
        print("flag_rotation:", tester.flag_sequences[case_id])
        print("space uti:", volumes/(bin_size[0]*bin_size[1]*bin_size[2]))
        print("loaded box number:", counter)

        # Set axis labels
        ax.set_xlim3d([0, bin_size[0]])
        ax.set_ylim3d([0, bin_size[1]])
        ax.set_zlim3d([0, bin_size[2]])
        ax.set_xticks(np.linspace(0, bin_size[0], bin_size[0]+1))
        ax.set_yticks(np.linspace(0, bin_size[1], bin_size[1]+1))
        ax.set_zticks(np.linspace(0, bin_size[2], bin_size[2]+1))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal')
        
        # plt.show()
        plt.savefig(f'/home/ubuntu/Pictures/real_{case_id}_{args.preview}.png')
        