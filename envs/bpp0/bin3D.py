from .space import Space
import numpy as np
import copy
import gym
from .cutCreator import CuttingBoxCreator
from .mdCreator  import MDlayerBoxCreator
from .binCreator import RandomBoxCreator, LoadBoxCreator, BoxCreator

class PackingGame(gym.Env):
    def __init__(self, box_creator=None, container_size = (20, 20, 20),
                 box_set = None, data_name = None, test = False,
                 data_type = 'cut1', enable_rotation=False, **kwargs):
        self.box_creator = box_creator
        self.bin_size = container_size
        self.area = int(self.bin_size[0] * self.bin_size[1])
        if 'h_comp' in kwargs:
            self.h_comp = kwargs['h_comp']
        else:
            self.h_comp = False
        self.space = Space(*self.bin_size, h_comp=self.h_comp)
        self.can_rotate = enable_rotation

        if not test and box_creator is None:
            assert box_set is not None
            if data_type == 'rs':
                print('using random data')
                self.box_creator = RandomBoxCreator(box_set)
            elif data_type == 'cut1':
                low = list(box_set[0])
                up = list(box_set[-1])
                low.extend(up)
                print(low)
                self.box_creator = CuttingBoxCreator(container_size, low, self.can_rotate)
            elif data_type == 'cut2':
                print('using md data')
                self.box_creator = MDlayerBoxCreator(container_size, [box_set[0][0], box_set[-1][0]])
            assert isinstance(self.box_creator, BoxCreator)

        if test:
            self.box_creator = LoadBoxCreator(data_name)

        self.act_len = self.area * (1+self.can_rotate)
        self.obs_len = self.area * (1+3)
        self.action_space = gym.spaces.Discrete(self.act_len)
        self.observation_space = gym.spaces.Box(low=0.0, high=self.space.height, shape=(self.obs_len,))
        

    def get_box_ratio(self):
        coming_box = self.next_box
        return (coming_box[0] * coming_box[1] * coming_box[2]) / (self.space.plain_size[0] * self.space.plain_size[1] * self.space.plain_size[2])


    def get_box_plain(self):
        x_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[0]
        y_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[1]
        z_plain = np.ones(self.space.plain_size[:2], dtype=np.int32) * self.next_box[2]
        return (x_plain, y_plain, z_plain)

    def reset(self):
        self.box_creator.reset()
        self.space = Space(*self.bin_size, h_comp=self.h_comp)
        self.box_creator.generate_box_size()
        return self.cur_observation

    @property
    def cur_observation(self):
        hmap = self.space.plain[:, :, 0]
        # mask = self.get_possible_position()
        size = self.get_box_plain()
        return np.reshape(np.stack((hmap,  *size)), newshape=(-1,))

    @property
    def next_box(self):
        return self.box_creator.preview(1)[0]

    def get_possible_position(self, plain=None):
        x = self.next_box[0]
        y = self.next_box[1]
        z = self.next_box[2]

        if plain is None:
            plain = self.space.plain

        width = self.space.plain_size[0]
        length = self.space.plain_size[1]

        action_mask = np.zeros(shape=(width, length), dtype=np.int32)
        
        for i in range(width-x+1):
            for j in range(length-y+1):
                if self.space.check_box(plain, x, y, i, j, z) >= 0:
                    action_mask[i, j] = 1

        if action_mask.sum() == 0:
            action_mask[:, :] = 1
        
        return action_mask

    def step(self, action):
        if isinstance(action, np.ndarray) or isinstance(action, list):
            idx = action[0]
        else:
            idx = action
        flag = False
        # check whether rotate the box
        if idx > self.area:
            assert self.can_rotate
            idx = idx - self.area
            flag = True
        succeeded = self.space.drop_box(self.next_box, idx, flag)

        if not succeeded:
            reward = 0.0
            done = True
            info = {'counter':len(self.space.boxes), 'ratio':self.space.get_ratio(), 'mask':np.ones(shape=self.act_len)}
            return self.cur_observation, reward, done, info

        box_ratio = self.get_box_ratio()
        # TODO: [testin] reward shaping for palletize far and low space first
        lx, ly = self.space.idx_to_position(idx)
        far_distance_reward = np.exp(-1.0 * np.linalg.norm([lx, ly])/np.linalg.norm(self.bin_size[:2]))
        lz = self.space.plain[lx, ly, 0]
        low_height_reward = np.exp(-1.0 * lz/self.bin_size[-1])

        self.box_creator.drop_box() # remove current box from the list
        self.box_creator.generate_box_size() # add a new box to the list

        plain = self.space.plain

        reward = box_ratio * 10 + far_distance_reward * 2.0 + low_height_reward * 0.0
        done = False
        info = dict()
        info['counter'] = len(self.space.boxes)
        info['ratio'] = self.space.get_ratio()
        # info['mask'] = self.get_possible_position().reshape((-1,))
        return self.cur_observation, reward, done, info

