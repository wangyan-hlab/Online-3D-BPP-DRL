# We'll use this script to generate box sequence data for training and test
import gym
from acktr.arguments import get_args
from unified_test import registration_envs
import torch


def check_boxgen(box_set, container_size):
    container_volume = container_size[0] * container_size[1] * container_size[2]
    box_volumes = 0
    for box in box_set:
        x, y, z = box
        box_volume = x * y * z
        box_volumes += box_volume
    ratio = box_volumes/container_volume

    return ratio == 1.0, ratio


if __name__ == "__main__":

    registration_envs()
    args = get_args()
    env = gym.make(args.env_name,
                box_set=args.box_size_set,
                container_size=args.container_size,
                test=False, 
                enable_rotation=args.enable_rotation,
                data_type=args.data_type)

    print()
    print("----- Box Generator Config -----")
    print('Env name: ', args.env_name)
    print('Item size range (xl,yl,zl,xh,yh,zh): ', args.item_size_range)
    print('Container size: ', args.container_size)
    print('Generator type: ', args.data_type)
    print('Case number: ', args.cases)

    if args.data_type == 'cut2':
        box_sets = []
        for i in range(args.cases):
            env.reset()
            box_set = env.box_creator.box_set[:-1]
            checkpass, ratio = check_boxgen(box_set, args.container_size)
            if checkpass:
                box_sets.append(box_set)
                if i % 1 == 0:
                    print("\n>>> Box set {}, length {}:\n{}".format(i, len(box_set), box_set))
            else:
                print("ratio:", ratio)
                raise ValueError("The space ratio of generated box is not 1.0")
        
        if len(box_sets) == args.cases:
            torch.save(box_sets, f'{args.data_type}_bound_{args.item_size_range[0]}_{args.item_size_range[3]}_bin_{args.container_size[0]}.pt')
    
    
        
            
