import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', help='pt file path')
parser.add_argument('--index', type=int, default=0, help='chose traj index')
args = parser.parse_args()

pt_file = torch.load(args.file_path)
# pt_file = np.asarray(pt_file).tolist()
for i in range(10):
    sequence = np.asarray(pt_file)[i]*40
    print(sequence)
    np.savetxt(f'fake_box_data_{i}.txt', sequence, fmt='%10.1f')
# print(type(pt_file), len(pt_file), len(pt_file[args.index]))
# print(pt_file[args.index])

# for idx, seq in enumerate(pt_file):
#     # print(seq)
#     if idx // 100 == 0:
#         print(seq)
#         print(len(seq))

