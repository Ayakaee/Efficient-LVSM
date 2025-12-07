import numpy as np
import os

os.chdir(os.path.dirname(__file__))
data = np.load('/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/objaverse-xl/scripts/rendering/data_debug_20_l/92f4bc3ee830481c9018ab61889b6286/006.npy')
print(data)