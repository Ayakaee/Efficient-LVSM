import os
import json

datapath = '/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/dataset/hf-objaverse-v1/rendered/'
path_dir = {}
cnt = 0
# with open('data/uid_path_list.json', 'r') as f:
#     train = json.load(f).keys()
    
# for subpath in os.listdir(datapath):
#     sub_path_abs = os.path.join(datapath, subpath)
#     print(sub_path_abs)
#     if not os.path.exists(sub_path_abs):
#         continue
#     # print(subpath, os.listdir(sub_path_abs))
#     for uid in os.listdir(sub_path_abs):
#         # print(uid)
#         if len(os.listdir(os.path.join(sub_path_abs, uid))) > 50 and uid not in train:
#             path_dir[uid] = os.path.join(sub_path_abs, uid)
#             cnt +=1
#         if cnt % 10000 == 0:
#             print(cnt)
# print(len(path_dir.keys()))
# with open('data/test_path_list.json', 'w') as f:
#     json.dump(path_dir, f)
    
datapath = '/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/dataset/abo_render_f'
path_dir = {}
for uid in os.listdir(datapath):
    if len(os.listdir(os.path.join(datapath, uid))) == 65:
        path_dir[uid] = os.path.join(datapath, uid)
print(len(path_dir.keys()))
with open('data/abo_list.json', 'w') as f:
    json.dump(path_dir, f)