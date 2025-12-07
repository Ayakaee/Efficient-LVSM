import sys
sys.path.append('/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/objaverse-xl')
import objaverse
import os
import time
import argparse
    
parser = argparse.ArgumentParser(description='删除小于指定大小的一级子目录')
parser.add_argument('--st', type=int, required=True,
                    help='最小目录大小（字节），小于此值的子目录将被删除')
parser.add_argument('--end', type=int, required=True,
                    help='只显示将要删除的目录，不实际执行删除')

args = parser.parse_args()
# 获取所有对象的UID
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['OBJAVERSE_CACHE_DIR'] = '/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/dataset'
uids = objaverse.load_uids()
print(f"Total objects: {len(uids)}")
# objects = objaverse.load_objects(
#         uids=uids,
#         start_idx='000-000',
#         end_idx='000-004',
#         download_processes=8
#     )
while(True):
    try:
        for ids in range(args.st, args.end+1):
            print(f'download id 000-{ids}')
            objects = objaverse.load_objects(
                uids=uids,
                start_idx=f'000-{ids}',
                end_idx=f'000-{ids}',
                download_processes=16,
                path = '/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/dataset/hf-objaverse-v1'
            )
        break
    except Exception:
        time.sleep(30)
        print('exception occurs, rerun it')