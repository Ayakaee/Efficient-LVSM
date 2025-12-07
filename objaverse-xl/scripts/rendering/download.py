import sys
sys.path.append('/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/objaverse-xl')
import objaverse
import os
import time
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
        for ids in range(50,60):
            print(f'download id 000-0{ids}')
            objects = objaverse.load_objects(
                uids=uids,
                start_idx=f'000-0{ids}',
                end_idx=f'000-0{ids}',
                download_processes=16
            )
        break
    except Exception:
        time.sleep(30)
        print('exception occurs, rerun it')