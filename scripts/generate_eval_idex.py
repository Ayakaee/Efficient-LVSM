import json
import random

def view_selector(frames, num_views=5, min_frame_dist=25, max_frame_dist=192):
    if len(frames) < num_views:
        return None
    min_frame_dist = max(min_frame_dist, num_views)
    max_frame_dist = min(len(frames) - 1, max_frame_dist)
    if max_frame_dist <= min_frame_dist:
        return None
    frame_dist = random.randint(min_frame_dist, max_frame_dist)
    if len(frames) <= frame_dist:
        return None
    start_frame = random.randint(0, len(frames) - frame_dist - 1)
    end_frame = start_frame + frame_dist
    sampled_frames = random.sample(range(start_frame + 1, end_frame), num_views-2)
    image_indices = [start_frame, end_frame] + sampled_frames
    return image_indices

list_path = 'data/test/full_list.txt'
output_json = 'data/view_idx_list-6-3.json'
input_num_views = 6
output_num_views = 3
view_idx_list = {}

with open(list_path, 'r') as f:
    all_scene_paths = f.read().splitlines()
all_scene_paths = [path for path in all_scene_paths if path.strip()]
cnt = 0

for scene_path in all_scene_paths:
    print(cnt, scene_path)
    cnt += 1
    with open(scene_path, 'r') as f:
        data_json = json.load(f)
    frames = data_json["frames"]
    scene_name = data_json["scene_name"]
    indices = view_selector(frames, num_views=output_num_views + input_num_views)
    if indices is None:
        view_idx_list[scene_name] = None
    else:
        context = sorted(indices[:input_num_views])
        target = sorted(indices[input_num_views:])
        view_idx_list[scene_name] = {
            "context": context,
            "target": target
        }

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(view_idx_list, f, indent=4, ensure_ascii=True)
