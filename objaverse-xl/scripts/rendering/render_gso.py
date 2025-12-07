import os
import subprocess
from tqdm import tqdm

# 1. 设置你的路径
gso_root_dir = "/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/dataset/gso"
output_render_dir = "/inspire/hdd/global_user/chenxinyan-240108120066/yihang/LVSM/dataset/gso_render6"
rendering_script_path = "blender_script_gso.py" # 假设和这个脚本在同一目录

# 获取所有 GSO 对象的文件夹名称
object_folders = [f for f in os.listdir(gso_root_dir) if os.path.isdir(os.path.join(gso_root_dir, f))]

# 2. 遍历每个对象并进行渲染
for folder_name in tqdm(object_folders, desc="Rendering GSO objects"):
    object_dir = os.path.join(gso_root_dir, folder_name)
    obj_path = os.path.join(object_dir, "meshes", "model.obj")
    
    # 检查 model.obj 是否存在
    if not os.path.exists(obj_path):
        print(f"Skipping {folder_name}: model.obj not found.")
        continue
        
    # 定义输出路径，以对象文件夹名作为唯一标识
    final_output_path = os.path.join(output_render_dir, folder_name)
    
    # 如果已经渲染过，可以跳过
    if os.path.exists(final_output_path) and len(os.listdir(final_output_path)) == 29:
        print(f"Skipping {folder_name}: Already rendered.")
        continue

    # 3. 构建并执行 Blender 渲染命令
    command = [
        'xvfb-run', '-a', 'blender-3.2.2-linux-x64/blender',
        '--background',
        '--python', rendering_script_path,
        '--',
        '--object_path', obj_path,
        '--output_dir', final_output_path,
        '--num_renders', '14',  # 你想要的渲染视角数量
        '--engine', 'BLENDER_EEVEE', # CYCLES 对真实感渲染效果更好，EEVEE 速度更快
        '--only_northern_hemisphere' # <-- 关键！
    ]
    
    print(f"\nRendering {folder_name}...")
    # 使用 subprocess.run 来执行命令
    subprocess.run(command)

print("All GSO objects have been processed.")