import os

def write_subdirs_to_file(root_dir, output_file):
    """
    将指定目录下的所有子目录的绝对路径写入到txt文件中
    
    :param root_dir: 要遍历的根目录
    :param output_file: 输出文件路径
    """
    paths = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # 遍历当前目录下的所有子目录
            for dirname in dirnames:
                # 获取子目录的绝对路径
                full_path = os.path.abspath(os.path.join(dirpath, dirname))
                paths.append(full_path)
        paths.sort()
        for path in paths:
            # 写入文件
            f.write(path + '\n')

if __name__ == '__main__':
    # 设置要遍历的目录
    target_directory = '/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/dataset/hf-objaverse-v1/glbs'
    # 设置输出文件路径
    output_filename = 'object_list.txt'
    
    # 检查目录是否存在
    if os.path.isdir(target_directory):
        write_subdirs_to_file(target_directory, output_filename)
        print(f"所有子目录路径已写入到 {output_filename}")
    else:
        print("错误: 指定的目录不存在")