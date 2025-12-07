#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import multiprocessing as mp
from pathlib import Path
import time
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading

def setup_logging(log_file, log_dir="./logs", name='render'):
    """设置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"parallel_render_{log_file}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重复输出
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def get_gpu_count():
    """获取可用GPU数量"""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            return len(result.stdout.strip().split('\n'))
        else:
            return 1
    except:
        return 1

def render_single_object(args_tuple):
    """渲染单个对象的函数，用于多进程"""
    tic = time.time()
    obj_path, output_dir, engine, only_northern_hemisphere, num_renders, gpu_id, time_out = args_tuple
    
    save_uid = os.path.basename(obj_path).split('.')[0]
    target_directory = os.path.join(output_dir, save_uid)
    os.makedirs(target_directory, exist_ok=True)
    
    # 检查是否已经渲染完成
    contents = os.listdir(target_directory)
    if len(contents) == 65:  # 假设完成渲染的文件数
        return f"SKIP: {save_uid} already rendered"
    
    cmd = [
        "blender-3.2.2-linux-x64/blender", "--background", "--python-exit-code", "1",
        "--python", "blender_script_gso.py",
        "--",
        "--object_path", obj_path,
        "--output_dir", target_directory,
        "--engine", engine,
        "--num_renders", str(num_renders)
    ]
    
    if only_northern_hemisphere:
        cmd.append("--only_northern_hemisphere")
    
    try:
        # 设置环境变量指定GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=time_out)
        
        
        if result.returncode == 0:
            return f"SUCCESS: {save_uid} rendered successfully. use time {time.time()-tic}"
        else:
            return f"ERROR: {save_uid} failed - {result.stderr[:200]}"
            
    except subprocess.TimeoutExpired:
        return f"TIMEOUT: {save_uid} rendering timeout"
    except Exception as e:
        return f"ERROR: {save_uid} exception - {str(e)}"

class ParallelBlenderRenderer:
    def __init__(self, max_workers=None, log_file=None, gpu_memory_limit=0.8):
        self.max_workers = max_workers or min(get_gpu_count() * 2, mp.cpu_count())
        self.gpu_count = get_gpu_count()
        self.gpu_memory_limit = gpu_memory_limit
        self.logger = log_file
        
    def distribute_jobs(self, obj_paths, output_dir, engine, only_northern_hemisphere, num_renders, time_out):
        """分配任务到不同GPU"""
        jobs = []
        for i, obj_path in enumerate(obj_paths):
            gpu_id = i % self.gpu_count
            jobs.append((obj_path, output_dir, engine, only_northern_hemisphere, num_renders, gpu_id, time_out))
        return jobs
    
    def render_parallel_process_pool(self, obj_paths, output_dir, engine, only_northern_hemisphere, num_renders, time_out):
        """使用ProcessPoolExecutor进行并行渲染"""
        jobs = self.distribute_jobs(obj_paths, output_dir, engine, only_northern_hemisphere, num_renders, time_out)
        
        completed = 0
        failed = 0
        timeout = 0
        skipped = 0
        
        self.logger.info(f"Starting parallel rendering with {self.max_workers} workers on {self.gpu_count} GPUs")
        self.logger.info(f"Total objects to render: {len(jobs)}")
        
        print(f'jobs:{len(jobs)}, workers:{self.max_workers}')
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_job = {executor.submit(render_single_object, job): job for job in jobs}
            
            for future in as_completed(future_to_job):
                result = future.result()
                
                if result.startswith("SUCCESS"):
                    completed += 1
                elif result.startswith("SKIP"):
                    skipped += 1
                elif result.startswith("TIMEOUT"):
                    timeout += 1
                else:
                    failed += 1
                
                self.logger.info(f"Progress: {completed + skipped + failed}/{len(jobs)} - {result}")
                
                # 打印进度
                if (completed + skipped + failed) % 10 == 0:
                    self.logger.info(f"Progress Summary - Completed: {completed}, Skipped: {skipped}, Failed: {failed}")
        
        self.logger.info(f"Rendering finished - Total: {len(jobs)}, Completed: {completed}, Skipped: {skipped}, Timeout: {timeout}, Failed: {failed}")
        return completed, skipped, timeout, failed

def render_parallel_subprocess(obj_paths, output_dir, engine, only_northern_hemisphere, num_renders, log_file='', ax_parallel=4):
    """使用subprocess并行运行多个Blender实例"""
    logger = setup_logging(log_file)
    
    def run_blender_instance(batch_paths, gpu_id):
        """运行单个Blender实例处理一批对象"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        for obj_path in batch_paths:
            save_uid = os.path.basename(obj_path).split('.')[0]
            target_directory = os.path.join(output_dir, save_uid)
            
            cmd = [
                "blender", "--background", "--python", "your_original_script.py",
                "--",
                "--object_path", obj_path,
                "--output_dir", target_directory,
                "--engine", engine,
                "--num_renders", str(num_renders)
            ]
            
            if only_northern_hemisphere:
                cmd.append("--only_northern_hemisphere")
            
            try:
                subprocess.run(cmd, env=env, check=True)
                logger.info(f"GPU {gpu_id}: Successfully rendered {save_uid}")
            except subprocess.CalledProcessError as e:
                logger.error(f"GPU {gpu_id}: Failed to render {save_uid} - {e}")
    
    # 将对象分成批次
    batch_size = len(obj_paths) // max_parallel + 1
    batches = [obj_paths[i:i + batch_size] for i in range(0, len(obj_paths), batch_size)]
    
    # 启动多个进程
    processes = []
    for i, batch in enumerate(batches):
        gpu_id = i % get_gpu_count()
        p = mp.Process(target=run_blender_instance, args=(batch, gpu_id))
        p.start()
        processes.append(p)
        logger.info(f"Started process {i} with {len(batch)} objects on GPU {gpu_id}")
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    logger.info("All rendering processes completed")

def main():
    parser = argparse.ArgumentParser(description="Parallel Blender Rendering")
    parser.add_argument("--object_path", type=str, required=True, help="Path to the object files directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--only_northern_hemisphere", action="store_true", default=True)
    parser.add_argument("--num_renders", type=int, default=32)
    parser.add_argument("--max_workers", type=int, help="Maximum number of parallel workers")
    parser.add_argument("--method", type=str, choices=["process_pool", "subprocess"], default="process_pool")
    parser.add_argument("--max_parallel", type=int, default=4, help="Maximum parallel Blender instances for subprocess method")
    parser.add_argument("--log_file", type=str, default='', help="log file path")
    parser.add_argument("--time_out", type=int, default=600, help="log file path")
    
    args = parser.parse_args()
    
    # 获取所有对象文件
    obj_paths = [os.path.join(args.object_path, f) for f in os.listdir(args.object_path) 
                 if f.endswith(('.obj', '.blend', '.fbx', '.glb', '.gltf'))]
    
    if not obj_paths:
        print("No object files found!")
        return
    
    print(f"Found {len(obj_paths)} object files to render")
    print(f"Available GPUs: {get_gpu_count()}")
    # output_dir = os.path.join(args.output_dir, args.object_path.split('/')[-1])
    logger = setup_logging(args.log_file)
    
    if args.method == "process_pool":
        # for retry in range(1):        
        renderer = ParallelBlenderRenderer(max_workers=args.max_workers, log_file=logger)
        completed, skipped, timeout, failed = renderer.render_parallel_process_pool(
            obj_paths, args.output_dir, args.engine, 
            args.only_northern_hemisphere, args.num_renders, args.time_out
        )
        logger.info(f'Completed: {completed}, Skipped: {skipped}, Timeout: {timeout}, Failed: {failed}')
            # if timeout == 0:
                # break
    else:
        render_parallel_subprocess(
            obj_paths, args.output_dir, args.engine,
            args.only_northern_hemisphere, args.num_renders, args.max_parallel
        )

if __name__ == "__main__":
    main()