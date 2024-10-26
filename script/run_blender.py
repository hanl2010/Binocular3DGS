import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
dataset_name = "Blender"
data_base_path='DATA_DIR'
out_base_path=f'output/{dataset_name}'
n_views = 8
resolution = 2

excluded_gpus = set()
mem_threshold = 0.2
base_dir = os.path.abspath(os.getcwd())


jobs = scenes

def train_block(gpu_id, scene):
       cmd = (
              f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}_{n_views}views '
              f'--n_views {n_views} --dataset_name {dataset_name} --resolution {resolution} --eval '
              f'--iterations 7000 --shift_cam_start 4000 '
              )
       print(cmd)
       os.system(cmd)

       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py '
              f'--model_path {out_base_path}/{scene}_{n_views}views '
              f'--n_views {n_views} --skip_train --resolution {resolution} --eval '
              f'--dataset_name {dataset_name} '
              )
       print(cmd)
       os.system(cmd)

       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py '
              f'--model_path {out_base_path}/{scene}_{n_views}views '
              )
       print(cmd)
       os.system(cmd)

       return True

def worker(gpu_id, block_id):
    print(f"Starting job on GPU {gpu_id} with block {block_id}\n")
    train_block(gpu_id, block_id)
    print(f"Finished job on GPU {gpu_id} with block {block_id}\n")
    # This worker function starts a job and returns when it's done.


def dispatch_jobs(jobs, executor):
       future_to_job = {}
       reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

       while jobs or future_to_job:
              # Get the list of available GPUs, not including those that are reserved.
              all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxLoad=0.9, maxMemory=mem_threshold))
              available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

              # Launch new jobs on available GPUs
              while available_gpus and jobs:
                     gpu = available_gpus.pop(0)
                     job = jobs.pop(0)
                     future = executor.submit(worker, gpu, job)  # Unpacking job as arguments to worker
                     future_to_job[future] = (gpu, job)
                     reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

              # Check for completed jobs and remove them from the list of running jobs.
              # Also, release the GPUs they were using.
              done_futures = [future for future in future_to_job if future.done()]
              for future in done_futures:
                     job = future_to_job.pop(future)  # Remove the job associated with the completed future
                     gpu = job[0]  # The GPU is the first element in each job tuple
                     reserved_gpus.discard(gpu)  # Release this GPU
                     print(f"Job {job} has finished., rellasing GPU {gpu}")
              # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
              # when there are no GPUs available.
              if len(jobs) > 0:
                     print("No GPU available at the moment. Retrying in 1 minutes.")
                     time.sleep(60)
              else:
                     time.sleep(10)

       print("All blocks have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)
