import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time

scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
dataset_name = "LLFF"
data_base_path='DATA_DIR'
out_base_path=f'output/{dataset_name}'
n_views = 3
resolution = 2


excluded_gpus = set()
mem_threshold = 0.2
base_dir = os.path.abspath(os.getcwd())


jobs = scenes

def train_block(gpu_id, scene):
       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python triangulate.py --data_path {data_base_path}/{scene} '
              f'--output_path {base_dir}/keypoints_to_3d/{dataset_name} '
              f'--resolution {resolution} --dataset_name {dataset_name} ')
       print(cmd)
       os.chdir(os.path.join(base_dir, "submodules", "dense_matcher"))
       os.system(cmd)
       os.chdir(base_dir)

       model_path = f"{out_base_path}/{scene}_{n_views}views"

       cmd = (
              f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {model_path} '
              f'--n_views {n_views} --dataset_name {dataset_name} --resolution {resolution} --eval '
              )
       print(cmd)
       os.system(cmd)

       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py '
              f'--model_path {model_path} '
              f'--n_views {n_views} --skip_train --resolution {resolution} --eval '
              f'--dataset_name {dataset_name} '
              )
       print(cmd)
       os.system(cmd)

       cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py '
              f'--model_path {model_path} '
              )
       print(cmd)
       os.system(cmd)

       return True

def worker(gpu_id, scene):
    print(f"Starting job on GPU {gpu_id} with scene {scene}\n")
    train_block(gpu_id, scene)
    print(f"Finished job on GPU {gpu_id} with scene {scene}\n")


def dispatch_jobs(jobs, executor):
       future_to_job = {}
       reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

       while jobs or future_to_job:
              # Get the list of available GPUs, not including those that are reserved.
              all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=mem_threshold))
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
