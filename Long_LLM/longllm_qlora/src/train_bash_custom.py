import os, subprocess

TRAIN_ARGS = os.getenv('TRAIN_ARGS')
TRAIN_ARGS_ADD = os.getenv('TRAIN_ARGS_ADD')
run_com = os.getenv('RUN_CMD','torchrun') # accelerate or deepspeed or torchrun
NUM_GPUS = os.getenv('NUM_GPUS','1')

# Set environment variable
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print(os.listdir('/opt/ml/input/data/training'), flush=True)

if run_com=='deepspeed':
    print("running deepspeed", flush=True)
    subprocess.run(
        ["deepspeed", "--num_gpus", NUM_GPUS, "/app/src/train_bash.py"]+
        ["--deepspeed", "/app/examples/deepspeed/ds_z3_config.json"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
elif run_com=='accelerate':
    print("running accelerate", flush=True)
    subprocess.run(
        ["accelerate", "launch", "--config_file", "/app/examples/accelerate/single_config_8.yaml",
         "/app/src/train_bash.py"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
elif run_com=='torchrun':
    print("running torchrun", flush=True)
    subprocess.run(
        ["torchrun", "--nproc_per_node", NUM_GPUS, "-m", "main.train"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
else:
    subprocess.run(
        ["python", "src/train_bash.py"]+
        TRAIN_ARGS.split()+
        TRAIN_ARGS_ADD.split(),
        check=True
        )
