import os
import yaml
import modal

app = modal.App("grpo-rl-sweep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "accelerate>=1.2.1",
    "datasets>=3.2.0",
    "peft>=0.14.0",
    "torch>=2.8.0",
    "transformers>=5.8.0",
    "trl>=0.12.1",
    "pyyaml>=6.0.2",
    "math-verify>=0.9.0",
    "tqdm>=4.67.0",
    "vllm>=0.7.2",
    "wandb",
    "ninja",
    "packaging",
    "wheel",
).add_local_dir(".", remote_path="/workspace", ignore=["outputs", ".venv", "__pycache__", ".git"])

volume = modal.Volume.from_name("grpo-outputs-vol", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G", # 24GB VRAM is plenty for 0.5B/1B models with LoRA
    timeout=3600 * 24,
    volumes={"/workspace/outputs": volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def run_sweep_job(
    config_path: str,
    model: str,
    loss_type: str,
):
    """
    Executes a single sweep permutation via run_prelim.py.
    This runs both training and evaluation for this (model, loss_type) pair.
    """
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "/workspace/run_prelim.py", 
        "--config", f"/workspace/{config_path}",
        "--models", model,
        "--loss-types", loss_type,
        "--no-parallel",
        "--yes"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    # Modal typically uses /workspace as HOME or we should ensure we have HuggingFace tokens if needed
    
    try:
        subprocess.run(cmd, env=env, check=True, cwd="/workspace")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed for {model} with {loss_type}")
        raise e
    
    print(f"Completed {model} with {loss_type}")

@app.local_entrypoint()
def main(config_path: str = "configs/sweeps/pipeline_sweep_presweep.yaml"):
    """
    Parses the sweep config locally and dispatches a job for each
    (model, loss_type) combination to Modal to run concurrently.
    """
    with open(config_path, "r") as f:
        sweep_config = yaml.safe_load(f)
        
    models = sweep_config.get("models", [])
    loss_types = sweep_config.get("loss_types", [])
    if not loss_types:
        loss_types = ["grpo"] # fallback
        
    print(f"Dispatching {len(models) * len(loss_types)} training jobs to Modal...")
    
    args_list = []
    for model in models:
        for loss_type in loss_types:
            args_list.append((config_path, model, loss_type))
            
    # Starmap runs these concurrently on multiple Modal containers
    for res in run_sweep_job.starmap(args_list):
        pass
    
    print("All training jobs completed!")
