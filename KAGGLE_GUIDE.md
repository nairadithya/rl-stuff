# Kaggle Sweep Guide

This guide explains how to run the GRPO ablative sweep on Kaggle using the provided script and notebook.

## 1. Preparing the Kaggle Environment
1. Log into [Kaggle](https://www.kaggle.com/) and click **Create -> Notebook**.
2. **Environment Settings**: 
   - **Accelerator**: Go to "Session Options" (the three dots top right or right sidebar) -> **Accelerator**. Select a GPU instance like **GPU T4 x2**, **L4**, or **P100**.
   - **Internet Access**: Ensure **Internet on** is enabled so the notebook can download model weights and datasets from Hugging Face.
   - **Persistence**: (Optional but recommended) Set persistence to "Files only" to keep your downloaded weights in `/kaggle/working`.

## 2. Getting the Code into Kaggle
You have a few options to bring the code in:

**Option A: Upload the Notebook**
1. Download `kaggle_sweep.ipynb` from your local machine.
2. In your Kaggle Notebook, click **File -> Import Notebook** and select the `.ipynb` file.
3. You will also need to clone this repository into the Kaggle environment so that files like `run_prelim.py`, `train_grpo.py`, and `configs/` are available. Add a cell at the top of your notebook:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/grpo_small_lm.git /kaggle/working/grpo_small_lm
   %cd /kaggle/working/grpo_small_lm
   ```

**Option B: Clone the Repository directly**
1. Create a blank notebook.
2. Clone your repo and sync the Jupytext python script:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/grpo_small_lm.git /kaggle/working/grpo_small_lm
   %cd /kaggle/working/grpo_small_lm
   !pip install jupytext
   !jupytext --to notebook kaggle_sweep.py
   ```

## 3. Configuring the Sweep
In the Kaggle Notebook, locate the cell defining the `sweep_config` dictionary. You can easily tweak:
- `models`: Add or remove models (e.g., `["Qwen/Qwen2.5-0.5B-Instruct"]`).
- `loss_types`: Modify the ablation types (e.g., `["grpo", "dapo"]`).
- `max_steps`, `learning_rate`, `train_max_samples`, etc.
- `parallel`: Keep this as `False` if you are using a standard Kaggle T4 instance to avoid Out of Memory (OOM) errors. Set to `True` only if you have a multi-GPU instance that can comfortably fit multiple model training runs in VRAM simultaneously.

## 4. Running the Sweep
Simply hit **Run All** in your Kaggle notebook!

The notebook will:
1. Setup the necessary persistence directories in `/kaggle/working`.
2. Install dependencies (like `trl`, `peft`, `accelerate`).
3. Dump your configured `sweep_config` to `kaggle_sweep_config.yaml`.
4. Launch `run_prelim.py` to baseline, train, and evaluate the models.

All outputs, models, and leaderboards will be saved in `/kaggle/working/outputs/sweep/`. You can download these artifacts directly from the Kaggle right-hand panel under "Output" once the run is complete.
