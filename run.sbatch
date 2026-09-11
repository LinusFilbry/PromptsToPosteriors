#!/bin/bash
#SBATCH --partition=msc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=bay_inf_eval
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/$USER/.cache
export PIP_CACHE_DIR=$XDG_CACHE_HOME/pip
export TMPDIR=/scratch/$USER/tmp

export FLASHINFER_WORKSPACE_BASE=/scratch-ssd/$USER

export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_CACHE=/scratch-ssd/oatml/huggingface/hub
export HF_DATASETS_CACHE=/scratch-ssd/oatml/huggingface/datasets

mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" \
         "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=0

# Nuke the existing environment
# /scratch-ssd/oatml/run_locked.sh \
#  /scratch-ssd/oatml/miniconda3/bin/conda env remove -n BayInf_env -y

/scratch-ssd/oatml/run_locked.sh \
  /scratch-ssd/oatml/miniconda3/bin/conda env update -f environment.yml

source /scratch-ssd/oatml/miniconda3/bin/activate BayInf_env

pip install -r requirements.txt

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
else
    echo "Warning: HUGGINGFACE_TOKEN is not set. Gated models will fail to download."
fi

echo "START TIME: $(date)"

srun python main.py "$@"

echo "END TIME: $(date)"
