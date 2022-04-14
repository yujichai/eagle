#!/bin/bash
#SBATCH -p gpu_test  # Partition to submit to
#SBATCH -t 0-08:00         # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -c 4              # Number of cores (-c)
#SBATCH --gres=gpu:1                # Number of cores (-c)
#SBATCH --mem=128000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o ./logs/job_log_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./logs/job_log_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
#SBATCH --mail-user=yuc927@g.harvard.edu

module load Anaconda3 CUDA cudnn
eval "$(conda shell.bash hook)"
conda activate eagle

PROJECT_DIR=/n/holylfs05/LABS/acc_lab/Users/yujichai/eagle
PREDICTOR=gcn
MODEL=nasbench201
DEVICE=desktop
CFG=configs/predictors/desktop_gcn_nasbench201.yaml
MEASUREMENT=results/nasbench201/latency/desktop/desktop-cpu-core-i7-7820x-fp32.pickle

echo "JobID: $SLURM_JOBID"
perl -e 'print "Job starting ...\n"'
cd $PROJECT_DIR
python3 -u -m eagle.predictors --predictor $PREDICTOR --model $MODEL --device $DEVICE --cfg $CFG --measurement $MEASUREMENT