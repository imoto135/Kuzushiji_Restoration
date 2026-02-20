#!/bin/bash
# MPRNet 200k iterations training script

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nafnet2

cd /home/imoto/Kuzushiji_Restoration/models/nafnet

# Add current directory to PYTHONPATH for basicsr
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "=== Starting MPRNet Training (200k iterations) ==="
echo "GPU: 0"
echo "Config: options/Kuzushiji/mprnet_mask_charb_percep.yml"
echo "Log file: ../../mprnet_200k.log"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Run with nohup to prevent termination when terminal closes
CUDA_VISIBLE_DEVICES=0 nohup python basicsr/train.py -opt options/Kuzushiji/mprnet_mask_charb_percep.yml >> ../../mprnet_200k_training.log 2>&1 &

# Save the process ID
echo $! > ../../mprnet_training.pid
echo "Training started in background. PID: $(cat ../../mprnet_training.pid)"
echo "Monitor progress: tail -f ~/Kuzushiji_Restoration/mprnet_200k_training.log"
