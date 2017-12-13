#!/bin/bash -e

MODEL="kaggle"
CHECKPOINT=$1
export CUDA_VISIBLE_DEVICES=$2
source activate root
python freeze.py --start_checkpoint=data/speech_commands_train/${MODEL}.ckpt-${CHECKPOINT} --model_architecture ${MODEL} --output_file=frozen_graph.pb
