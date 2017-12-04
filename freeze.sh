#!/bin/bash -e

source activate
python freeze.py --start_checkpoint=data/speech_commands_train/kaggle.ckpt-2000 --model_architecture kaggle --output_file=frozen_graph.pb
