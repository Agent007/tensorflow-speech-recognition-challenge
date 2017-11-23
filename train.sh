#!/bin/bash -e

source activate
python ~/tensorflow/tensorflow/examples/speech_commands/train.py --data_url '' --data_dir data/train/audio --testing_percentage 0 --validation_percentage 1 --summaries_dir data/retrain_logs --train_dir data/speech_commands_train