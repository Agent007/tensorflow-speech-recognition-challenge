#!/bin/bash -e

source activate
#TRAINING_SIZE=64073  # math.floor(64721 * .99)
BATCH_SIZE=512
EVAL_STEP_INTERVAL=128  # approximately TRAINING_SIZE / BATCH_SIZE
python ~/tensorflow/tensorflow/examples/speech_commands/train.py --data_url '' --data_dir data/train/audio --background_volume 0.5 --testing_percentage 0 --validation_percentage 1 --how_many_training_steps 15000,3000 --learning_rate 0.001,0.0001 --batch_size ${BATCH_SIZE} --summaries_dir data/retrain_logs --train_dir data/speech_commands_train

python ~/tensorflow/tensorflow/examples/speech_commands/train.py --data_url '' --data_dir data/train/audio --background_volume 0.5 --testing_percentage 0 --validation_percentage 1 --how_many_training_steps 21000 --learning_rate 0.0001 --batch_size ${BATCH_SIZE} --summaries_dir data/retrain_logs --train_dir data/speech_commands_train --start_checkpoint data/speech_commands_train/conv.ckpt-18000
