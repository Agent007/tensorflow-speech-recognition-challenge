#!/bin/bash -e

source activate root
export CUDA_VISIBLE_DEVICES=$1
#TRAINING_SIZE=64073  # math.floor(64721 * .99)
BATCH_SIZE=128
EVAL_STEP_INTERVAL=200  # approximately TRAINING_SIZE / BATCH_SIZE
python train.py --data_url '' --data_dir data/train/audio --background_volume 0.75 --silence_percentage 8.333333333 --unknown_percentage 8.333333333 --testing_percentage 0 --validation_percentage 1 --how_many_training_steps 3000,3000,3000 --eval_step_interval ${EVAL_STEP_INTERVAL} --learning_rate 0.001,0.001,0.0001 --batch_size ${BATCH_SIZE} --summaries_dir data/retrain_logs --train_dir data/speech_commands_train --model_architecture kaggle

#python train.py --data_url '' --data_dir data/train/audio --background_volume 0.75 --silence_percentage 8.333333333 --unknown_percentage 8.333333333 --testing_percentage 0 --validation_percentage 1 --how_many_training_steps 2400 --learning_rate 0.00003 --batch_size ${BATCH_SIZE} --summaries_dir data/retrain_logs --train_dir data/speech_commands_train --start_checkpoint data/speech_commands_train/kaggle.ckpt-2000 --model_architecture kaggle
