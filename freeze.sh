#!/bin/bash -e

source activate
python ~/tensorflow/tensorflow/examples/speech_commands/freeze.py --start_checkpoint=data/speech_commands_train/conv.ckpt-18000 --output_file=frozen_graph.pb
