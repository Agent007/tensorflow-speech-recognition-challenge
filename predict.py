# Ouputs predictions as CSV.
# See tensorflow/examples/speech_commands/label_wav.py

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import os
import pandas as pd


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


load_graph('frozen_graph.pb')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


model_name = 'kaggle'
labels_filename = model_name + '_labels.txt'
labels_filepath = 'data/speech_commands_train/' + labels_filename
labels = load_labels(labels_filepath)


def predict(wav_data, sess, labels, input_layer_name='wav_data:0', output_layer_name='labels_softmax:0', num_top_predictions=1):
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    return [(labels[node_id], predictions[node_id]) for node_id in top_k]  # list of (label, score)


test_path = 'data/test/audio'

rows = []
with tf.Session() as sess:
    for wav in os.listdir(test_path):
        with open(test_path + '/' + wav, 'rb') as wav_file:
            wav_data = wav_file.read()
            top_prediction = predict(wav_data, sess, labels)[0]
            label, score = top_prediction
            new_row = [wav, label, score]
            rows.append(new_row)


predictions = pd.DataFrame(rows, columns=['fname', 'label', 'score'])
predictions.to_csv('predictions.csv', index=False)
