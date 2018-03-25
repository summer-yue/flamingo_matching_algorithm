import sys
import numpy as np
import tensorflow as tf

from decomposable_attention_nli import DecomposableAttentionNLI as daNLI

TRAINING_FILE_PATH = "data/snli_1.0/snli_1.0_train.jsonl"

if __name__ == '__main__':
    print("Training starts ...")
    processor = DataProcessor()
    model = daNLI()
    sess = tf.Session()

    training_data_dict = processor.get_embeddings(TRAINING_FILE_PATH)

    with sess.as_default():
        model.train(training_data_dict)