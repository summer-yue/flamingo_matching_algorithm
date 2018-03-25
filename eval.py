import sys
import numpy as np
import tensorflow as tf

from decomposable_attention_nli import DecomposableAttentionNLI as daNLI

TEST_FILE_PATH = "data/snli_1.0/snli_1.0_test.jsonl"

if __name__ == '__main__':
    print("Evaluation for DecomposableAttentionNLI starts ...")
    processor = DataProcessor()
    model = daNLI()
    sess = tf.Session()

    testing_data_dict = processor.get_embeddings(TEST_FILE_PATH)

    with sess.as_default():
        model.eval(testing_data_dict)