import numpy as np
import unittest
import tensorflow as tf

from decomposable_attention_nli import DecomposableAttentionNLI as DaNli
TEST_FILE_PATH = "data/snli_1.0/snli_1.0_test.jsonl"

class DecomposableAttentionNLITest(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            self.nli = DaNli()

    # def test_build_graph(self):
    #     self.nli.build_graph()

    def test_train(self):
        with self.sess.as_default():
            self.nli.train(TEST_FILE_PATH, batch_size=1000)

if __name__ == '__main__':
    unittest.main()