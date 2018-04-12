import numpy as np
import unittest
import tensorflow as tf

from decomposable_attention_nli import DecomposableAttentionNLI as DaNli
TRAIN_FILE_PATH = "data/snli_1.0/snli_1.0_train.jsonl"
TEST_FILE_PATH = "data/snli_1.0/snli_1.0_test.jsonl"

# TRAIN_FILE_PATH = "data/snli_1.0/mock_training_file.jsonl"
# TEST_FILE_PATH = "data/snli_1.0/mock_testing_file.jsonl"

class DecomposableAttentionNLITest(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            self.nli = DaNli(learning_rate=0.05, batch_size=32)

    # def test_train(self):
    #     with self.sess.as_default():
    #         self.nli.train(TRAIN_FILE_PATH, epoch_number=20)
    #         self.nli.print_training_accuracy_graph()
    #         self.nli.eval(TEST_FILE_PATH, "./models/-20")

    # def test_test_accuracy(self):
    #     with self.sess.as_default():
    #         self.nli.print_testing_accuracy_graph(epoch_number=6, test_file_path=TEST_FILE_PATH)

    # def test_predict(self):
    #     with self.sess.as_default():
    #         print(self.nli.predict("An apple is good.", "The church is good.", "./models/-100"))
    #         print(self.nli.predict("An apple is good.", "The apple is yummy.", "./models/-100"))

    def test_predict(self):
        sentence1and2s = [("looking for a front end engineer who knows Java", "looking for a software engineer"),
            ("someone to watch Netflix with me", "someone to watch TV shows"),
            ("a person to walk my dog", "someone nice to walk dogs"),
            ("a Penn student to chat for coffee", "chat and get to know a penn student"),
            ("a designer to cofound my startup", "a software designer interested in entrepreneurship")]

        #expected_outcome: all entailments
        with self.sess.as_default():
            for (s1, s2) in sentence1and2s:
                print(self.nli.predict(s1, s2, "./models/-20"))

if __name__ == '__main__':
    unittest.main()