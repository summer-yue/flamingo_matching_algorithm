import numpy as np
import unittest

from decomposable_attention_nli import DecomposableAttentionNLI as DaNli
TEST_FILE_PATH = "data/snli_1.0/snli_1.0_test.jsonl"

class DecomposableAttentionNLITest(unittest.TestCase):
    def setUp(self):
        self.nli = DaNli()

    def test_build_graph(self):
        self.nli.build_graph()
        
    # def test_train(self):
    #     self.nli.train(TEST_FILE_PATH)

if __name__ == '__main__':
    unittest.main()