import numpy as np
import unittest

from data_processor import DataProcessor

class DataProcessorTest(unittest.TestCase):
    def setUp(self):
        self.dp = DataProcessor()

    # def test_pad_sentence_short(self):
    #     sentence = "he"
    #     max_len = 5
    #     self.assertEqual(self.dp.pad_sentence(sentence, max_len), "he   ")

    # def test_pad_sentence_same(self):
    #     sentence = "hello"
    #     max_len = 5
    #     self.assertEqual(self.dp.pad_sentence(sentence, max_len), sentence)

    # def test_pad_sentence_long(self):
    #     sentence = "helloW"
    #     max_len = 5
    #     self.assertEqual(self.dp.pad_sentence(sentence, max_len), "hello")

    # def test_preprocess_jsonl(self):
    #     print(len(self.dp.preprocess_jsonl(input_file_path="data/snli_1.0/snli_1.0_test.jsonl", sentence_max_len=100)))
    #     print(self.dp.preprocess_jsonl(input_file_path="data/snli_1.0/snli_1.0_test.jsonl", sentence_max_len=100)[-1])

    # def test_loade_glove(self):
    #     path = "models/glove.twitter.27B.200d.txt"
    #     self.dp.loadGloveModel(path)

    def test_gloVe_embeddings(self):
        sentence = "My mom has a cat meoww".lower()
        embeddings = self.dp.gloVe_embeddings(sentence)
        print(embeddings)
        print(len(embeddings))

if __name__ == '__main__':
    unittest.main()