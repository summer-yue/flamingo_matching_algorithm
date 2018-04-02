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
    #     print(len(self.dp.preprocess_jsonl(input_file_path="data/snli_1.0/snli_1.0_test.jsonl", max_token_num=100)))
    #     print(self.dp.preprocess_jsonl(input_file_path="data/snli_1.0/snli_1.0_test.jsonl", max_token_num=100)[-1])

    # def test_load_glove(self):
    #     path = "models/glove.twitter.27B.200d.txt"
    #     self.dp.loadGloveModel(path)

    def test_gloVe_embeddings(self):
        sentence = "An apple is good"
        embeddings = self.dp.gloVe_embeddings(sentence, 10)
        print("embedding for null token is ", embeddings)

    # def test_get_batched_data(self):
    #     batch_num = 0
    #     for batch_data in self.dp.get_batched_data(input_file_path="data/snli_1.0/snli_1.0_test.jsonl", batch_size=1000):
    #         batch_num += 1
    #         print("batch_num is ", batch_num) 
    #         print(batch_data["sentence1"].shape)
    #         print(batch_data["sentence2"].shape)
    #         print(batch_data["batch_max_word_count"])
    #         print(batch_data["gold_label"].shape)

    #     print("batch num is ", batch_num)

if __name__ == '__main__':
    unittest.main()