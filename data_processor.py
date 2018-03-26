import numpy as np
import json_lines

class DataProcessor():
    """ The DataProcessor class handles reading data from file and 
        writing preprocessed data into a separate file
        embed data into vectors
    """
    def __init__(self):
        # Used for hashing gold labels into numeric format for training, omitting "-"
        self.GOLD_LABELS = {'entailment': 1, 'neutral': 0, 'contradiction': -1}

    def get_embeddings(self, input_file_path):
        """ Preprocess the data in a file into a dictionary of sentences embeddings and
        the corresponding gold label in a numeric format
        Args:
            input_file_path: path to file where the input jsonl is
        Returns:
            embedded_data: a np array of dictionaries containing embeddings of 2 sentences and their corresponding gold label in a numeric form
        """
        postprocessed_data = self.preprocess_jsonl()
        embedded_data = [{
                            "sentence1": self.gloVe_embeddings(entry.sentence1),
                            "sentence2": self.gloVe_embeddings(entry.sentence2),
                            "gold_label": entry.gold_label
                         } for entry in postprocessed_data]
        return np.array(embedded_data)

    def preprocess_jsonl(self, input_file_path, sentence_max_len):
        """ handles reading data from input jsonl file and writing preprocessed data into a separate file
        Preprocessed data is in the json format: np array of {"sentence1":..., "sentence2":..., "gold_label": 1} semi sorted

        1. Extracting setences and gold label from jsonl file, removing instances with label "-" for gold label from the dataset
        2. Prepending each sentence with the NULL token
        3. Adding padding to the sentences to the maximum length
        4. Semi-sorting the data by length < 20, length < 50 and others (to ensure each batch has similar length)
        Args:
            input_file_path: path to file where the input jsonl is
            sentence_max_len: the length of the sentences we are adding padding to
        Returns:
            np array of new dictionayrs  {"sentence1":..., "sentence2":..., "gold_label": 1} semi sorted
        """
        short_data_list = []
        medium_data_list = []
        long_data_list = []

        with open(input_file_path, 'rb') as input_file: # opening file in binary(rb) mode    
            for item in json_lines.reader(input_file):
                if item["gold_label"] != "-": # Removing unlabeled data
                    new_item = {}
                    data_len = max(len(item["sentence1"]), len(item["sentence2"]))
                    new_item["sentence1"] = self.pad_sentence('\0' + item["sentence1"], sentence_max_len)  # Prepending sentence1 with the NULL token
                    new_item["sentence2"] = self.pad_sentence('\0' + item["sentence2"], sentence_max_len) # Prepending sentence2 with the NULL token
                    new_item["gold_label"] = self.GOLD_LABELS[item["gold_label"]] # Converting gold label to numeric

                    if data_len < 20:
                        short_data_list.append(new_item)
                    elif data_len < 50:
                        medium_data_list.append(new_item)
                    else:
                        long_data_list.append(new_item)
            return np.concatenate((short_data_list, medium_data_list, long_data_list), axis=0)

    def gloVe_embeddings(self, sentence):
        """ Use 300 dimensional GloVe embeddings (Pennington et al., 2014) to represent words
        Normalize each vector to l2 norm of 1 and projected down to 200 dimensions
        Out of Vocab words are hashed to random embedding
        """
        pass

    def pad_sentence(self, sentence, sentence_max_len):
        """ Padding the input sentence to sentence_max_len with " ", which is masked out during training,
        take the first sentence_max_len characters if the number of chars in the sentences already exceeds sentence_max_len
        Args:
            sentence: string of sentence to be processed
            sentence_max_len: int indicating the largest number of chars a sentence can contain
        Returns:
            The padded sentence of sentence_max_len characters
        """
        if len(sentence) >= sentence_max_len:
            return sentence[0: sentence_max_len]

        return sentence.ljust(sentence_max_len)









