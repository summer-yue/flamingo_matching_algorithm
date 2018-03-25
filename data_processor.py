import numpy as np

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
        """
        postprocessed_data = self.preprocess_jsonl()
        embedded_data = [{
                            "sentence1": self.gloVe_embeddings(entry.sentence1),
                            "sentence2": self.gloVe_embeddings(entry.sentence2),
                            "gold_label": entry.gold_label
                         } for entry in postprocessed_data]
        return embedded_data

    def preprocess_jsonl(self, input_file_path):
        """ handles reading data from input jsonl file and writing preprocessed data into a separate file
        Preprocessed data is in the csv format: {sentence1, sentence2, gold_label} semi sorted

        1. Extracting setences and gold label from jsonl file, removing instances with label "-" for gold label from the dataset
        2. Prepending each sentence with the NULL token
        3. Adding padding to the sentences to the maximum length
        4. Semi-sorting the data by length < 20, length < 50 and others (to ensure each batch has similar length)
        """
        pass

    def gloVe_embeddings(self, sentence):
        """ Use 300 dimensional GloVe embeddings (Pennington et al., 2014) to represent words
        Normalize each vector to l2 norm of 1 and projected down to 200 dimensions
        Out of Vocab words are hashed to random embedding
        """
        pass