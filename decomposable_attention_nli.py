import numpy as np
import tensorflow as tf

from data_processor import DataProcessor

class DecomposableAttentionNLI():
    """ Decomposable Attention Natural Language Inference Implementation in Tensorflow from Parikh et al.,
    """
    def __init__(self):
        #self.dp = DataProcessor()
        self.PROJECTED_DIMENSION = 150
        self.EMBEDDING_DIM = 200

    def build_graph(self):
        self.la = tf.placeholder(tf.int32, [None], name="sentence1_max_size")
        self.lb = tf.placeholder(tf.int32, [None], name="sentence2_max_size")
        self.a = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence1")
        self.b = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence2")
        self.labels = tf.placeholder(tf.float32, [None], name="gold_label")

        self.f1a = tf.contrib.layers.fully_connected(self.a, self.PROJECTED_DIMENSION)
        self.f1b = tf.contrib.layers.fully_connected(self.b, self.PROJECTED_DIMENSION)
  
    def train(self, train_file_path, batch_size):

        for batch_data in self.dp.get_batched_data(input_file_path=train_file_path, batch_size=batch_size):
            # This is not used currently
            batch_feed_dict = {
                self.la: batch_data["la"],
                self.lb: batch_data["lb"],
                self.a: batch_data["sentence1"],
                self.b: batch_data["sentence2"],
                self.labels: batch_data["gold_label"]
            }

            # Attend
            # Calculate unnormalized attention weight e[i][j]
            unnormalized_attention_w = np.zeros((batch_data["la"], batch_data["lb"]))
            Fa = np.zeros(batch_data["la"])
            Fb = np.zeros(batch_data["lb"])
            for i in range(batch_data["la"]):
                # Feed batch_data["sentence1"][i] into the neural net and get f1a
                Fa[i] = sess.run(self.f1a, feed_dict={self.a: batch_data["sentence1"]})
            for j in range(batch_data["lb"]):
                # Feed batch_data["sentence2"][j] into the neural net and get f1b
                Fb[j] = sess.run(self.f1b, feed_dict={self.b: batch_data["sentence2"]})
            for i in range(batch_data["la"]):
                for j in range(batch_data["lb"]):
                    unnormalized_attention_w[i][j] = np.dot(Fa[i], Fb[j])

            # Calculate alpha and beta
            Beta = np.zeros(batch_data["la"])
            Alpha = np.zeros(batch_data["lb"])
            for i in range(batch_data["la"]):
                for j in range(batch_data["lb"]):
                    sum_exp_e_ik = 0
                    for k in range(batch_data["lb"]):
                        sum_exp_e_ik += np.exp(unnormalized_attention_w[i][k])
                    Beta[i] += np.exp(unnormalized_attention_w[i][j] / sum_exp_e_ik * batch_data["sentence2"][j])
            for j in range(batch_data["lb"]):
                for i in range(batch_data["la"]):
                    sum_exp_e_kj= 0
                    for k in range(batch_data["la"]):
                        sum_exp_e_kj += np.exp(unnormalized_attention_w[k][j])
                    Alpha[j] += np.exp(unnormalized_attention_w[i][j] / sum_exp_e_kj * batch_data["sentence1"][i])

            # Compare

            # Aggregate

            # Preprocess feed dict from train file
            # np array of dictionaries containing embeddings of 2 sentences (word num x 200 np array)
        
    def eval(self, test_file_path):
        pass

    def predict(self):
        pass
