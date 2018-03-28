import numpy as np
import tensorflow as tf

from data_processor import DataProcessor

class DecomposableAttentionNLI():
    """ Decomposable Attention Natural Language Inference Implementation in Tensorflow from Parikh et al.,
    """
    def __init__(self, learning_rate=0.05):
        self.dp = DataProcessor()
        self.PROJECTED_DIMENSION_F = 150
        self.PROJECTED_DIMENSION_G = 100
        self.PROJECTED_DIMENSION_H = 3 # Number of gold labels
        self.EMBEDDING_DIM = 200
        self.sess = tf.get_default_session()
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
        self.learning_rate = learning_rate

    def build_graph(self):
        self.la = tf.placeholder(tf.int32, [None], name="sentence1_max_size")
        self.lb = tf.placeholder(tf.int32, [None], name="sentence2_max_size")
        self.a = tf.placeholder(tf.float32, [None, self.EMBEDDING_DIM], name="sentence1")
        self.b = tf.placeholder(tf.float32, [None, self.EMBEDDING_DIM], name="sentence2")
        self.labels = tf.placeholder(tf.float32, [None], name="gold_label")

        self.f1a = tf.contrib.layers.fully_connected(self.a, self.PROJECTED_DIMENSION_F)
        self.f1b = tf.contrib.layers.fully_connected(self.b, self.PROJECTED_DIMENSION_F)
        print(self.f1a)

        # Used for the compare step, input the concatenation of ai and the subphrase in b soft-aligned with ai
        # Same for bj and its corresponding soft aligned subphrase in a
        self.concat_a_beta_b_alpha = tf.placeholder(tf.float32, [None, 2*self.EMBEDDING_DIM], name="contatenation_a_beta_or_b_alpha")
        self.g = tf.contrib.layers.fully_connected(self.concat_a_beta_b_alpha, self.PROJECTED_DIMENSION_G)

        # Used for the aggregate step
        self.concat_vs = tf.placeholder(tf.float32, [None, 2*self.PROJECTED_DIMENSION_G], name="contatenation_of_v1_and_v2")
        self.h_logits = tf.contrib.layers.fully_connected(self.concat_vs, self.PROJECTED_DIMENSION_H, activation_fn=None)
        self.h_output = tf.nn.softmax(self.h_logits)

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE) as scope:
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.h_logits, labels=self.label)
            )
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
  
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
            Fa = np.zeros((batch_data["la"], batch_size, self.PROJECTED_DIMENSION_F))
            Fb = np.zeros((batch_data["lb"], batch_size, self.PROJECTED_DIMENSION_F))
            for i in range(batch_data["la"]):
                # Feed batch_data["sentence1"][i] into the neural net and get f1a
                Fa[i] = self.sess.run(self.f1a, feed_dict={self.a: batch_data["sentence1"][:][i][:]})
            for j in range(batch_data["lb"]):
                # Feed batch_data["sentence2"][j] into the neural net and get f1b
                Fb[j] = self.sess.run(self.f1b, feed_dict={self.b: batch_data["sentence2"][:][j][:]})
            for i in range(batch_data["la"]):
                for j in range(batch_data["lb"]):
                    unnormalized_attention_w[i][j] = np.dot(Fa[i], Fb[j])

            # Calculate alpha and beta
            Beta = np.zeros((batch_data["la"], batch_size, self.EMBEDDING_DIM))
            Alpha = np.zeros((batch_data["lb"], batch_size, self.EMBEDDING_DIM))
            for i in range(batch_data["la"]):
                for j in range(batch_data["lb"]):
                    sum_exp_e_ik = 0
                    for k in range(batch_data["lb"]):
                        sum_exp_e_ik += np.exp(unnormalized_attention_w[i][k])
                    Beta[i] += np.exp(unnormalized_attention_w[i][j]) / sum_exp_e_ik * batch_data["sentence2"][j]
            for j in range(batch_data["lb"]):
                for i in range(batch_data["la"]):
                    sum_exp_e_kj= 0
                    for k in range(batch_data["la"]):
                        sum_exp_e_kj += np.exp(unnormalized_attention_w[k][j])
                    Alpha[j] += np.exp(unnormalized_attention_w[i][j]) / sum_exp_e_kj * batch_data["sentence1"][i]

            print(Alpha.shape)
            print(Beta.shape)

            # Compare
            v1 = np.zeros((batch_data["la"], batch_size, 2*self.EMBEDDING_DIM))
            v2 = np.zeros((batch_data["lb"], batch_size, 2*self.EMBEDDING_DIM))
            for i in range(batch_data["la"]):
                v1[i] = self.sess.run(
                    self.g,
                    feed_dict={
                        self.concat_a_beta_b_alpha: np.concatenate((batch_data["sentence1"][i], Beta[i]), axis=1)
                    }
                )
            for j in range(batch_data["lb"]):
                v2[j] = self.sess.run(
                    self.g,
                    feed_dict={
                        self.concat_a_beta_b_alpha: np.concatenate((batch_data["sentence2"][j], Alpha[j]), axis=1)
                    }
                )

            # Aggregate
            v1_aggregate = np.sum(v1, axis=0)
            v2_aggregate = np.sum(v2, axis=0)
            softmax_output = self.sess.run(
                self.h_output,
                feed_dict={
                    self.concat_vs: np.concatenate((v1_aggregate, v2_aggregate), axis=1)
                }
            )
            predicted_gold_labels = np.argmax(softmax_output, axis=0)

            
    def eval(self, test_file_path):
        pass

    def predict(self):
        pass
