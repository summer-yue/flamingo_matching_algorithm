import numpy as np
import tensorflow as tf

from data_processor import DataProcessor

class DecomposableAttentionNLI():
    """ Decomposable Attention Natural Language Inference Implementation in Tensorflow from Parikh et al.,
    """
    def __init__(self, learning_rate, batch_size):
        self.dp = DataProcessor()
        self.PROJECTED_DIMENSION_F = 150
        self.PROJECTED_DIMENSION_G = 100
        self.PROJECTED_DIMENSION_H = 3 # Number of gold labels
        self.EMBEDDING_DIM = 200
        self.learning_rate = learning_rate
        self.token_count = 15
        self.sess = tf.get_default_session()
        self.batch_size = batch_size
        self.build_graph(batch_size=batch_size)
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self, batch_size):
        self.a = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence1") # batch x la x embedding_dim
        self.b = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence2") # batch x lb x embedding_dim
        self.labels = tf.placeholder(tf.int32, [None], name="gold_label")

        # Attend
        # Calculate unnormalized attention weight e[i][j]
        unnormalized_attention_w = [[0] * self.token_count] * self.token_count # Dimension (la, lb)
        self.f1a = tf.contrib.layers.fully_connected(self.a, self.PROJECTED_DIMENSION_F) # Dimension (batch_size, la, PROJECTED_DIMENSION_F)
        self.f1b = tf.contrib.layers.fully_connected(self.b, self.PROJECTED_DIMENSION_F) # Dimension (batch_Size, lb, PROJECTED_DIMENSION_F)
        
        for i in range(self.token_count):
            for j in range(self.token_count):
                unnormalized_attention_w[i][j] = tf.reduce_sum(tf.multiply(self.f1a[:, i, :], self.f1b[:, j, :]))

        # Calculate alpha and beta
        Beta = [[[0.0] * self.EMBEDDING_DIM] * batch_size] * self.token_count
        Alpha = [[[0.0] * self.EMBEDDING_DIM] * batch_size] * self.token_count
        for i in range(self.token_count):
            for j in range(self.token_count):
                sum_exp_e_ik = 0.0
                for k in range(self.token_count):
                    sum_exp_e_ik = tf.add(sum_exp_e_ik, tf.exp(unnormalized_attention_w[i][k]))
                Beta[i] = tf.add(
                    Beta[i], 
                    tf.exp(tf.multiply(tf.divide(unnormalized_attention_w[i][j], sum_exp_e_ik), self.b[:][j][:]))
                )
        for j in range(self.token_count):
            for i in range(self.token_count):
                sum_exp_e_kj = 0.0
                for k in range(self.token_count):
                    sum_exp_e_kj = tf.add(sum_exp_e_kj, tf.exp(unnormalized_attention_w[k][j]))
                Alpha[j] = tf.add(Alpha[j],
                    tf.exp(
                        tf.multiply(tf.divide(unnormalized_attention_w[i][j], sum_exp_e_kj), self.a[:][i][:])
                    )
                )

        print(Alpha[0][0].shape)

        # Compare
        v1 = [[[0] * 2*self.EMBEDDING_DIM] * batch_size] * self.token_count
        v2 = [[[0] * 2*self.EMBEDDING_DIM] * batch_size] * self.token_count
        for i in range(self.token_count):
            v1[i] = tf.contrib.layers.fully_connected(
                tf.concat([self.a[:][i][:], Beta[i]], axis=1), 
                self.PROJECTED_DIMENSION_G
            )

        for j in range(self.token_count):
            v2[j] = tf.contrib.layers.fully_connected(
                tf.concat([self.b[:][j][:], Alpha[j]], axis=1),
                self.PROJECTED_DIMENSION_G
            )

        # Aggregate
        v1_aggregate = tf.reduce_sum(v1, axis=0)
        v2_aggregate = tf.reduce_sum(v2, axis=0)
        # Used for the aggregate step
        self.concat_vs = tf.concat([v1_aggregate, v2_aggregate], 0)
        self.h_logits = tf.contrib.layers.fully_connected(self.concat_vs, self.PROJECTED_DIMENSION_H, activation_fn=None)
        self.h_output = tf.nn.softmax(self.h_logits)

        predicted_gold_labels = tf.argmax(self.h_output, axis=0)        

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE) as scope:
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.h_logits, labels=self.labels)
            )
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
  
    def train(self, train_file_path):

        for batch_data in self.dp.get_batched_data(input_file_path=train_file_path, batch_size=self.batch_size):
            batch_feed_dict = {
                self.a: batch_data["sentence1"],
                self.b: batch_data["sentence2"],
                self.labels: batch_data["gold_label"],
            }
            self.sess.run(self.train_op, feed_dict=batch_feed_dict)
            print("next batch")

    def eval(self, test_file_path):
        pass

    def predict(self):
        pass
