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
        self.a = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence1") # token_count x batch x embedding_dim
        self.b = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence2") # batch x lb x embedding_dim
        self.labels = tf.placeholder(tf.float32, [None, 3], name="gold_label")

        # Attend
        # Calculate unnormalized attention weight e[i][j]
        # (15, 15)
        self.unnormalized_attention_w = [[0] * self.token_count] * self.token_count # Dimension (la, lb)
        #(1092, 15, 150)
        self.f1a = tf.contrib.layers.fully_connected(self.a, self.PROJECTED_DIMENSION_F) # Dimension (batch_size, la, PROJECTED_DIMENSION_F)
        self.f1b = tf.contrib.layers.fully_connected(self.b, self.PROJECTED_DIMENSION_F) # Dimension (batch_Size, lb, PROJECTED_DIMENSION_F)
        
        #(1092, 150)
        print("self.f1a[:, i, :] shape is ", self.f1a[:, 0, :])
        print("self.f1b[:, j, :] shape is ", self.f1a[:, 0, :])
        for i in range(self.token_count):
            for j in range(self.token_count):
                self.unnormalized_attention_w[i][j] = tf.reduce_sum(tf.multiply(self.f1a[:, i, :], self.f1b[:, j, :]))

        # Calculate self.Alpha and self.Beta
        self.Beta = [[[0.0] * self.EMBEDDING_DIM] * batch_size] * self.token_count
        self.Alpha = [[[0.0] * self.EMBEDDING_DIM] * batch_size] * self.token_count
        # (15, 1092, 200)
        print("self.b[:][j][:])")
        print(self.b[:][0][:])
        for i in range(self.token_count):
            for j in range(self.token_count):
                sum_exp_e_ik = 0.0
                for k in range(self.token_count):
                    sum_exp_e_ik = tf.add(sum_exp_e_ik, tf.exp(self.unnormalized_attention_w[i][k]))
                # print("check dim")
                # print(tf.multiply(tf.divide(tf.exp(self.unnormalized_attention_w[i][j]), sum_exp_e_ik), self.b[:][j][:]))
                self.Beta[i] = tf.add(
                    self.Beta[i], 
                    tf.multiply(tf.divide(tf.exp(self.unnormalized_attention_w[i][j]), sum_exp_e_ik), self.b[:, j, :])
                )
        for j in range(self.token_count):
            for i in range(self.token_count):
                sum_exp_e_kj = 0.0
                for k in range(self.token_count):
                    sum_exp_e_kj = tf.add(sum_exp_e_kj, tf.exp(self.unnormalized_attention_w[k][j]))
                self.Alpha[j] = tf.add(self.Alpha[j],
                    tf.multiply(tf.divide(tf.exp(self.unnormalized_attention_w[i][j]), sum_exp_e_kj), self.a[:, i, :])
                )

        # Compare
        self.v1 = [0.0] * self.token_count # (15, ?, 100)
        self.v2 = [0.0] * self.token_count # (15, ?, 100)
        for i in range(self.token_count):
            self.v1[i] = tf.contrib.layers.fully_connected(
                tf.concat([self.a[:, i, :], self.Beta[i]], axis=1), 
                self.PROJECTED_DIMENSION_G
            )

        for j in range(self.token_count):
            self.v2[j] = tf.contrib.layers.fully_connected(
                tf.concat([self.b[:, j, :], self.Alpha[j]], axis=1),
                self.PROJECTED_DIMENSION_G
            )

        # print(len(self.v1))
        # print(len(self.v2))
        # print(self.v1[0])
        # print(self.v2[0])
        # Aggregate
        self.v1_aggregate = tf.reduce_sum(self.v1, axis=0, name="self.v1_aggregate")
        self.v2_aggregate = tf.reduce_sum(self.v2, axis=0, name="self.v2_aggregate")
        print(self.v1_aggregate )
        print(self.v2_aggregate )
        # Used for the aggregate step
        self.concat_vs = tf.concat([self.v1_aggregate, self.v2_aggregate], 1, name="concat_vs")
        print(self.concat_vs) # (?, 200)
        self.h_logits = tf.contrib.layers.fully_connected(self.concat_vs, self.PROJECTED_DIMENSION_H, activation_fn=None)
        print(self.h_logits ) # (?, 3)
        self.h_output = tf.nn.softmax(self.h_logits, name="h_output")

        predicted_gold_labels = tf.argmax(self.h_output, axis=0, name="predicted_gold_labels")        

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE) as scope:
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.h_logits, labels=self.labels),
                name="loss"
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
            # print("a1")
            # Expecting (batch_size, 15, 200)
            # print(batch_data["sentence1"].shape)
            # Expecting (batch_size, 200)
            # print(batch_data["sentence1"][:, 0, :].shape)

            # print("f1a") # (batch_size, 15, 150)
            # print(len(self.sess.run(self.f1a, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.f1a, feed_dict=batch_feed_dict)[0]))
            # print(len(self.sess.run(self.f1a, feed_dict=batch_feed_dict)[0][0]))
            # print("f1b") # (batch_size, 15, 150)
            # print(len(self.sess.run(self.f1b, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.f1b, feed_dict=batch_feed_dict)[0]))
            # print(len(self.sess.run(self.f1b, feed_dict=batch_feed_dict)[0][0]))
            # print("w") # (15, 15)
            # self.sess.run(self.unnormalized_attention_w[0][0], feed_dict=batch_feed_dict)
            # print("self.Beta") #(15, batch_size, 200)
            # print(self.sess.run(self.Beta[0], feed_dict=batch_feed_dict))
            # print("self.Alpha") #(15, batch_size, 200)
            # print(self.sess.run(self.Alpha[0], feed_dict=batch_feed_dict))

            # print("self.v1") # (15, batch_size, 100)
            # print(len(self.sess.run(self.v1, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.v1, feed_dict=batch_feed_dict)[0]))
            # print(len(self.sess.run(self.v1, feed_dict=batch_feed_dict)[0][0]))
            # print("self.v2")
            # print(len(self.sess.run(self.v2[0], feed_dict=batch_feed_dict)))

            # print("self.v1_aggregate ") # (batch_size, 100)
            # print(len(self.sess.run(self.v1_aggregate, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.v1_aggregate, feed_dict=batch_feed_dict)[0]))
            # print("self.v2_aggregate ")  # (batch_size, 100)
            # print(len(self.sess.run(self.v2_aggregate, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.v2_aggregate, feed_dict=batch_feed_dict)[0]))
            # print("self.concat_vs") # (batch_size, 200)
            # print(len(self.sess.run(self.concat_vs, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.concat_vs, feed_dict=batch_feed_dict)[0]))
            # print("self.h_logits")
            # print(len(self.sess.run(self.h_logits, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.h_logits, feed_dict=batch_feed_dict)[0]))
            # print("self.h_output")
            # print(len(self.sess.run(self.h_output, feed_dict=batch_feed_dict)))
            # print(len(self.sess.run(self.h_output, feed_dict=batch_feed_dict)[0]))
            self.sess.run(self.train_op, feed_dict=batch_feed_dict)
            print("next batch")

    def eval(self, test_file_path):
        pass

    def predict(self):
        pass
