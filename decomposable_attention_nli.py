import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_processor import DataProcessor

class DecomposableAttentionNLI():
    """ Decomposable Attention Natural Language Inference Implementation in Tensorflow from Parikh et al.,
    """
    def __init__(self, learning_rate, batch_size):
        self.dp = DataProcessor()
        self.keep_prob = 0.8 # Indicating dropout constant = 0.2
        self.PROJECTED_DIMENSION_F = 300 # Number of neurons for the first fully connected layer
        self.PROJECTED_DIMENSION_G = 100 # Number of neurons for the second fully connected layer
        self.PROJECTED_DIMENSION_H = 3 # Number of gold labels = # Number of neurons for the third fully connected layer
        self.EMBEDDING_DIM = 200 # Dimension of used in Glove embeddings
        self.learning_rate = learning_rate # Learning rate used for the adagrad optimizer
        self.token_count = 20 # Number of tokens in each sentence, with padding
        self.batch_size = batch_size

        # Build and initialize tf graph
        self.sess = tf.get_default_session()
        self.build_graph(batch_size=batch_size)
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self, batch_size):
        """ Build the architecture used in natural language inference with attention
        """
        self.a = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence1") # batch_size x token_count x embedding_dim
        self.b = tf.placeholder(tf.float32, [None, None, self.EMBEDDING_DIM], name="sentence2") # batch_size x token_count x embedding_dim
        self.drop_a = tf.nn.dropout(self.a, self.keep_prob)
        self.drop_b = tf.nn.dropout(self.b, self.keep_prob)
        self.labels = tf.placeholder(tf.float32, [None, 3], name="gold_label") # 3 final potential categories

        # Attend
        # Calculate unnormalized attention weight e[i][j]
        self.unnormalized_attention_w = [[0] * self.token_count] * self.token_count # Dimension (token_count, token_count)
        self.f1a = tf.contrib.layers.fully_connected(self.drop_a, self.PROJECTED_DIMENSION_F) # Dimension (batch_size, token_count, PROJECTED_DIMENSION_F)
        self.f1a = tf.contrib.layers.fully_connected(self.f1a, self.PROJECTED_DIMENSION_F)
        self.f1b = tf.contrib.layers.fully_connected(self.drop_b, self.PROJECTED_DIMENSION_F) # Dimension (batch_Size, token_count, PROJECTED_DIMENSION_F)
        self.f1b = tf.contrib.layers.fully_connected(self.f1b, self.PROJECTED_DIMENSION_F)
        for i in range(self.token_count):
            for j in range(self.token_count):
                self.unnormalized_attention_w[i][j] = tf.reduce_sum(tf.multiply(self.f1a[:, i, :], self.f1b[:, j, :]))

        # Calculate self.Alpha and self.Beta
        self.Beta = [[[0.0] * self.EMBEDDING_DIM] * batch_size] * self.token_count
        self.Alpha = [[[0.0] * self.EMBEDDING_DIM] * batch_size] * self.token_count
        for i in range(self.token_count):
            for j in range(self.token_count):
                sum_exp_e_ik = 0.0
                for k in range(self.token_count):
                    sum_exp_e_ik = tf.add(sum_exp_e_ik, tf.exp(self.unnormalized_attention_w[i][k]))
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
        self.v1 = [0.0] * self.token_count
        self.v2 = [0.0] * self.token_count
        for i in range(self.token_count):
            self.v1[i] = tf.contrib.layers.fully_connected(
                tf.concat([self.a[:, i, :], self.Beta[i]], axis=1), 
                self.PROJECTED_DIMENSION_G
            )
            self.v1[i] = tf.contrib.layers.fully_connected(
                self.v1[i],
                self.PROJECTED_DIMENSION_G
            )

        for j in range(self.token_count):
            self.v2[j] = tf.contrib.layers.fully_connected(
                tf.concat([self.b[:, j, :], self.Alpha[j]], axis=1),
                self.PROJECTED_DIMENSION_G
            )
            self.v2[j] = tf.contrib.layers.fully_connected(
                self.v2[j],
                self.PROJECTED_DIMENSION_G
            )

        # Aggregate
        self.v1_aggregate = tf.reduce_sum(self.v1, axis=0, name="self.v1_aggregate")
        self.v2_aggregate = tf.reduce_sum(self.v2, axis=0, name="self.v2_aggregate")
        self.concat_vs = tf.concat([self.v1_aggregate, self.v2_aggregate], 1, name="concat_vs")
        self.h_logits = tf.contrib.layers.fully_connected(self.concat_vs, self.PROJECTED_DIMENSION_H, activation_fn=None)
        self.h_output = tf.nn.softmax(self.h_logits, name="h_output")
        self.predicted_gold_labels = tf.argmax(self.h_output, axis=1, name="predicted_gold_labels")        

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE) as scope:
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.h_logits, labels=self.labels),
                name="loss"
            )
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.variable_scope("eval", reuse=tf.AUTO_REUSE) as scope:
            correct_prediction = tf.equal(tf.argmax(self.labels, 1), self.predicted_gold_labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name="accuracy")

    def train(self, train_file_path, epoch_number):
        """ Train the attention NLI model 
        Args:
            train_file_path: jsonl file path to training data
            epoch_number: number of epochs of training
        Notes:
            trained models are saved in models/ every 50 epochs
        """
        saver = tf.train.Saver(max_to_keep=500)
        batch_acc_list = []
        batch_loss_list = []

        self.accuracy_records_by_epoch = []
        for i in range(epoch_number):
            batch_num = 0
            for batch_data in self.dp.get_batched_data(input_file_path=train_file_path, batch_size=self.batch_size):
                batch_num += 1
                batch_feed_dict = {
                    self.a: batch_data["sentence1"],
                    self.b: batch_data["sentence2"],
                    self.labels: batch_data["gold_label"],
                }
               
                if batch_num % 100 == 0:
                    print("batch", str(batch_num))
                
                _, acc, loss = self.sess.run([self.train_op, self.accuracy, self.loss], feed_dict=batch_feed_dict)
                batch_acc_list.append(acc)
                batch_loss_list.append(loss)
            self.accuracy_records_by_epoch.append(sum(batch_acc_list)/len(batch_acc_list))
            print("finishing epoch " + str(i) + ", training accuracy, loss")
            print(str(sum(batch_acc_list)/len(batch_acc_list)) + "," + str(sum(batch_loss_list)/len(batch_loss_list)))

            if i % 100 == 0 and i > 0:
                save_path = saver.save(self.sess, './models/', global_step=i)
                print("Model saved in file: %s" % save_path)

    def eval(self, test_file_path, model_path):
        """ Evaluate model's performance on test data
        Args:
            test_file_path: path to the jsonl file containing test datamodel_path
            model_path: path to the model to be restored
        Returns:
            float number indicating test accuracy
        """
        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(self.sess, model_path)
        print("Model restored from " + str(model_path))

        accuracies = []
        for test_data in self.dp.get_batched_data(input_file_path=test_file_path, batch_size=self.batch_size):
            test_feed = {
                self.a: test_data["sentence1"],
                self.b: test_data["sentence2"],
                self.labels: test_data["gold_label"],
            }
            accuracy, predictions = self.sess.run([self.accuracy, self.predicted_gold_labels], feed_dict=test_feed)
            print("predictions for the batch")
            print(predictions)
            print("Actual gold labels")
            print(test_data["gold_label"])
            accuracies.append(accuracy)
        test_acc = sum(accuracies)/len(accuracies)
        print(test_acc)

        return test_acc

    def predict(self, sentence1, sentence2, model_path):
        embeddings1_list = ["\0"] * self.batch_size
        embeddings2_list = ["\0"] * self.batch_size
        embeddings1_list[0] = self.dp.gloVe_embeddings(sentence1, self.token_count)
        embeddings2_list[0] = self.dp.gloVe_embeddings(sentence2, self.token_count)
        for i in range(self.batch_size):
            if i == 0:
                embeddings1_list[0] = self.dp.gloVe_embeddings(sentence1, self.token_count)
                embeddings2_list[0] = self.dp.gloVe_embeddings(sentence2, self.token_count)
            else:
                embeddings1_list[i] = self.dp.gloVe_embeddings("\0", self.token_count)
                embeddings2_list[i] = self.dp.gloVe_embeddings("\0", self.token_count)
        returned_label = self.predict_by_embeddings(np.array(embeddings1_list), np.array(embeddings2_list), model_path)
        if returned_label == 0:
            return "entailment"
        elif returned_label == 1:
            return "neutral"
        else:
            return "contradiction"

    def predict_by_embeddings(self, embeddings1_list, embeddings2_list, model_path):
        """Args:
            embeddings1_list: np array of sentence1's embeddings
            embeddings2_list: np array of sentence2's embeddings 
            model_path: path to the model to be restored
        """
        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(self.sess, model_path)
        feeds = {
            self.a: embeddings1_list,
            self.b: embeddings2_list,
        }
        return self.sess.run(self.predicted_gold_labels, feed_dict=feeds)

    def print_training_accuracy_graph(self):
        """ Draw a line graph on training accuracy by epoch number
        Data taken from self.accuracy_records_by_epoch recorded throughout the training process.
        """
        epoch_nums = [i+1 for i in range(len(self.accuracy_records_by_epoch))]
        plt.plot(epoch_nums, self.accuracy_records_by_epoch)
        plt.title('Training Accuracy on NLI task')
        plt.legend(['Training Accuracy'], loc='upper right')
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.show()
