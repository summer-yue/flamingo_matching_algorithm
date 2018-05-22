import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_processor import DataProcessor

class DecomposableAttentionNLI():
    """ Decomposable Attention Natural Language Inference Implementation in Tensorflow from Parikh et al.,
    """
    def __init__(self, learning_rate):
        self.keep_prob = 0.8 # Indicating dropout constant = 0.2
        self.PROJECTED_DIMENSION_F = 300 # Number of neurons for the fully connected layers in the Attend step
        self.PROJECTED_DIMENSION_G = 100 # Number of neurons for the fully connected layers in the Compare step
        self.PROJECTED_DIMENSION_H = 3 # Label dimension
        self.EMBEDDING_DIM = 200 # Dimension used in Glove embeddings
        self.learning_rate = learning_rate # Learning rate used for the adagrad optimizer
        self.token_count = 20 # Number of tokens in each sentence, with padding

        # Build and initialize tf graph
        self.sess = tf.get_default_session()
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        """ Build the architecture used in natural language inference with attention
        """
        self.a = tf.placeholder(tf.float32, [None, self.EMBEDDING_DIM], name="sentence1")  # ? x embedding_dim
        self.b = tf.placeholder(tf.float32, [None, self.EMBEDDING_DIM], name="sentence2")  # ? x embedding_dim
        self.drop_a = tf.nn.dropout(self.a, self.keep_prob)
        self.drop_b = tf.nn.dropout(self.b, self.keep_prob)
        self.labels = tf.placeholder(tf.int64, (3, ), name="gold_label") # Number of final potential categories

        # Attend
        with tf.variable_scope("Attend", reuse=tf.AUTO_REUSE) as scope:
            self.fa = tf.contrib.layers.fully_connected(self.drop_a, self.PROJECTED_DIMENSION_F)
            self.fa = tf.contrib.layers.fully_connected(self.fa, self.PROJECTED_DIMENSION_F)  # Dimension (?, PROJECTED_DIMENSION_F)
            self.fb = tf.contrib.layers.fully_connected(self.drop_b, self.PROJECTED_DIMENSION_F)
            self.fb = tf.contrib.layers.fully_connected(self.fb, self.PROJECTED_DIMENSION_F)  # Dimension (?, PROJECTED_DIMENSION_F)

            # Calculate unnormalized attention weight e[i][j],
            # Indicating the connections between the ith word in a and the jth word in b 
            self.attention_weights = tf.matmul(self.fa, self.fb, transpose_b=True, name='attention_weights')
            attention_soft1 = tf.nn.softmax(self.attention_weights, name='attention_soft1')
            attention_soft2 = tf.nn.softmax(tf.transpose(self.attention_weights), name='attention_soft2')

            # Beta_i indicates the phrases in b softly aligned with the ith phrase in a
            self.Beta = tf.matmul(attention_soft1, self.b)  # Dimension (?, 200)
            # Alpha_j indicates the phrases in a softly aligned with the jth phrase in b
            self.Alpha = tf.matmul(attention_soft2, self.a)  # Dimension (?, 200)

        # Compare
        with tf.variable_scope("Compare", reuse=tf.AUTO_REUSE) as scope:
            a_beta = tf.concat([self.a, self.Beta], 1)  # Dimension (?, 400)
            b_alpha = tf.concat([self.b, self.Alpha], 1)  # Dimension (?, 400)

            v1i = tf.layers.dense(inputs=a_beta, units=self.PROJECTED_DIMENSION_G, name='G1')
            v1i = tf.layers.dense(inputs=v1i, units=self.PROJECTED_DIMENSION_G, name='G2')
            v2j = tf.layers.dense(inputs=b_alpha, units=self.PROJECTED_DIMENSION_G, name='G1')
            v2j = tf.layers.dense(inputs=v2j, units=self.PROJECTED_DIMENSION_G, name='G2')

        # Aggregate
        with tf.variable_scope("Aggregate", reuse=tf.AUTO_REUSE) as scope:
            self.v1_aggregate = tf.reduce_sum(v1i, axis=0, name="self.v1_aggregate")  # Dimension (PROJECTED_DIMENSION_G, )
            self.v2_aggregate = tf.reduce_sum(v2j, axis=0, name="self.v2_aggregate")  # Dimension (PROJECTED_DIMENSION_G, )
            self.concat_vs = tf.concat([self.v1_aggregate, self.v2_aggregate], 0, name="concat_vs")  # Dimension (2 x PROJECTED_DIMENSION_G, )
            self.concat_vs = tf.reshape(self.concat_vs, [1, 2*self.PROJECTED_DIMENSION_G])

            self.h_logits = tf.contrib.layers.fully_connected(self.concat_vs, self.PROJECTED_DIMENSION_H, activation_fn=None)
            self.h_output = tf.argmax(self.h_logits, axis=1, name='h_output')

            self.h_logits = tf.reshape(self.h_logits, (self.PROJECTED_DIMENSION_H,))

        with tf.variable_scope("train", reuse=tf.AUTO_REUSE) as scope:
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.h_logits, labels=self.labels),
                name="loss"
            )
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.variable_scope("eval", reuse=tf.AUTO_REUSE) as scope:
            correct_prediction = tf.equal(tf.argmax(self.labels, axis=0), self.h_output)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name="accuracy")

    def train(self, train_file_path, epoch_number, save_models = True):
        """ Train the attention NLI model stochastically
        Args:
            train_file_path: jsonl file path to training data
            epoch_number: number of epochs of training
            save_models: saving models for each epoch
        Notes:
            trained models are saved in models/ every 50 epochs
        """
        saver = tf.train.Saver(max_to_keep=500)
        self.dp_train = DataProcessor(input_file_path=train_file_path)
        acc_list = []
        loss_list = []

        s1 = "someone to watch Netflix with me"
        s2 = "someone to watch TV shows"
        embeddings1 = self.dp_train.gloVe_embeddings(s1, self.token_count)
        embeddings2 = self.dp_train.gloVe_embeddings(s2, self.token_count)
        print("Prediction is:", self.sess.run(self.h_output, feed_dict = {self.a: embeddings1, self.b: embeddings2}))

        s1 = "a Penn student to chat for coffee"
        s2 = "chat and get to know a penn student"
        embeddings1 = self.dp_train.gloVe_embeddings(s1, self.token_count)
        embeddings2 = self.dp_train.gloVe_embeddings(s2, self.token_count)
        print("Prediction is:", self.sess.run(self.h_output, feed_dict = {self.a: embeddings1, self.b: embeddings2}))

        s1 = "a designer to cofound my startup"
        s2 = "a software designer interested in entrepreneurship"
        embeddings1 = self.dp_train.gloVe_embeddings(s1, self.token_count)
        embeddings2 = self.dp_train.gloVe_embeddings(s2, self.token_count)
        print("Prediction is:", self.sess.run(self.h_output, feed_dict = {self.a: embeddings1, self.b: embeddings2}))

        self.accuracy_records_by_epoch = []
        for i in range(epoch_number):
            data_num = 0
            for data in self.dp_train.get_single_data():
                data_num = data_num + 1
                data_feed_dict = {
                    self.a: data["sentence1"],
                    self.b: data["sentence2"],
                    self.labels: data["gold_label"]
                }
                _, acc, loss = self.sess.run([self.train_op, self.accuracy, self.loss], feed_dict=data_feed_dict)
                acc_list.append(acc)
                loss_list.append(loss)
                if (data_num % 1000 == 0):
                    print("At epoch: {}, {} data processed".format(i, data_num))
            epoch_acc = sum(acc_list)/len(acc_list)
            epoch_loss = sum(loss_list)/len(loss_list)
            self.accuracy_records_by_epoch.append(epoch_acc)
            print("finishing epoch {}, training accuracy: {}, loss:{}".format(i, epoch_acc, epoch_loss))

            if save_models:
                save_path = saver.save(self.sess, './models/', global_step=i)
                print("Model saved in file: %s" % save_path)
            elif not save_models and i + 1 == epoch_number:
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
        self.dp_test = DataProcessor(input_file_path=test_file_path)

        accuracies = []
        for data in self.dp_test.get_single_data():
            test_feed = {
                self.a: data["sentence1"],
                self.b: data["sentence2"],
                self.labels: data["gold_label"]
            }
            accuracy, predictions = self.sess.run([self.accuracy, self.h_output], feed_dict=test_feed)
            print("predictions for the batch: {}".format(predictions))
            print("Actual gold labels: {}".format(test_data["gold_label"]))
            accuracies.append(accuracy)

        test_acc = sum(accuracies)/len(accuracies)
        print("Overall test accuracy is {}".format(test_acc))
        return test_acc

    def predict(self, sentence1, sentence2, model_path):
        """ Predict the relationship between 2 sentences according to a specified model
        """
        embeddings1 = self.dp_test.gloVe_embeddings(sentence1, self.token_count)
        embeddings2 = self.dp_test.gloVe_embeddings(sentence2, self.token_count)
      
        return self.predict_by_embeddings(embeddings1, embeddings2, model_path)

    def predict_by_embeddings(self, embeddings1, embeddings2, model_path):
        """Args:
            embeddings1: sentence1's embeddings
            embeddings2: sentence2's embeddings 
            model_path: path to the model to be restored
        """
        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(self.sess, model_path)
        feeds = {
            self.a: embeddings1,
            self.b: embeddings2,
        }
        returned_label = self.sess.run(self.h_output, feed_dict=feeds)
        if returned_label[0] == 0:
            return "entailment"
        elif returned_label[0] == 1:
            return "neutral"
        else:
            return "contradiction"

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

    def print_testing_accuracy_graph(self, epoch_number, test_file_path):
        """ Draw a line graph on testing accuracy by epoch number
        Args:
            epcoh_number: the largest number of epoch in the models
            test_file_path: tile that contains the test data
        """
        saver = tf.train.Saver(max_to_keep=500)
        
        test_accuracy_records_by_epoch = []
        acc_list = []
        loss_list = []

        for i in range(epoch_number):
            saver.restore(self.sess, "./models/-" + str(i))
            batch_num = 0
            for data in self.dp_test.get_single_data():
                feed_dict = {
                    self.a: data["sentence1"],
                    self.b: data["sentence2"],
                    self.labels: data["gold_label"],
                }
                
                acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict=feed_dict)
                acc_list.append(acc)
                loss_list.append(loss)
            test_accuracy_records_by_epoch.append(sum(batch_acc_list)/len(batch_acc_list))
            test_accuracy = sum(acc_list)/len(acc_list)
            test_loss = sum(loss_list)/len(loss_list)
            print("For model with epoch {} testing accuracy: {}, loss: {}".format(i, test_accuracy, test_loss))
            
        epoch_nums = [i+1 for i in range(len(accuracy_records_by_epoch))]

        plt.plot(epoch_nums, test_accuracy_records_by_epoch)
        plt.title('Test Accuracy on NLI task')
        plt.legend(['Test Accuracy'], loc='upper Left')
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.show()
