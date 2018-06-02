import pickle

import edward as ed
from edward.models import Normal, Bernoulli
from tensorflow.contrib.tensorboard.plugins import projector

from utils import *


class bayesian_emb_model():
    def __init__(self, d, sess, logdir):
        self.K = d.K
        self.sess = sess
        self.logdir = logdir

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.target_placeholder = tf.placeholder(tf.int32)
                self.context_placeholder = tf.placeholder(tf.int32)
                self.labels_placeholder = tf.placeholder(tf.int32, shape=[d.n_minibatch])
                self.ones_placeholder = tf.placeholder(tf.int32)
                self.zeros_placeholder = tf.placeholder(tf.int32)

            # Index Masks
            with tf.name_scope('priors'):
                self.U = Normal(loc=tf.zeros((d.L_target, self.K), dtype=tf.float32),
                                scale=tf.ones((d.L_target, self.K), dtype=tf.float32))
                self.V = Normal(loc=tf.zeros((d.L_context, self.K), dtype=tf.float32),
                                scale=tf.ones((d.L_context, self.K), dtype=tf.float32))

        with tf.name_scope('natural_param'):
            # Taget and Context Indices
            with tf.name_scope('target_word'):
                pos_indexes = tf.where(
                    tf.equal(self.labels_placeholder, tf.ones(self.labels_placeholder.shape, dtype=tf.int32)))
                pos_words = tf.gather(self.target_placeholder, pos_indexes)
                self.p_rhos = tf.nn.embedding_lookup(self.U, pos_words)
                pos_contexts = tf.gather(self.context_placeholder, pos_indexes)
                self.pos_ctx_alpha = tf.nn.embedding_lookup(self.V, pos_contexts)

            with tf.name_scope('negative_samples'):
                neg_indexes = tf.where(
                    tf.equal(self.labels_placeholder, tf.zeros(self.labels_placeholder.shape, dtype=tf.int32)))
                neg_words = tf.gather(self.target_placeholder, neg_indexes)
                self.n_rho = tf.nn.embedding_lookup(self.U, neg_words)
                neg_contexts = tf.gather(self.context_placeholder, neg_indexes)
                self.neg_ctx_alpha = tf.nn.embedding_lookup(self.V, neg_contexts)

            # Natural parameter
            self.p_eta = tf.reduce_sum(tf.multiply(self.p_rhos, self.pos_ctx_alpha), -1)
            self.n_eta = tf.reduce_sum(tf.multiply(self.n_rho, self.neg_ctx_alpha), -1)

        self.y_pos = Bernoulli(logits=self.p_eta)
        self.y_neg = Bernoulli(logits=self.n_eta)

        # INFERENCE
        self.sigU = tf.nn.softplus(
            tf.matmul(tf.get_variable("sigU", shape=(d.L_target, 1), initializer=tf.ones_initializer()), tf.ones([1, self.K])),
            name="sigmasU")
        self.sigV = tf.nn.softplus(
            tf.matmul(tf.get_variable("sigV", shape=(d.L_context, 1), initializer=tf.ones_initializer()), tf.ones([1, self.K])),
            name="sigmasV")
        self.locU = tf.get_variable("qU/loc", [d.L_target, self.K], initializer=tf.zeros_initializer())
        #self.locV = tf.get_variable("qV/loc", [d.L_context, self.K], initializer=tf.zeros_initializer())

        self.qU = Normal(loc=self.locU, scale=self.sigU)
        self.qV = Normal(loc=d.pretreained_embeddings, scale=self.sigV)

        self.inference = ed.KLqp({self.U: self.qU, self.V: self.qV},
                                 data={self.y_pos: self.ones_placeholder,
                                       self.y_neg: self.zeros_placeholder
                                       })
        with self.sess.as_default():
            tf.global_variables_initializer().run()
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()

        alpha = config.embeddings.add()
        alpha.tensor_name = 'qU/loc'
        alpha.metadata_path = '../vocab_alpha.tsv'
        rho = config.embeddings.add()
        rho.tensor_name = 'qV/loc'
        rho.metadata_path = '../vocab_rho.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    def dump(self, fname, data):
        with self.sess.as_default():
            dat = {'rhos': self.qU.loc.eval(),
                   'alpha': self.qV.loc.eval(),
                   'sigma_rhos': self.sigU.eval()[:, 0],
                   'sigma_alphas': self.sigV.eval()[:, 0]}
            pickle.dump(dat, open(fname, "wb+"))

    def build_words_list(self, labels, list_length):
        if len(labels) < list_length:
            empty_list = [''] * (list_length - len(labels))
            labels.extend(empty_list)
        return labels
