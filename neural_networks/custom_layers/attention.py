import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    return tf.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)


class ZeroLabelWiseAttention(Layer):

    def __init__(self, graph_op='concat', cutoff=None):

        self.supports_masking = True
        self.graph_op = graph_op
        super(ZeroLabelWiseAttention, self).__init__()

    def get_config(self):
        config = super(ZeroLabelWiseAttention, self).get_config()
        config.update({"graph_op": self.graph_op})
        return config

    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        assert input_shape[0][-1] == input_shape[1][-1]

        self.W_d = self.add_weight(shape=(input_shape[1][-1], input_shape[0][-1]),
                                   name='{}_Wd'.format(self.name))

        self.b_d = self.add_weight(shape=(input_shape[1][-1],),
                                   initializer='zeros',
                                   name='{}_bd'.format(self.name))

        if self.graph_op is not None:
            self.W_p = []
            self.W_c = []
            self.W_s = []
            self.b_g = []
            for i in range(2):
                self.W_p.append(self.add_weight(shape=(input_shape[1][-1], input_shape[1][-1]),
                                                name='Wp{}'.format(i + 1),
                                                trainable=True))

                self.W_c.append(self.add_weight(shape=(input_shape[1][-1], input_shape[1][-1]),
                                                name='Wc{}'.format(i + 1),
                                                trainable=True))

                self.W_s.append(self.add_weight(shape=(input_shape[1][-1], input_shape[1][-1]),
                                                name='Ws{}'.format(i + 1),
                                                trainable=True))

                self.b_g.append(self.add_weight(shape=(input_shape[1][-1],),
                                                initializer='zeros',
                                                name='bg{}'.format(i + 1),
                                                trainable=True))

            if self.graph_op == 'concat':
                self.W_o = self.add_weight(shape=(input_shape[1][-1]*2, input_shape[1][-1]),
                                           name='{}_Wo'.format(self.name),
                                           trainable=True)
                self.b_o = self.add_weight(shape=(input_shape[1][-1]*2,),
                                           initializer='zeros',
                                           name='{}_bo'.format(self.name),
                                           trainable=True)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # Unfold inputs (document representations, label representations, labels' graph)
        if len(x) == 3:
            doc_reps, label_reps, labels_graph = x
        else:
            doc_reps, label_reps = x
            labels_graph = None

        doc2_reps = tf.tanh(dot_product(doc_reps, self.W_d) + self.b_d)

        # Compute Attention Scores
        doc_a = dot_product(doc2_reps, label_reps)

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = tf.nn.softmax(tf.transpose(ai))
            label_aware_doc_rep = dot_product(ai, tf.transpose(doc_repi))
            return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores = tf.map_fn(label_wise_attention, [doc_reps, doc_a])

        if labels_graph is not None:
            # 2-level Graph Convolution
            label_reps_p = dot_product(label_reps, self.W_p[0])
            label_reps_c = dot_product(label_reps, self.W_c[0])
            label_reps_s = dot_product(label_reps, self.W_s[0])
            graph_h_p = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_p, [1, 0]), labels_graph), [1, 0])
            graph_h_c = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_c, [1, 0]), K.permute_dimensions(labels_graph, [1, 0])), [1, 0])
            parents_sums = tf.expand_dims(tf.reduce_sum(labels_graph, axis=0), axis=-1)
            parents_sums = tf.where(tf.equal(0.0, parents_sums), tf.ones_like(parents_sums), parents_sums)
            children_sums = tf.expand_dims(tf.reduce_sum(K.permute_dimensions(labels_graph, [1, 0]), axis=0), axis=-1)
            children_sums = tf.where(tf.equal(0.0, children_sums), tf.ones_like(children_sums), children_sums)
            graph_h_p = graph_h_p / parents_sums
            graph_h_c = graph_h_c / children_sums
            label_reps_g = tf.tanh(label_reps_s + graph_h_p + graph_h_c + self.b_g[0])

            label_reps_p = dot_product(label_reps_g, self.W_p[1])
            label_reps_c = dot_product(label_reps_g, self.W_c[1])
            label_reps_s = dot_product(label_reps_g, self.W_s[1])
            graph_h_p = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_p, [1, 0]), labels_graph), [1, 0])
            graph_h_c = K.permute_dimensions(dot_product(K.permute_dimensions(label_reps_c, [1, 0]), K.permute_dimensions(labels_graph, [1, 0])), [1, 0])
            graph_h_p = graph_h_p / parents_sums
            graph_h_c = graph_h_c / children_sums
            label_reps_g = tf.tanh(label_reps_s + graph_h_p + graph_h_c + self.b_g[1])

            # Combine label embeddings + graph-aware label embeddings
            if self.graph_op == 'concat':
                label_reps = tf.keras.layers.concatenate([label_reps, label_reps_g], axis=-1)
                label_aware_doc_reprs = tf.tanh(dot_product(label_aware_doc_reprs, self.W_o) + self.b_o)
            else:
                label_reps = label_reps + label_reps_g

        # Compute label-scores
        label_aware_doc_reprs = tf.reduce_sum(label_aware_doc_reprs * label_reps, axis=-1)
        label_aware_doc_reprs = tf.sigmoid(label_aware_doc_reprs)

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[1][0]


class LabelWiseAttention(Layer):

    def __init__(self, n_classes=4271):

        self.supports_masking = True
        self.n_classes = n_classes
        super(LabelWiseAttention, self).__init__()

    def get_config(self):
        config = super(LabelWiseAttention, self).get_config()
        config.update({"n_classes": self.n_classes})
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.Wa = self.add_weight(shape=(self.n_classes, input_shape[-1]),
                                  trainable=True, name='Wa')

        self.Wo = self.add_weight(shape=(self.n_classes, input_shape[-1]),
                                  trainable=True, name='Wo')

        self.bo = self.add_weight(shape=(self.n_classes,),
                                  initializer='zeros',
                                  trainable=True, name='bo')

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        a = dot_product(x, self.Wa)

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = tf.nn.softmax(tf.transpose(ai))
            label_aware_doc_rep = dot_product(ai, tf.transpose(doc_repi))
            return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores = K.map_fn(label_wise_attention, [x, a])

        # Compute label-scores
        label_aware_doc_reprs = tf.reduce_sum(label_aware_doc_reprs * self.Wo, axis=-1) + self.bo
        label_aware_doc_reprs = tf.sigmoid(label_aware_doc_reprs)

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_classes
