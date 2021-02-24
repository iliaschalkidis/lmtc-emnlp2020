import os
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.layers import Conv1D, Activation, Embedding
from tensorflow.keras.layers import Input, SpatialDropout1D
from tensorflow.keras.layers import GRU, add, concatenate
from tensorflow.keras.models import Model
from configuration import Configuration
from neural_networks.custom_layers import Camouflage, GlobalMeanPooling1D, TimestepDropout, \
    SymmetricMasking, ElmoEmbeddingLayer, ZeroLabelWiseAttention, LabelWiseAttention
from transformers import TFBertModel, TFAutoModel
from data import VECTORS_DIR


class LWAN:
    def __init__(self, label_terms_ids, labels_graph, true_labels_cutoff):
        super().__init__()
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.label_encoder = Configuration['model']['label_encoder']
        self.token_encoder = Configuration['model']['token_encoder']
        self.word_embedding_path = Configuration['model']['embeddings']
        self.label_terms_ids = label_terms_ids
        self.true_labels_cutoff = true_labels_cutoff
        self.bert_version = Configuration['model']['bert']
        if 'graph' in Configuration['model']['architecture'].lower():
            self.labels_graph = labels_graph
        else:
            self.labels_graph = None

    def build_compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate):

        if 'zero' in Configuration['model']['architecture']:
            return self._compile_label_wise_attention_zero(n_hidden_layers=n_hidden_layers,
                                                           hidden_units_size=hidden_units_size,
                                                           dropout_rate=dropout_rate,
                                                           word_dropout_rate=word_dropout_rate)
        else:
            return self._compile_label_wise_attention(n_hidden_layers=n_hidden_layers,
                                                      hidden_units_size=hidden_units_size,
                                                      dropout_rate=dropout_rate,
                                                      word_dropout_rate=word_dropout_rate)

    def PretrainedEmbedding(self):

        inputs = Input(shape=(None,), dtype='int32')
        embeddings = KeyedVectors.load_word2vec_format(os.path.join(VECTORS_DIR, self.word_embedding_path),
                                                       binary=False)
        word_embeddings_weights = K.cast_to_floatx(
            np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0))
        embeds = Embedding(len(word_embeddings_weights), word_embeddings_weights.shape[-1],
                           weights=[word_embeddings_weights], trainable=False)(inputs)

        return Model(inputs=inputs, outputs=embeds, name='embedding')

    def TokenEncoder(self, inputs, encoder, dropout_rate, word_dropout_rate, hidden_layers, hidden_units_size):

        # Apply variational drop-out
        inner_inputs = SpatialDropout1D(dropout_rate)(inputs)
        inner_inputs = TimestepDropout(word_dropout_rate)(inner_inputs)

        if encoder == 'grus':
            # Bi-GRUs over token embeddings
            for i in range(hidden_layers):
                bi_grus = Bidirectional(GRU(units=hidden_units_size, return_sequences=True, activation="tanh",
                                            recurrent_activation='sigmoid'))(inner_inputs)
                bi_grus = Camouflage(mask_value=0)(inputs=[bi_grus, inputs])
                if i == 0:
                    inner_inputs = SpatialDropout1D(dropout_rate)(bi_grus)
                else:
                    inner_inputs = add([bi_grus, inner_inputs])
                    inner_inputs = SpatialDropout1D(dropout_rate)(inner_inputs)

            outputs = Camouflage()([inner_inputs, inputs])
        elif encoder == 'cnns':
            # CNNs over token embeddings
            convs = Conv1D(filters=hidden_units_size, kernel_size=3, strides=1, padding="same")(inner_inputs)
            convs = Activation('tanh')(convs)
            convs = SpatialDropout1D(dropout_rate)(convs)
            outputs = Camouflage(mask_value=0)(inputs=[convs, inputs])
        elif encoder == 'bert':
            if self.bert_version != 'scibert':
                self.bert_model = TFBertModel.from_pretrained(self.bert_version, from_pt=True)
            else:
                self.bert_model = TFAutoModel.from_pretrained(self.bert_version)
            inner_inputs = self.bert_model(inputs)[0]
            inner_inputs = SpatialDropout1D(dropout_rate)(inner_inputs)
            outputs = Camouflage(mask_value=0)(inputs=[inner_inputs, inputs])
        else:
            raise NotImplementedError

        return outputs

    def LabelEncoder(self, inputs, encoder, dropout_rate, hidden_units_size):

        # Apply variational drop-out + Mask input to exclude paddings
        inner_inputs = SpatialDropout1D(dropout_rate)(inputs)

        if encoder == 'centroids':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inputs])
            outputs = GlobalMeanPooling1D()(inner_inputs)
        elif encoder == 'centroids+':
            inner_inputs = SymmetricMasking(mask_value=0.0)([inner_inputs, inputs])
            inner_inputs = GlobalMeanPooling1D()(inner_inputs)
            outputs = Dense(units=hidden_units_size)(inner_inputs)
        else:
            raise NotImplementedError

        return outputs

    def _compile_label_wise_attention(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate):

        # Document Encoding
        if self.elmo:
            inputs = Input(shape=(1,), dtype='string', name='inputs')
            elmo_embeddings = ElmoEmbeddingLayer()(inputs)
            inputs_2 = Input(shape=(None,), name='inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            w2v_embeddings = self.pretrained_embeddings(inputs_2)
            embeddings = concatenate([w2v_embeddings, elmo_embeddings])

        else:
            inputs = Input(shape=(None,), name='inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            embeddings = self.pretrained_embeddings(inputs)

        token_encodings = self.TokenEncoder(inputs=embeddings, encoder=self.token_encoder,
                                            dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate,
                                            hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size)

        # Label-wise Attention Mechanism matching documents with labels
        document_label_encodings = LabelWiseAttention(n_classes=len(self.label_terms_ids))(token_encodings)

        model = Model(inputs=[inputs] if not self.elmo else [inputs, inputs_2],
                      outputs=[document_label_encodings])

        return model

    def _compile_label_wise_attention_zero(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate):

        # Document Encoding
        if self.token_encoder == 'bert':
            inputs = Input(shape=(None,), name='word_inputs', dtype='int32')
            embeddings = [inputs]
        else:
            inputs = Input(shape=(None,), name='inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            embeddings = self.pretrained_embeddings(inputs)
            self._features = embeddings.shape[-1]

        token_encodings = self.TokenEncoder(inputs=embeddings, encoder=self.token_encoder,
                                            dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate,
                                            hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size)

        if self.labels_graph is None:
            self.label_terms_ids = self.label_terms_ids[:self.true_labels_cutoff]

        # Labels Encoding
        labels_inputs = Input(shape=(len(self.label_terms_ids),), dtype='int32', name='label_inputs')
        labels_embs = self.pretrained_embeddings(labels_inputs)

        label_encodings = self.LabelEncoder(labels_embs, encoder=self.label_encoder, dropout_rate=dropout_rate,
                                            hidden_units_size=token_encodings.shape[-1])

        # Set Labels' graph as input
        if self.labels_graph is not None:
            labels_graph = Input(shape=(len(self.labels_graph),), dtype='float32', name='label_graph')
        else:
            labels_graph = None

        # Label-wise Attention Mechanism matching documents with labels
        if labels_graph is None:
            outputs = \
                ZeroLabelWiseAttention(graph_op=None)([token_encodings, label_encodings])
        elif 'add' in Configuration['model']['architecture'].lower():
            outputs = \
                ZeroLabelWiseAttention(graph_op='add')([token_encodings, label_encodings, labels_graph])
        else:
            outputs = \
                ZeroLabelWiseAttention(graph_op='concat')([token_encodings, label_encodings, labels_graph])

        # Compile network
        return Model(inputs=[inputs, labels_inputs]
        if labels_graph is None else [inputs, labels_inputs, labels_graph],
                     outputs=[outputs])


if __name__ == '__main__':
    Configuration.configure()
    model = LWAN(label_terms_ids=np.ones((500,), dtype=np.int32), true_labels_cutoff=300, labels_graph=None)
    model = model.build_compile(1, 100, 0.1, 0.1)
    model([np.ones((2, 100,), dtype=np.int32), np.ones((2, 100,), dtype=np.int32), np.ones((500, 15,), dtype=np.int32)])
