import tensorflow as tf
from transformers import TFAutoModel, TFBertModel, TFRobertaModel
from configuration import Configuration


class BERT(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, label_terms_ids, dropout_rate=0.1):
        super(BERT, self).__init__()
        self._decision_type = Configuration['task']['decision_type']
        self.bert_version = Configuration['model']['bert']
        self.n_classes = len(label_terms_ids)
        if self.bert_version == 'roberta-base-unncased':
            self.bert_model = TFRobertaModel.from_pretrained(self.bert_version)
        elif self.bert_version == 'bert-base-uncased':
            self.bert_model = TFBertModel.from_pretrained(self.bert_version)
        else:
            self.bert_model = TFAutoModel.from_pretrained(self.bert_version)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(self.n_classes, activation='sigmoid', name='outputs')

    def call(self, inputs):
        doc_encoding = self.bert_model(inputs)[0][:, 0, :]
        doc_encoding = self.dropout(doc_encoding)
        return self.classifier(doc_encoding)


