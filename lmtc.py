import os
import logging
import json
import re
import time
import tempfile
import glob
import tqdm
import tensorflow as tf
import numpy as np
from copy import deepcopy
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score
from json_loader import JSONLoader
from vectorizer import W2VTokenizer, ELMoTokenizer, BERTTokenizer
from data import DATA_SET_DIR, MODELS_DIR
from configuration import Configuration
from metrics import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
from neural_networks.bert import BERT
from neural_networks.lwan import LWAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

LOGGER = logging.getLogger(__name__)


class LMTC:

    def __init__(self):
        super().__init__()
        if 'elmo' in Configuration['model']['token_encoding']:
            self.vectorizer = ELMoTokenizer()
            self.vectorizer2 = W2VTokenizer(w2v_model=Configuration['model']['embeddings'])
        elif 'bert' in Configuration['model']['token_encoding']:
            self.vectorizer = BERTTokenizer(bert_version=Configuration['model']['bert'])
        else:
            self.vectorizer = W2VTokenizer(w2v_model=Configuration['model']['embeddings'])
        self.load_label_descriptions()

    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))

        frequent, few = [], []
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'],
                               '{}.json'.format(Configuration['task']['dataset']))) as file:
            data = json.load(file)
            none = set(data.keys())

        none = none.difference(train_concepts.union((rest_concepts)))
        parents = []
        for key, value in data.items():
            parents.extend(value['parents'])
        none = none.intersection(set(parents))

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))
        true_zero = deepcopy(zero)
        zero = zero + list(none)

        self.label_ids = dict()
        self.margins = [(0, len(frequent) + len(few) + len(true_zero))]
        k = 0
        for group in [frequent, few, zero]:
            self.margins.append((k, k + len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent) + len(few) + len(true_zero))

        label_terms = []
        self.label_terms_text = []
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([token for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])
            self.label_terms_text.append(data[label]['label'])

        if 'bert' not in Configuration['model']['token_encoder']:
            self.label_terms_ids = self.vectorizer.tokenize(label_terms,
                                                            max_sequence_size=Configuration['sampling'][
                                                                'max_label_size'], features=['word'])
        else:
            self.label_terms_ids = self.vectorizer.tokenize(label_terms,
                                                            max_sequence_size=Configuration['sampling']['max_label_size'])

        LOGGER.info('#Labels:         {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))

        # Compute label hierarchy depth and build labels' graph
        if 'graph' in Configuration['model']['architecture'].lower():
            self.labels_graph = np.zeros((len(self.label_ids), len(self.label_ids)), dtype=np.float32)
            # Populate label graph
            for label_id in self.label_ids.keys():
                if label_id in data:
                    parents = data[label_id]['parents'] if data[label_id]['parents'] is not None else []
                for parent_id in parents:
                    if parent_id in self.label_ids:
                        self.labels_graph[self.label_ids[label_id]][self.label_ids[parent_id]] = 1.0
        else:
            self.labels_graph = None

        self.true_labels_cutoff = len(frequent) + len(few) + len(true_zero)

    def load_dataset(self, dataset_name):
        """
        Load dataset and return list of documents
        :param dataset_name: the name of the dataset
        :return: list of Document objects
        """
        filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], dataset_name, '*.json'))
        loader = JSONLoader()

        documents = []
        for filename in tqdm.tqdm(sorted(filenames)):
            documents.append(loader.read_file(filename))

        return documents

    def process_dataset(self, documents):
        """
         Process dataset documents (samples) and create targets
         :param documents: list of Document objects
         :return: samples, targets
         """
        samples = []
        targets = []
        for document in documents:
            samples.append(document.tokens)
            targets.append(document.tags)

        del documents
        return samples, targets

    def encode_dataset(self, sequences, tags):
        samples = self.vectorizer.tokenize(sequences=sequences,
                                           max_sequence_size=Configuration['sampling']['max_sequence_size'])

        if 'elmo' in Configuration['model']['token_encoding']:
            samples2 = self.vectorizer2.tokenize(sequences=sequences,
                                                 max_sequence_size=Configuration['sampling']['max_sequence_size'])

        targets = np.zeros((len(sequences), len(self.label_ids)), dtype=np.int32)
        for i, (document_tags) in enumerate(tags):
            for tag in document_tags:
                if tag in self.label_ids:
                    targets[i][self.label_ids[tag]] = 1

        del sequences, tags

        if 'elmo' in Configuration['model']['token_encoding']:
            return [samples, samples2], targets

        return samples, targets

    def train(self):
        LOGGER.info('\n---------------- Train Starting ----------------')

        for param_name, value in Configuration['model'].items():
            LOGGER.info('\t{}: {}'.format(param_name, value))

        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')

        documents = self.load_dataset('train')
        train_samples, train_tags = self.process_dataset(documents)
        train_generator = SampleGenerator(train_samples, train_tags, experiment=self,
                                          batch_size=Configuration['model']['batch_size'])

        documents = self.load_dataset('dev')
        val_samples, val_tags = self.process_dataset(documents)
        val_generator = SampleGenerator(val_samples, val_tags, experiment=self,
                                        batch_size=Configuration['model']['batch_size'])

        # Compile neural network
        LOGGER.info('Compile neural network')
        LOGGER.info('------------------------------')
        if 'lwan' in Configuration['model']['architecture'].lower():
            model = LWAN(self.label_terms_ids, self.labels_graph, self.true_labels_cutoff)
            model = model.build_compile(n_hidden_layers=Configuration['model']['n_hidden_layers'],
                                        hidden_units_size=Configuration['model']['hidden_units_size'],
                                        dropout_rate=Configuration['model']['dropout_rate'],
                                        word_dropout_rate=Configuration['model']['word_dropout_rate'])
        else:
            model = BERT(self.label_terms_ids, dropout_rate=Configuration['model']['dropout_rate'])

        # Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=Configuration['model']['lr']),
                      loss='binary_crossentropy')

        model.summary(line_length=200, print_fn=LOGGER.info)

        with tempfile.NamedTemporaryFile(delete=True) as w_fd:
            weights_file = w_fd.name

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                           verbose=1, save_best_only=True, save_weights_only=True)

        # Fit model
        LOGGER.info('Fit model')
        LOGGER.info('-----------')
        start_time = time.time()
        fit_history = model.fit_generator(train_generator,
                                          validation_data=val_generator,
                                          epochs=Configuration['model']['epochs'],
                                          callbacks=[early_stopping, model_checkpoint])

        # model.save(os.path.join(MODELS_DIR, '{}_{}.h5'.format(Configuration['task']['dataset'].upper(),
        #                                                       Configuration['model']['architecture'].upper())))

        best_epoch = np.argmin(fit_history.history['val_loss']) + 1
        n_epochs = len(fit_history.history['val_loss'])
        val_loss_per_epoch = '- ' + ' '.join(
            '-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
            else '+' for i in range(1, len(fit_history.history['val_loss'])))
        LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
        LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

        del train_generator

        LOGGER.info('Load valid data')
        LOGGER.info('------------------------------')
        _, val_targets = self.encode_dataset(val_samples, val_tags)
        self.calculate_performance(model=model, generator=val_generator, true_targets=val_targets)

        LOGGER.info('Load test data')
        LOGGER.info('------------------------------')

        test_documents = self.load_dataset('test')
        test_samples, test_tags = self.process_dataset(test_documents)
        test_generator = SampleGenerator(test_samples, test_tags, experiment=self,
                                         batch_size=Configuration['model']['batch_size'])
        _, test_targets = self.encode_dataset(test_samples, test_tags)
        self.calculate_performance(model=model, generator=test_generator, true_targets=test_targets)

        total_time = time.time() - start_time
        LOGGER.info('\nTotal Training Time: {} secs'.format(total_time))

    def calculate_performance(self, model, generator, true_targets):

        predictions = model.predict(generator)
        pred_targets = (predictions > 0.5).astype('int32')

        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

        # Overall
        for labels_range, frequency, message in zip(self.margins,
                                                    ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)',
                                                     'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                r_k = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                p_k = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                rp_k = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                ndcg_k = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
            LOGGER.info('----------------------------------------------------')


class SampleGenerator(Sequence):
    """
    Generates data for Keras
    :return: x_batch, y_batch
    """

    def __init__(self, samples, targets, experiment, batch_size=32, shuffle=True):
        """Initialization"""
        self.data_samples = samples
        self.targets = targets
        self.batch_size = batch_size
        self.indices = np.arange(len(samples))
        self.experiment = experiment
        self.shuffle = shuffle
        if 'zero' in Configuration['model']['architecture'].lower():
            self.zero = True
        else:
            self.zero = False
        if 'graph' in Configuration['model']['architecture'].lower():
            self.graph_aware = True
        else:
            self.graph_aware = False

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.data_samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + targets
        samples = [self.data_samples[k] for k in indices]
        targets = [self.targets[k] for k in indices]
        # Vectorize inputs (x,y)
        x_batch, y_batch = self.experiment.encode_dataset(samples, targets)

        if self.zero and not self.graph_aware:
            x_batch = [x_batch, self.experiment.label_terms_ids[:self.experiment.true_labels_cutoff]]
            y_batch = y_batch[:, :self.experiment.true_labels_cutoff]
        elif self.graph_aware:
            x_batch = [x_batch, self.experiment.label_terms_ids, self.experiment.labels_graph]

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == '__main__':
    Configuration.configure()

    LMTC().train()
