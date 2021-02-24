# An Empirical Study on Large-Scale Multi-Label Text Classification Including Few and Zero-Shot Labels

This is the code used for the experiments described in the following paper:

> I. Chalkidis, M. Fergadiotis, S. Kotitsas, P. Malakasiotis, N. Aletras and I. Androutsopoulos, "An Empirical Study on Large-Scale Multi-Label Text Classification including Few and Zero-Shot Labels". In the Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), (Held online due to COVID-19), November 16–20, 2020


## Requirements:

* \>= Python 3.7
* == TensorFlow 2.3
* == TensorFlow-Hub 0.7.0
* >= Transformers 2.4.0
* \>= Gensim 3.5.0
* \>= Scikit-Learn 0.20.1
* \>= Spacy 2.1.0
* \>= TQDM 4.28.1

## Quick start:

### Install python requirements:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Download pre-trained word embeddings (GloVe):

```
wget -P data/vectors/ http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/glove.6B.zip data/vectors/glove.6B.200d.txt
echo -e "400000 200\n$(cat data/vectors/glove.6B.200d.txt)" > data/vectors/glove.6B.200d.txt
```

### Download dataset (EURLEX57K):

```
wget -O data/datasets/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/datasets/datasets.zip -d data/datasets/EURLEX57K
rm data/datasets/datasets.zip
rm -rf data/datasets/EURLEX57K/__MACOSX
mv data/datasets/EURLEX57K/dataset/* data/datasets/EURLEX57K/
rm -rf data/datasets/EURLEX57K/dataset
wget -O data/datasets/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
```

### Select training options from the configuration JSON file:

**Example Label-wise Attention Network with BIGRUs (BIGRU-LWAN)**

```
nano ltmc_configuration.json

{
  "task": {
    "dataset": "EURLEX57K",
  },
  "model": {
    "architecture": "BIGRU-LWAN",
    "token_encoder": "grus",
    "label_encoder": null,
    "n_hidden_layers": 1,
    "hidden_units_size": 300,
    "dropout_rate": 0.4,
    "word_dropout_rate": 0.00,
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "attention_mechanism": "attention",
    "token_encoding": "word2vec",
    "embeddings": "glove.6B.200d.txt",
    "bert": null
  },
  "sampling": {
    "max_sequence_size": 5000,
    "max_label_size": 15,
    "few_threshold": 50,
    "evaluation@k": 10
  }
}
```

**Example BERT-BASE (BERT-BASE)**

```
nano ltmc_configuration.json

{
  "task": {
    "dataset": "EURLEX57K",
  },
  "model": {
    "architecture": "BERT-BASE",
    "token_encoder": null,
    "label_encoder": null,
    "n_hidden_layers": null,
    "hidden_units_size": null,
    "dropout_rate": 0.1,
    "word_dropout_rate": null,
    "lr": 5e-5,
    "batch_size": 8,
    "epochs": 20,
    "token_encoding": "bert",
    "embeddings": null,
    "bert": "bert-base-uncased"
  },
  "sampling": {
    "max_sequence_size": 512,
    "max_label_size": 15,
    "few_threshold": 50,
    "evaluation@k": 10
  }
}
```


**Supported models:** LWAN-BIGRU, ZERO-LWAN-BIGRU, GRAPH-ZERO-LWAN-BIGRU, BERT-BASE, ROBERTA-BASE

**Supported token encodings:** word2vec, elmo, bert 

**Supported token encoders:** grus, cnns, bert

**Supported label encoders:** centroids, centroids+

### Train a model:

```
python lmtc.py
```
