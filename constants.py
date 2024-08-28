'''All the constants used in the project are defined here'''
from dataclasses import dataclass, field
from typing import List


@dataclass
class BertConstants:
    ''' Constants for BERT model '''
    MODEL_CHECKPOINT = 'google-bert/bert-large-uncased'
    TEST_SIZE = 0.2
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    EPOCHS = 10
    NUMBER_OF_CLASSES = 2
    DATASET_PATH = "data/training/dataset.tsv"
    SAVE_MODEL_DIR = "models/Bert"
    SAVE_TOKENIZER_DIR = "models/Bert"
    SAVE_MODEL_PATH = "bert-large-uncased-finetuned-v0.2"

@dataclass
class CreateDatasetConstants:
    ''' Constants for creating dataset '''
    TRIPLES_PATH = "data/MSMARCO/triples.tsv"
    SAVE_PATH = "data/training/dataset.tsv"
    CHUNK_SIZE = 1500

@dataclass
class DownloadMSMARCOConstants:
    ''' Constants for downloading MSMARCO dataset '''
    DOWNLOAD_PATH = "data/MSMARCO"
    URLS: List = field(default_factory=lambda: [
        # Docs
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz",

        # Train dataset
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz",
        
        # Dev dataset
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz"
    ])

@dataclass
class MapMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    QUERIES_PATH = "data/MSMARCO/msmarco-doctrain-queries.tsv.gz"
    DOCS_LOOKUP_PATH = "data/MSMARCO/msmarco-docs-lookup.tsv.gz"
    DOCTRAIN_QRELS_PATH = "data/MSMARCO/msmarco-doctrain-qrels.tsv.gz"
    DOCTRAIN_TOP100_PATH = "data/MSMARCO/msmarco-doctrain-top100.gz"
    DOCS_PATH = "data/MSMARCO/msmarco-docs.tsv.gz"
    SAVE_PATH = "data/MSMARCO/triples_10k.tsv"
    NUMBER_OF_TRIPLES = 10000
    TRIPLET_COMPLETED = 1000

@dataclass
class MapAllMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    DOCS_PATH = "data/MSMARCO/msmarco-docs.tsv.gz"
    DOCS_LOOKUP_PATH = "data/MSMARCO/msmarco-docs-lookup.tsv.gz"
    QUERIES_PATH = "data/MSMARCO/msmarco-test2019-queries.tsv.gz"
    DOCDEV_QRELS_PATH = "data/MSMARCO/msmarco-docdev-qrels.tsv.gz"
    DOCDEV_TOP100_PATH = "data/MSMARCO/msmarco-doctest2019-top100.gz"
    SAVE_PATH = "data/MSMARCO/dev_all_mapped_t10.tsv"