'''All the constants used in the project are defined here'''
from dataclasses import dataclass, field
from typing import List


@dataclass
class BertConstants:
    ''' Constants for BERT model '''
    MODEL_CHECKPOINT = 'bert-base-uncased'
    TEST_SIZE = 0.2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    NUMBER_OF_CLASSES = 2
    DATASET_PATH = "data/training/dataset.tsv"
    SAVE_MODEL_DIR = "models/Bert"
    SAVE_TOKENIZER_DIR = "models/Bert"
    SAVE_MODEL_PATH = "bert-base-uncased-finetuned-v0.3"

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
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz"])

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