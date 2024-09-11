'''All the constants used in the project are defined here'''
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class BertConstants:
    ''' Constants for BERT model '''
    MODEL_CHECKPOINT = 'google-bert/bert-large-uncased'
    TEST_SIZE = 0.2
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-5
    EPOCHS = 10
    NUMBER_OF_CLASSES = 2
    DATASET_PATH = "data/tmp_train/msmarco-doc-train.tsv"
    SAVE_MODEL_DIR = "models/Bert"
    SAVE_TOKENIZER_DIR = "models/Bert"
    SAVE_MODEL_PATH = "bert-large-uncased-finetuned-v3.0"

@dataclass
class ModelEvaluationConstants:
    ''' Constants for model evaluation '''
    MODEL_CHECKPOINT = os.path.join(BertConstants.SAVE_MODEL_DIR, 'bert-large-uncased-finetuned-v2.1')
    BATCH_SIZE = 32
    TEST_DATASET_PATH = 'data/tmp_test/msmarco-doctrain-all-rel.tsv'

@dataclass
class GetTheScoreConstants:
    ''' Constants for getting the score class'''
    MODEL_CHECKPOINT = BertConstants.SAVE_MODEL_PATH
    DEFAULT_DEVICE = "cpu"
    TEST_DATASET_PATH = "data/training/dataset.tsv"

@dataclass
class MapMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    QUERIES_PATH = "data/MSMARCO/msmarco-doctrain-queries.tsv.gz"
    DOCS_LOOKUP_PATH = "data/MSMARCO/msmarco-docs-lookup.tsv.gz"
    DOCTRAIN_QRELS_PATH = "data/MSMARCO/msmarco-doctrain-qrels.tsv.gz"
    DOCTRAIN_TOP100_PATH = "data/MSMARCO/msmarco-doctrain-top100.gz"
    DOCS_PATH = "data/MSMARCO/msmarco-docs.tsv.gz"
    SUBSET_SIZE = 10
    SAVE_PATH = "data/tmp_train/msmarco-doctrain.tsv"
    RELEVANT_SAVE_PATH = "data/tmp_train/msmarco-doctrain-relevant.tsv"
    NON_RELEVANT_SAVE_PATH = "data/tmp_train/msmarco-doctrain-non-relevant.tsv"

@dataclass
class CreateDatasetConstants:
    ''' Constants for creating dataset '''
    SAVE_PATH = "data/tmp_train/msmarco-doc-train.tsv"
    CHUNK_SIZE = 1500
    RELEVANT_DOCS_PATH = MapMSMARCOConstants.RELEVANT_SAVE_PATH
    NON_RELEVANT_DOCS_PATH = MapMSMARCOConstants.NON_RELEVANT_SAVE_PATH

@dataclass
class CustomEvaluationConstants:
    ''' Constants for custom evaluation '''
    MODEL_CHECKPOINT = ModelEvaluationConstants.MODEL_CHECKPOINT
    DATASET_PATH = "data/dev/dev_all_mapped_t10_with_scores.tsv"
    K = 10
    CHUNK_SIZE = CreateDatasetConstants.CHUNK_SIZE
    SAVE_PATH = "results/trec_eval_2019.txt"
    RUN_ID = "trec_eval_2019"

@dataclass
class MapQrelsConstants:
    ''' Constants for mapping qrels '''
    QUERIES_PATH = "data/MSMARCO/msmarco-doctrain-queries.tsv.gz"
    DOCS_LOOKUP_PATH = "data/MSMARCO/msmarco-docs-lookup.tsv.gz"
    DOCTRAIN_QRELS_PATH = "data/MSMARCO/msmarco-doctrain-qrels.tsv.gz"
    QRELS_PATH = "data/MSMARCO/msmarco-doctrain-qrels.tsv.gz"
    DOCS_PATH = MapMSMARCOConstants.DOCS_PATH
    SAVE_PATH = "data/training/msmarco-doctrain-qrels-mapped-10k-n.tsv"
    TOP_K = 10000
    MAX_POSITIVE_DOC_COUNT = 16000

@dataclass
class MapAllMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    DOCS_PATH = "data/MSMARCO/msmarco-docs.tsv.gz"
    DOCS_LOOKUP_PATH = "data/MSMARCO/msmarco-docs-lookup.tsv.gz"
    QUERIES_PATH = "data/MSMARCO/msmarco-test2019-queries.tsv.gz"
    DOCDEV_QRELS_PATH = "data/MSMARCO/msmarco-docdev-qrels.tsv.gz"
    DOCDEV_TOP100_PATH = "data/MSMARCO/msmarco-doctest2019-top100.gz"
    SAVE_PATH = "data/MSMARCO/dev_all_mapped_t10.tsv"
    TOP_K_DOCUMENTS = 10

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
        
        # DL 19 dataset
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz",
        "https://trec.nist.gov/data/deep/2019qrels-docs.txt"
    ])
