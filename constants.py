'''All the constants used in the project are defined here'''
import json
import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class DownloadMSMARCOConstants:
    ''' Constants for downloading MSMARCO dataset '''
    SAVE_DIR = "data/MSMARCO"
    DOWNLOAD_DATASET_LINK_PATH = "data/msmarco-dataset-download-links.json"
    URLS: List[str] = field(init=False)

    def __post_init__(self):
        try:
            with open(self.DOWNLOAD_DATASET_LINK_PATH, 'r', encoding='utf-8') as file:
                self.URLS = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.DOWNLOAD_DATASET_LINK_PATH}..." ) from None
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {self.DOWNLOAD_DATASET_LINK_PATH}...") from None

@dataclass
class AllUrls:
    ''' All the URLs used in the project '''
    URLS: List = field(default_factory=lambda: {
        "msmarco": {
            "docs": "data/MSMARCO/documents/docs/msmarco-docs.tsv.gz",

            "train": {
                "subset_size": 5,
                "queries": "data/MSMARCO/documents/train/msmarco-doctrain-queries.tsv.gz",
                "qrels": "data/MSMARCO/documents/train/msmarco-doctrain-qrels.tsv.gz",
                "top100": "data/MSMARCO/documents/train/msmarco-doctrain-top100.gz",
                "relevant_save_path": "data/tmp_train/msmarco-doctrain-relevant.tsv",
                "non_relevant_save_path": "data/tmp_train/msmarco-doctrain-non-relevant.tsv",
            },
            "test": {
                "subset_size": 100,
                "queries": "data/MSMARCO/documents/test/msmarco-test2019-queries.tsv.gz",
                "qrels": "data/MSMARCO/documents/test/2019qrels-docs.txt",
                "top100": "data/MSMARCO/documents/test/msmarco-doctest2019-top100.gz"
            },
            "dl19-judged": {
                "subset_size": 100,
                "queries": "data/MSMARCO/documents/dl19_judged/2019queries-docs-judged.tsv.gz",
                "qrels": "data/MSMARCO/documents/dl19_judged/2019qrel-docs-judged.tsv.gz",
                "top100": "data/MSMARCO/documents/dl19_judged/2019top100-docs-judged.tsv.gz"
            }
        }
    })

@dataclass
class MapMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    DOCS_PATH = AllUrls().URLS["msmarco"]["docs"]

    DATASET_TYPE = "msmarco"
    TYPE = "train"  # train
    SUBSET_SIZE = AllUrls().URLS[DATASET_TYPE][TYPE]["subset_size"]
    QUERIES_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["queries"]
    QRELS_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["qrels"]
    TOP100_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["top100"]
    RELEVANT_SAVE_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["relevant_save_path"]
    NON_RELEVANT_SAVE_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["non_relevant_save_path"]

@dataclass
class MapAllMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    DOCS_PATH = AllUrls().URLS["msmarco"]["docs"]

    DATASET_TYPE = "msmarco"
    TYPE = "dl19-judged"  #test or dl19-judged
    SUBSET_SIZE = AllUrls().URLS[DATASET_TYPE][TYPE]["subset_size"]
    QUERIES_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["queries"]
    QRELS_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["qrels"]
    TOP100_PATH = AllUrls().URLS[DATASET_TYPE][TYPE]["top100"]
    FILE_NAME = TOP100_PATH.rsplit("/", maxsplit=1)[-1].split(".")[0]
    SAVE_PATH = f"data/tmp_test/msmarco-{FILE_NAME}-mapped.tsv"

@dataclass
class CreateDatasetConstants:
    ''' Constants for creating dataset '''
    RANDOM_STATE = 0
    NUMBER_OF_INSTANCE_PER_CLASS = 100000  # max 367012
    RELEVANT_DOCS_PATH = MapMSMARCOConstants.RELEVANT_SAVE_PATH
    NON_RELEVANT_DOCS_PATH = MapMSMARCOConstants.NON_RELEVANT_SAVE_PATH
    SAVE_PATH = f"data/tmp_train/msmarco-doc-train-{2*NUMBER_OF_INSTANCE_PER_CLASS}.tsv"

@dataclass
class BertConstants:
    ''' Constants for BERT model '''
    MODEL_CHECKPOINT = 'google-bert/bert-large-uncased'
    TEST_SIZE = 0.2
    BATCH_SIZE = 32
    VISIBLE_DEVICES = "1"
    RANDOM_STATE = 0
    LEARNING_RATE = 1e-5
    EPOCHS = 5
    NUMBER_OF_CLASSES = 2
    NUM_WORKERS = 4
    TOKENIZATION_BATCH_SIZE = 1000
    DATASET_PATH = CreateDatasetConstants.SAVE_PATH
    SAVE_MODEL_DIR = "models/Bert"
    SAVE_TOKENIZER_DIR = "models/Bert"
    SAVE_MODEL_PATH = f"bert-large-uncased-finetuned-v3.1-{2*CreateDatasetConstants.NUMBER_OF_INSTANCE_PER_CLASS}"

@dataclass
class FairnessDatasetCreationConstants:
    ''' Constants for creating fairness dataset '''
    DIRECTORY_PATH = '/home/satyanshu/satyanshu_fair_retrieval_data'
    SAVE_PATH = "data/fairness/all-fair-dataset-merged.tsv"
    QREL_PATH = "data/fairness/all-fair-dataset-merged-qrels.tsv"

@dataclass
class GenerateScoresConstants:
    ''' Constants for getting the score class'''
    MODEL_CHECKPOINT = os.path.join(BertConstants.SAVE_MODEL_DIR, BertConstants.SAVE_MODEL_PATH)
    VISIBLE_DEVICES = BertConstants.VISIBLE_DEVICES
    BATCH_SIZE = BertConstants.BATCH_SIZE

    TYPE = 'fair'          # fair or mamarco

    if TYPE == 'fair':
        TEST_DATASET_PATH = FairnessDatasetCreationConstants.SAVE_PATH
    else:
        TEST_DATASET_PATH = MapAllMSMARCOConstants.SAVE_PATH

    SCORE_TYPE = 1   # 0 for absolute score, 1 for relative score (diff of logit 1 & logit 0)
    FILE_NAME = "-".join(TEST_DATASET_PATH.split(".")[0].split("-")[:-1])
    SAVE_PATH = f"{FILE_NAME}-scores-{2*CreateDatasetConstants.NUMBER_OF_INSTANCE_PER_CLASS}.tsv"

@dataclass
class EvaluationConstants:
    ''' Constants for model evaluation '''
    DOCS_PER_QUERY = 100      # to consider all: 100
    EVALUATE_FAIRNESS = False
    EVALUATION_QUERYWISE = True

    if GenerateScoresConstants.TYPE == 'fair':
        EVALUATE_FAIRNESS = True
        QREL_PATH = FairnessDatasetCreationConstants.QREL_PATH
    else:
        QREL_PATH = MapAllMSMARCOConstants.QRELS_PATH

    if EVALUATION_QUERYWISE:
        MERTIC_PATH = f"data/fairness/scores/metrics-{DOCS_PER_QUERY}-{2*CreateDatasetConstants.NUMBER_OF_INSTANCE_PER_CLASS}.tsv"
        FAIRNESS_METRIC_PATH = f"data/fairness/scores/fairness-{DOCS_PER_QUERY}-{2*CreateDatasetConstants.NUMBER_OF_INSTANCE_PER_CLASS}.tsv"

    RESULT_PATH = GenerateScoresConstants.SAVE_PATH