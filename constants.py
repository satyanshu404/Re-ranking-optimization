'''All the constants used in the project are defined here'''
from dataclasses import dataclass

@dataclass
class BertConstants:
    ''' Constants for BERT model '''
    MODEL_CHECKPOINT = 'bert-base-uncased'

@dataclass
class MapMSMARCOConstants:
    ''' Constants for MAP@MSMARCO '''
    QUERIES_PATH = "data/MSMARCO/msmarco-doctrain-queries.tsv.gz"
    DOCS_LOOKUP_PATH = "data/MSMARCO/msmarco-docs-lookup.tsv.gz"
    DOCTRAIN_QRELS_PATH = "data/MSMARCO/msmarco-doctrain-qrels.tsv.gz"
    DOCTRAIN_TOP100_PATH = "data/MSMARCO/msmarco-doctrain-top100.gz"
    DOCS_PATH = "data/MSMARCO/msmarco-docs.tsv.gz"
    SAVE_PATH = "data/MSMARCO/triples.tsv"
    NUMBER_OF_TRIPLES = 1000