'''
Script to get the score for each documents based on the query and rank them based on the score
For example:
for query q1, the documents are d1, d2, d3, d4, d5
the score for each document is 0.8, 0.6, 0.4, 0.2, 0.1
The output will be: d1, d2, d3, d4, d5

Finally, a file will be saved in the following format:
query_id, q0, document_id, rank, score, runid1

where:
the first column is the topic (query) number.
the second column is currently unused and should always be “Q0”.
the third column is the identifier of the retrieved document 
the fourth column is the rank the passage/document is retrieved.
the fifth column shows the score (integer/floating point) in descending order.
the sixth column is the ID of the run you are submitting.
'''
# Import the required libraries
import os
import torch
import logging
import pandas as pd
from constants import GetTheScoreConstants as const
from transformers import BertForSequenceClassification, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

class GetTheScore:
    '''Class to get the score for each document'''
    def __init__(self, 
                 model_checkpoint: str=const.MODEL_CHECKPOINT,
                 device: str = const.DEFAULT_DEVICE) -> None:
        '''Initialize the class'''
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model.to(self.device)
        self.model.eval()

        # loggings
        logging.log(logging.INFO, "MODEL INFORMATION: %s", '='*50)
        logging.log(logging.INFO, "Model and tokenizer loaded from: %s...", model_checkpoint)
        logging.log(logging.INFO, "Device set to: %s...", self.device)
        logging.log(logging.INFO, "%s", '='*69)

    def read_data(self, data_path: str) -> pd.DataFrame:
        '''Read the data'''
        return pd.read_csv(data_path, sep='\t', header=None, names=['query_id', 'query', 'document_id'])