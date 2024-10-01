'''
This script is to evalutate the trained model on the test data.
This script requires the result file and the qrel file to evaluate the model.
Pyterrier is used to evaluate the model.
'''
import logging
import pandas as pd
import pyterrier as pt 
from pyterrier.measures import *  # pylint: disable = wildcard-import, disable = unused-wildcard-import
from constants import EvaluationConstants as const
if not pt.started(): 
    pt.init()

logging.basicConfig(level=logging.INFO)

class Evaluator:
    ''' Class to evaluate'''
    def __init__(self, result_path: str, qrel_path: str):
        ''' Initialize the class '''
        self.result_path = result_path
        self.qrel_path = qrel_path

    def load_files(self) -> tuple:
        ''' Read the result and qrel files'''
        # evaluate 
        res = pd.read_csv(self.result_path, encoding='utf-8', sep='\t')
        res = res.loc[:, ~res.columns.str.contains('^Unnamed')]

        # assert 'qid' in res.columns and 'docid' in res.columns, "The result file should have 'qid' and 'docid' columns"
        print(res.columns)
        if 'docno' not in res.columns:
            res['docno'] = res['docid']

        # qrels = pd.read_csv(self.qrel_path, sep='\t')
        qrels = pd.read_csv(self.qrel_path, sep=' ', names=['qid','run','docno','label'])
        qrels = qrels.drop('run', axis=1)
        # qrels = qrels.drop('run', axis=1)
        # assert 'qid' in qrels.columns and 'docno' in qrels.columns, "The qrel file should have 'qid' and 'docno' columns"
        qrels['qid'] = qrels['qid'].astype('str')
        qrels['qid'] = qrels['qid'].astype('object')
        qrels['docno'] = qrels['docno'].astype('str')
        qrels['docno'] = qrels['docno'].astype('object')
        return res, qrels
    
    def evaluate(self):
        ''' Evaluate the model '''
        res, qrels = self.load_files()
        # evaluate 
        logging.info("Evaluating the model...")
        metrics= AP(rel=2)@100, P@10, "map", NDCG(cutoff=1), NDCG(cutoff=5), NDCG(cutoff=10), NDCG(cutoff=100)  # pylint: disable = undefined-variable
        eval_ = pt.Evaluate(res, qrels, metrics=metrics)
        # scores_df = pd.DataFrame(scores)
        logging.info(eval_)

if __name__ == "__main__":
    QREL_PATH = const.QREL_PATH
    RESULT_PATH = const.RESULT_PATH
    evaluator = Evaluator(RESULT_PATH, QREL_PATH)
    evaluator.evaluate()
