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
        
        if const.DOCS_PER_QUERY != 100:
            res = res.groupby('qid').head(const.DOCS_PER_QUERY)

        if const.EVALUATE_FAIRNESS:
            assert 'annotation' in res.columns, "The result file should have 'annotation' column"
            assert 'prediction' in res.columns, "The result file should have 'prediction' column"
            res['annotation'] = res['annotation'].astype('Int64')
            res['prediction'] = res['prediction'].astype('Int64')

        # assert 'qid' in res.columns and 'docid' in res.columns, "The result file should have 'qid' and 'docid' columns"
        print(res.columns)
        if 'docno' not in res.columns:
            res['docno'] = res['docid']

        # qrels = pd.read_csv(self.qrel_path, sep='\t')
        qrels = pd.read_csv(self.qrel_path, sep=' ', names=['qid','run','docno','label'])
        if 'run' in qrels.columns:
            qrels = qrels.drop('run', axis=1)
        # qrels = qrels.drop('run', axis=1)
        # assert 'qid' in qrels.columns and 'docno' in qrels.columns, "The qrel file should have 'qid' and 'docno' columns"
        qrels['qid'] = qrels['qid'].astype('str')
        qrels['qid'] = qrels['qid'].astype('object')
        qrels['docno'] = qrels['docno'].astype('str')
        qrels['docno'] = qrels['docno'].astype('object')
        return res, qrels
    
    def complete_evaluate(self):
        ''' Evaluate the model '''
        res, qrels = self.load_files()
        # evaluate 
        logging.info("Evaluating the model...")
        metrics= AP(rel=2)@100, P@10, "map", NDCG(cutoff=1), NDCG(cutoff=5), NDCG(cutoff=10), NDCG(cutoff=100)  # pylint: disable = undefined-variable
        eval_ = pt.Evaluate(res, qrels, metrics=metrics)
        # scores_df = pd.DataFrame(scores)
        logging.info(eval_)

    def compute_fairness(self, df: pd.DataFrame, type_: str) -> tuple:
        '''compute the fraction of annotation for the relevant and non-relevant documents'''

        if type_ not in ['relevant', 'non-relevant']:
            raise ValueError("Type should be either 'relevant' or 'non-relevant'")
        
        label = 0
        if type_ == 'relevant':
            label = 1

        data = df[df['prediction'] == label]['annotation'].value_counts()
        annotation_count = data.to_dict()
        toatl_annotation_count = data.sum()
        frac_annotation = {k: v/toatl_annotation_count for k, v in annotation_count.items()}
        return toatl_annotation_count, frac_annotation

    def complete_evaluate_fairness(self):
        ''' Evaluate the fairness part (sentiment for the top-k docs for each query) '''
        logging.info("Evaluating the fairness part...")
        res, _ = self.load_files()

        # evaluate
        # relevant
        total_rel_annotation_count, relevant_frac_annotation = self.compute_fairness(res, 'relevant')
        logging.info("Total relevant count: %s", total_rel_annotation_count)
        logging.info("Fraction of annotation: %s", relevant_frac_annotation)

        # non-relevant
        total_non_annotation_count, non_relevant_frac_annotation = self.compute_fairness(res, 'non-relevant')
        logging.info("Total non-relevant count: %s", total_non_annotation_count)
        logging.info("Fraction of annotation: %s", non_relevant_frac_annotation)

    def get_dataframe_querywise(self, df: pd.DataFrame) -> pd.DataFrame:
        ''' Get the dataframe querywise '''
        return df.groupby('qid')

    def evalute_querywise(self):
        ''' Evaluate the model querywise '''
        res, qrels = self.load_files()
        # evaluate 
        logging.info("Evaluating the model querywise...")
        metrics= P@10, "map", NDCG(cutoff=1), NDCG(cutoff=5), NDCG(cutoff=10), NDCG(cutoff=100) # pylint: disable = undefined-variable
        
        rows = []
        for _, group in self.get_dataframe_querywise(res):
            eval_ = pt.Evaluate(group, qrels, metrics=metrics)
            row_dict = {
                'qid': group['qid'].iloc[0],
                'query': group['query'].iloc[0],
                'precision@10': eval_['P@10'],
                'map': eval_['map'], 
                'nDCG@1': eval_['nDCG@1'],
                'nDCG@5': eval_['nDCG@5'],
                'nDCG@10': eval_['nDCG@10'],
                'nDCG@100': eval_['nDCG@100']
            }
            rows.append(row_dict)
        results = pd.DataFrame(rows)
        results.to_csv(const.MERTIC_PATH, sep='\t', index=False)
        logging.info("Querywise evaluation saved at %s...", const.MERTIC_PATH)

    def evaluate_fairness_querywise(self):
        ''' Evaluate the fairness part querywise '''
        logging.info("Evaluating the fairness part querywise...")
        res, _ = self.load_files()
        # evaluate
        rows = []
        for _, group in self.get_dataframe_querywise(res):
            # relevant
            total_rel_annotation_count, relevant_frac_annotation = self.compute_fairness(group, 'relevant')
            # non-relevant
            total_non_annotation_count, non_relevant_frac_annotation = self.compute_fairness(group, 'non-relevant')
            row_dict = {
                'qid': group['qid'].iloc[0],
                'query': group['query'].iloc[0],
                'total_rel_annotation_count': total_rel_annotation_count,
                'relevant_frac_annotation': relevant_frac_annotation,
                'total_non_annotation_count': total_non_annotation_count,
                'non_relevant_frac_annotation': non_relevant_frac_annotation
            }
            rows.append(row_dict)
        results = pd.DataFrame(rows)
        results.to_csv(const.FAIRNESS_METRIC_PATH, sep='\t', index=False)
        logging.info("Querywise fairness evaluation saved at %s...", const.FAIRNESS_METRIC_PATH)


if __name__ == "__main__":
    QREL_PATH = const.QREL_PATH
    RESULT_PATH = const.RESULT_PATH
    evaluator = Evaluator(RESULT_PATH, QREL_PATH)
    evaluator.complete_evaluate()

    if const.EVALUATE_FAIRNESS:
        evaluator.complete_evaluate_fairness()

    if const.EVALUATION_QUERYWISE:
        evaluator.evalute_querywise()
        evaluator.evaluate_fairness_querywise()
