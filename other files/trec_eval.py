''' 
The script evaluates the model performance using the BERT model.
'''
import os
import sys

# Set the PYTHONPATH to include the directory with the pyserini module
# sys.path.insert(0, os.path.abspath('./a s pyserini'))

# Import the pyserini module and run the evaluation commands
from pyserini.pyserini.eval import trec_eval

# Define the command arguments
args_map = ["-c", "-M", "10", "-m", "map", "dl19-doc", "results.trec_eval_2019.txt"]
args_ndcg_cut = ["-c", "-m", "ndcg_cut.10", "dl19-doc", "run.msmarco-v1-doc.bm25-doc-default.dl19.txt"]
args_recall = ["-c", "-m", "recall.1000", "dl19-doc", "run.msmarco-v1-doc.bm25-doc-default.dl19.txt"]

# Execute the evaluation functions
trec_eval.main(args_map)
# trec_eval.main(args_ndcg_cut)
# trec_eval.main(args_recall)
