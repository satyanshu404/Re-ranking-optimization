'''
Script to evaluate the performance of the re-ranking model

The script takes the uses the following:
    - model_checkpoint: str
    - dataset_path: str

the dataset should be in the following format:
    qid \t docid \t query \t doc_content \t rank

the script calculates the following metrics:
    - NDGC
    - MAP
'''
import logging
import torch
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import CustomEvaluationConstants as const
from sklearn.metrics import ndcg_score, average_precision_score
from transformers import BertForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)

class CustomEvaluator:
    '''Class to evaluate the model performance'''
    def __init__(self,
                 model_checkpoint: str = const.MODEL_CHECKPOINT,
                 k: int = const.K) -> None:
        '''Initialize the class'''
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.batch_size = k
        self.k = k
        
        logging.info("Model loaded successfully from: %s ...", model_checkpoint)
        logging.info("Device: %s ...", self.device)
        logging.info("K: %s ...", self.k)
        logging.info("Batch Size: %s ...", self.batch_size)
    
    def read_data(self, data_path: str) -> pd.DataFrame:
        '''Read the data'''
        logging.info("Reading data from: %s...", data_path)
        df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
        return df
    
    def chunk_text(self, text: str) -> str:
        '''Split the text into chunks'''
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=const.CHUNK_SIZE,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks[0] if chunks else text
    
    def prompt_template(self, query: str, doc1: str, doc2: str) -> str:
        '''Prompt the user to select the best document'''
        text = f"Query: {query}\nDocument 1: {doc1}\nDocument 2: {doc2}"
        return text
    
    def make_prompt(self, query, response_a, response_b) -> str:
        '''Generate the prompt for the dataset'''
        text = self.prompt_template(self.chunk_text(str(query)),
                                    self.chunk_text(str(response_a)),
                                    self.chunk_text(str(response_b)))
        return text
    
    def tokenize_data(self, texts: list[str]) -> tuple:
        '''Tokenize the data 
        Args:
            data: list[str]: The data to tokenize
        Returns:
            tuple: The tokenized input_ids and attention_mask'''
        
        # Tokenize the data
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum(axis=0)
    
    def rank_documents(self, query: str, documents: list[str]) -> tuple:
        '''Rank the documents using pairwise comparison for one query with n documents
        Args:
            query: str: The query
            documents: list[str]: The documents to rank
            Returns:
            list[int]: The rank of documents
            torch.Tensor: The predicted relevance scores for each document
        '''
        scores = [0] * len(documents)  

        for i in range(len(documents)):   # pylint: disable = consider-using-enumerate
            for j in range(i + 1, len(documents)):
                # Generate prompt for pairwise comparison
                prompt = self.make_prompt(query, documents[i], documents[j])
                input_ids, attention_mask = self.tokenize_data([prompt])
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Model predicts Document 1 is more relevant than Document 2
                if logits[0, 1] > logits[0, 0]:  
                    scores[i] += 1
                else: 
                    scores[j] += 1

        # Rank based on scores
        ranked_docs = sorted(range(len(documents)), key=lambda x: scores[x], reverse=True)
        # calclate the softmax of scores
        final_predctions = self.softmax(np.array(scores))

        return ranked_docs, np.round(final_predctions, 2)
    
    def evaluate(self, data_path: str) -> None:
        '''Evaluate the model performance using the NDCG and MAP scores'''

        data = self.read_data(data_path)

        assert 'qid' in data.columns, "Data does not have the required 'query_id' column..."
        assert 'query' in data.columns, "Data does not have the required 'query' column..."
        assert 'doc_content' in data.columns, "Data does not have the required 'doc_content' column..."
        assert 'rank' in data.columns, "Data does not have the required 'rank' column..."
        assert 'score' in data.columns, "Data does not have the required 'score' column..."

        unique_queries = data['qid'].unique()
        ndcg_list = []
        ap_list = []
        
        logging.log(logging.INFO, "%s", '='*69)
        for idx, query_id in enumerate(unique_queries):
            logging.info("Evaluating query: %s/ %s...", idx+1, len(unique_queries))
            subset = data[data['qid'] == query_id]
            query = subset['query'].iloc[0]
            documents = subset['doc_content'].tolist()
            true_rank = subset['rank'].tolist()
            true_score = subset['score'].tolist()
            
            # true_score = self.softmax(true_score)
            true_score = [10+score for score in true_score]

            # Rank documents
            genertated_rank, predict_relevance = self.rank_documents(query, documents)
            
            # Create relevance scores based on original true_rank
            predicted_ranks = [true_rank[doc_id] for doc_id in genertated_rank]
            logging.info("Predicted Relevance: %s", predict_relevance)
            logging.info("True Relevance: %s", true_score)
            logging.info("Predicted Rank: %s", predicted_ranks)
            logging.info("True Rank: %s", true_rank)
            
            # Calculate NDCG and MAP
            ndcg = ndcg_score([true_score], [predict_relevance], k=self.k)
            # ap = MAP().average_precision_at_k(true_rank, predicted_ranks, self.k)
            ap = average_precision_score(true_score, predict_relevance)

            ndcg_list.append(ndcg)
            # ap_list.append(ap)

        avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0
        avg_map = sum(ap_list) / len(ap_list) if ap_list else 0

        logging.info("Average NDCG: %f", avg_ndcg)
        logging.info("Average MAP: %f", avg_map)

if __name__ == "__main__":
    # try:
    logging.info("Evaluating the model...")
    evaluator = CustomEvaluator()
    evaluator.evaluate(const.DATASET_PATH)
    # except Exception as e: # pylint: disable = broad-exception-caught
    #     logging.error("An error occurred: %s", e)
