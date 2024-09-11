'''
Script to get the score for each documents based on the query and rank them based on the score
For example:
for query q1, the documents are d1, d2, d3, d4, d5
the score for each document is 0.8, 0.6, 0.4, 0.2, 0.1
The output will be: d1, d2, d3, d4, d5

Finally, a file will be saved in txt format with the following format:
query_id, q0, document_id, rank, score, runid1

where:
the first column is the topic (query) number.
the second column is currently unused and should always be “Q0”.
the third column is the identifier of the retrieved document 
the fourth column is the rank the passage/document is retrieved.
the fifth column shows the score (integer/floating point) in descending order.
the sixth column is the ID of the run you are submitting.
'''
import logging
import torch
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import CustomEvaluationConstants as const
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
        '''Tokenize the data'''
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def compute_average_logits(self, query: str, documents: list[str], doc_ids: list[str]) -> list:
        '''Compute average logits for ranking documents'''
        scores = np.zeros(len(documents))

        for i in range(len(documents)):    # pylint: disable = consider-using-enumerate
            for j in range(i + 1, len(documents)): # pylint: disable = consider-using-enumerate
                prompt = self.make_prompt(query, documents[i], documents[j])
                input_ids, attention_mask = self.tokenize_data([prompt])
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.detach().cpu().numpy()
                
                scores[i] += logits[0, 1]  # Sum logits where doc[i] is considered better
                scores[j] += logits[0, 0]  # Sum logits where doc[j] is considered better

        avg_scores = scores / (len(documents) - 1)  # Average out the scores

        # Combine doc_ids, scores, and ranks
        results = sorted(zip(doc_ids, avg_scores), key=lambda x: x[1], reverse=True)
        return results

    def save_results(self, results: list, query_id: str, run_id: str, output_file: str) -> None:
        '''Save the results to a text file'''
        with open(output_file, 'a', encoding='utf-8') as file:
            for rank, (doc_id, score) in enumerate(results, start=1):
                file.write(f"{query_id}, Q0, {doc_id}, {rank}, {score:.4f}, {run_id}\n")

    def evaluate(self, data_path: str, output_file: str, run_id: str) -> None:
        '''Evaluate the model performance and save the results'''
        data = self.read_data(data_path)

        assert 'qid' in data.columns, "Data does not have the required 'qid' column..."
        assert 'query' in data.columns, "Data does not have the required 'query' column..."
        assert 'doc_content' in data.columns, "Data does not have the required 'doc_content' column..."
        assert 'docid' in data.columns, "Data does not have the required 'docid' column..."

        unique_queries = data['qid'].unique()
        
        logging.info("%s", '='*69)
        for idx, query_id in enumerate(unique_queries):
            logging.info("Evaluating query: %s/%s...", idx+1, len(unique_queries))
            subset = data[data['qid'] == query_id]
            query = subset['query'].iloc[0]
            documents = subset['doc_content'].tolist()
            doc_ids = subset['docid'].tolist()
            
            # Compute average logits
            results = self.compute_average_logits(query, documents, doc_ids)
            
            # Save results
            self.save_results(results, query_id, run_id, output_file)

        logging.info("Evaluation completed and results saved to %s", output_file)

if __name__ == "__main__":
    logging.info("Evaluating the model...")
    evaluator = CustomEvaluator()
    evaluator.evaluate(const.DATASET_PATH, const.SAVE_PATH, const.RUN_ID)
