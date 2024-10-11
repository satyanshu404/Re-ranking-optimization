'''
This script is use to get the lables for the query-doc pairs using the fine-tuned BERT model.
The input to this script are:
    - path to the dev/test dataset (top100 file of test/dev dataset)
    - path to the fine-tuned model
The output of this script is:
    - a file with the query-doc pairs in the ranked for each query 
'''
import os
import logging
import torch
import pandas as pd
from constants import GenerateScoresConstants as const
from transformers import BertForSequenceClassification, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = const.VISIBLE_DEVICES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO)


class GenerateScores:
    ''' Generate the scores for the query-doc pairs '''
    def __init__(self, model_checkpoint: str, device: torch.device="cpu", batch_size:int = const.BATCH_SIZE) -> None:
        ''' Initialize the class '''
        self.device = device
        self.batch_size = batch_size
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model.to(self.device) 
        self.model.eval()

        # loggings
        logging.log(logging.INFO, "MODEL INFORMATION: %s", '='*50)
        logging.log(logging.INFO, "Model and tokenizer loaded from: %s...", model_checkpoint)
        logging.log(logging.INFO, "Visible device set to: %s...", const.VISIBLE_DEVICES)
        logging.log(logging.INFO, "Device set to: %s...", self.device)
        logging.log(logging.INFO, "Batch size set to: %s...", self.batch_size)
        if const.SCORE_TYPE == 0:
            logging.log(logging.INFO, "Score type set to: Absolute score (logit 1)...")
        else:
            logging.log(logging.INFO, "Score type set to: Relative score (diff of logit 1 & logit 0)...")
        logging.log(logging.INFO, "%s", '='*69)

    def load_data(self, data_path: str) -> pd.DataFrame:
        ''' Load the data '''
        df = pd.read_csv(data_path, sep='\t')
        # df = df[:10]
        # print(df.head())
        return df
    
    def prompt_template(self, query:str, document: str) -> str:
        '''Prompt template'''
        return f"Query:{query}\nDocument:{document}"
    
    def tokenize_data(self, data: pd.DataFrame) -> tuple:
        ''' Tokenize the data for the model
        Returns:
            tuple: The input_ids and attention_mask for the data
        '''
        logging.info("Tokenizing the data...")
        assert 'query' in data.columns and 'doc' in data.columns, "The data should have 'query' and 'doc' columns"
        texts = data.apply(lambda x: self.prompt_template(x['query'], x['doc']), axis=1)
        inputs = self.tokenizer(texts.tolist(), padding=True, truncation=True)
        return inputs['input_ids'], inputs['attention_mask']
    
    def get_predictions(self,  input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size:int) -> list:
        ''' Get the predictions for the input data
        Args:
            input_ids (torch.Tensor): The input_ids for the data
            attention_mask (torch.Tensor): The attention_mask for the data
            batch_size (int): The batch size for the predictions
        Returns:
            list: logits for the data
        '''
        # process in the batches
        logging.info("Getting predictions for the data...")
        predictions, scores = [], []
        with torch.no_grad():
            for i in range(0, len(input_ids), batch_size):
                batch_input_ids = torch.tensor(input_ids[i:i+batch_size]).to(self.device)
                batch_attention_mask = torch.tensor(attention_mask[i:i+batch_size]).to(self.device)
                
                outputs = self.model(batch_input_ids, 
                                     attention_mask=batch_attention_mask)
                # get the predictions
                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist())
                # probs = torch.softmax(outputs.logits, dim=-1)
                # preds = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
                # predictions.extend(preds.cpu().numpy())
                # get the scores (absolute or difference)
                if const.SCORE_TYPE == 0:
                    score = [x[1] for x in outputs.logits.cpu().numpy().tolist()]
                else:
                    score = [x[1]-x[0] for x in outputs.logits.cpu().numpy().tolist()]
                scores.extend(score)


        return predictions, scores
    
    def rank_data(self, data: pd.DataFrame) -> pd.DataFrame:
        ''' Rank the data based on the scores (logits) '''
        logging.info("Ranking the data...")
        assert 'qid' in data.columns and 'score' in data.columns, "The data should have 'qid' and 'score (logits)' columns"

        ranked_data = pd.DataFrame()
        unique_ids = data['qid'].unique()
        for query_id in unique_ids:
            query_data = data[data['qid'] == query_id]
            sorted_data = query_data.sort_values('score', ascending=False)
            sorted_data['rank'] = range(1, len(sorted_data) + 1)
            # sorted_data = sorted_data.sort_values('rank', ascending=True)
            sorted_data = sorted_data.reset_index(drop=True)
            ranked_data = pd.concat([ranked_data, sorted_data], ignore_index=True)

        if const.TYPE == 'fair':
            ranked_data = ranked_data[['qid', 'query', 'docid', 'rank', 'score', 'prediction', 'annotation']]
        else:
            ranked_data = ranked_data[['qid', 'docid', 'rank', 'score', 'prediction']]
        return ranked_data
    
    def get_scores(self, file_path: str, save_path:str) -> None:
        ''' Get the scores for the data '''
        # Load the data
        data = self.load_data(file_path)
        # Tokenize the data
        input_ids, attention_mask = self.tokenize_data(data)
        # Get the predictions
        data['prediction'], data['score'] = self.get_predictions(input_ids, attention_mask, self.batch_size)
        # Rank the data
        ranked_data = self.rank_data(data)

        # Save the data
        ranked_data.to_csv(save_path, sep='\t', index=False)
        logging.info("Data saved to %s...", save_path)


if __name__ == "__main__":
    # Initialize the class
    generate_scores = GenerateScores(const.MODEL_CHECKPOINT,
                                     device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                     batch_size=const.BATCH_SIZE)
    generate_scores.get_scores(const.TEST_DATASET_PATH,
                               const.SAVE_PATH)       