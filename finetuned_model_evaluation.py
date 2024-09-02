'''
This script is for getting the performance of the model on the test data.
This script logs the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Confusion Matrix
'''
import logging
import torch
import pandas as pd
from constants import ModelEvaluationConstants as const
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from transformers import BertForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)

class ModelEvaluation:
    '''Class to evaluate the model performance'''
    def __init__(self,
                 model_checkpoint: str = const.MODEL_CHECKPOINT,
                 batch_size: int = const.BATCH_SIZE) -> None:
        '''Initialize the class'''
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()
        logging.log(logging.INFO, "Model loaded successfully from: %s ...", model_checkpoint)
        logging.log(logging.INFO, "Device: %s ...", self.device)
        logging.log(logging.INFO, "Batch Size: %s ...", self.batch_size)


    def read_data(self, data_path: str) -> pd.DataFrame:
        '''Read the data'''
        logging.log(logging.INFO, "Reading data from: %s...", data_path)
        return pd.read_csv(data_path, sep='\t', encoding='utf-8')
    
    def tokenize_data(self, texts: list[str]) -> tuple:
        '''Tokenize the data 
        Args:
            data: pd.DataFrame: The data to tokenize
        Returns:
            tuple: The tokenized input_ids and attention_mask'''
        
        # Tokenize the data
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, attention_mask

    def get_predictions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''Get the predictions from the model
        Args:
            input_ids: torch.Tensor: The input ids
            attention_mask: torch.Tensor: The attention mask
        Returns:
            torch.Tensor: The predictions from the model'''
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        return predictions
    
    def get_metrics(self, predictions: torch.Tensor, labels: torch.Tensor) -> tuple:
        '''Get the metrics for the model
        Args:
            predictions: torch.Tensor: The predicted values
            labels: torch.Tensor: The actual values
            Returns:
            tuple: The metrics for the model (accuracy, precision, recall, f1, confusion matrix)'''
        
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        confusion = confusion_matrix(labels, predictions)   
        return accuracy, precision, recall, f1, confusion
    
    
    def evaluate(self, data_path: str) -> None:
        '''Evaluate the model'''
        # Read the data
        data = self.read_data(data_path)

        # Check if the data has the required columns
        assert 'text' in data.columns, "Data does not have the required 'text' column..."
        assert 'label' in data.columns, "Data does not have the required 'label' column..." 
        logging.log(logging.INFO, "%s", '='*69)
        # Tokenize the data
        input_ids, attention_mask = self.tokenize_data(data['text'].to_list())

        # Get the predictions
        # predictions = self.get_predictions(input_ids, attention_mask)
        all_predictions = []

        # Process in batches
        logging.log(logging.INFO, "Evaluating the model...")
        for start_idx in range(0, len(data), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(data))
            batch_texts = data['text'].iloc[start_idx:end_idx].to_list()

            # Tokenize the batch
            input_ids, attention_mask = self.tokenize_data(batch_texts)

            # Get the predictions
            predictions = self.get_predictions(input_ids, attention_mask)

            # Collect predictions
            all_predictions.extend(predictions)

        # Get the metrics
        accuracy, precision, recall, f1, confusion = self.get_metrics(all_predictions, data['label'].to_list())


        # Log the metrics
        logging.log(logging.INFO, "Model Performance Metrics: ...")
        logging.log(logging.INFO, "Accuracy: %.2f", accuracy)
        logging.log(logging.INFO, "Precision: %.2f", precision)
        logging.log(logging.INFO, "Recall: %.2f", recall)
        logging.log(logging.INFO, "F1 Score: %.2f", f1)
        logging.log(logging.INFO, "Confusion Matrix:\n %s", confusion)


if __name__ == "__main__":
    try:
        model_eval = ModelEvaluation()
        model_eval.evaluate(const.TEST_DATASET_PATH)
    except Exception as e: # pylint: disable = broad-exception-caught
        logging.log(logging.ERROR, e)
        
