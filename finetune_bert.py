'''Script to fine-tune BERT model on the MS MARCO dataset'''
import os
import logging
import torch
import pandas as pd
from constants import BertConstants as const
from losses import cross_entropy_loss
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from Documents.testing.full_fine_tuning import DATASET_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO)

class TokenData(Dataset):
    '''Dataset class for tokenized data'''
    def __init__(self,
                 train_x: pd.DataFrame,
                 train_y: pd.DataFrame,
                 train_tokens: dict) -> None:
        '''Initialize the dataset'''
        self.text_data = train_x
        self.tokens = train_tokens
        self.labels = list(train_y)
        
    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.tokens.items():
            sample[k] = torch.tensor(v[idx])
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

class Bert:
    """BERT model for sequence classification"""
    def __init__(self, model_checkpoint:str=const.MODEL_CHECKPOINT,
                 num_labels: int=const.NUMBER_OF_CLASSES) -> None:
        """Initialize the BERT model"""
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.optimizer = AdamW(self.model.parameters(), lr=const.LEARNING_RATE)
        self.device = self.select_device()
        self.model.to(self.device)
        # loggings
        logging.log(logging.INFO, "Model and tokenizer loaded with %s classes...", num_labels)
        logging.log(logging.INFO, "Optimizer set to %s with learning rate %s...", self.optimizer, const.LEARNING_RATE)
        logging.log(logging.INFO, "Device set to %s...", self.device)

    def select_device(self, required_free_memory_gb: int=10) -> torch.device:
        '''Select the device based on the required free memory'''
        if torch.cuda.is_available():
            # Convert required memory to bytes
            required_free_memory = required_free_memory_gb * 1024 ** 3
            
            # Check if the 0th GPU has at least 10 GB of free memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
            if free_memory >= required_free_memory:
                device = torch.device("cuda:0")
                logging.log(logging.INFO, "Using GPU:0...")
            else:
                device = torch.device("cuda:1")
                logging.log(logging.INFO, "Using GPU:1...")
        else:
            device = torch.device("cpu")
            logging.log(logging.INFO, "No GPU has at least %s GB free, using CPU...", required_free_memory_gb)
        return device
    
    def read_data(self, data_path: str) -> pd.DataFrame:
        '''Read the data from the given tsv path only'''
        assert data_path.endswith(".tsv"), "Only TSV files are supported..."
        data = pd.read_csv(data_path, sep="\t")
        logging.log(logging.INFO, "Data read successfully...")
        return data
    
    def prepare_data(self,
                     data: pd.DataFrame,
                     labels: pd.DataFrame,
                     test_size: float=const.TEST_SIZE) -> tuple:
        '''Prepare the data for training and testing'''
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size)

        train_tokens = self.tokenizer(list(x_train), padding = True, truncation=True)
        test_tokens = self.tokenizer(list(x_test), padding = True, truncation=True)

        train_dataset = TokenData(x_train, y_train, train_tokens)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=const.BATCH_SIZE)

        test_dataset = TokenData(x_test, y_test, test_tokens)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=const.BATCH_SIZE)

        logging.log(logging.INFO, "Train and test data prepared successfully...")
        return train_loader, test_loader
    
    def evaluator(self,
                  test_loader: DataLoader,
                  loss_fn) -> None:
        '''Evaluate the model'''
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                pred = outputs.logits
                loss = loss_fn(pred, batch['labels'])
                # loss = outputs[0]
                # Calculating the running loss for logging purposes
                test_batch_loss = loss.item()
                test_last_loss = test_batch_loss / const.BATCH_SIZE
                logging.log(logging.INFO, "Testing batch %s last loss: %s", idx + 1, test_last_loss)
    
    def trainer(self,
                train_loader: DataLoader,
                test_loader: DataLoader,
                epochs: int,
                loss_fn) -> None:
        '''Train the model'''
        for epoch in range(epochs):
            self.model.train()
            for idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                pred = outputs.logits
                loss = loss_fn(pred, batch['labels'])
                # loss = outputs[0]
                loss.backward()
                self.optimizer.step()
                # Calculating the running loss for logging purposes
                train_batch_loss = loss.item()
                train_last_loss = train_batch_loss / const.BATCH_SIZE
                logging.log(logging.INFO, "Training batch %s last loss: %s", idx + 1, train_last_loss)
            logging.log(logging.INFO, "Epoch %s completed...", epoch+1)
            self.evaluator(test_loader, loss_fn)

    def train(self,
              file_path: str,
              epochs: int=const.EPOCHS,
              loss_function = cross_entropy_loss()) -> None:
        '''Train the model'''
        # Read the data
        data = self.read_data(file_path)

        # Check if the data has the required columns
        assert 'text' in data.columns, "Data does not have the required 'text' column..."
        assert 'label' in data.columns, "Data does not have the required 'label' column..."

        # Prepare the data
        train_loader, test_loader = self.prepare_data(data['text'], data['label'])
        
        # Train the model
        self.trainer(train_loader, test_loader, epochs, loss_function)
        logging.log(logging.INFO, "Training completed...")
        
        # check if the directory exists
        if not os.path.exists(const.SAVE_MODEL_DIR):
            logging.log(logging.INFO, "No existing directory '%s' found, creating one...", const.SAVE_MODEL_DIR)
            os.makedirs(const.SAVE_MODEL_DIR)

        # Save the model and tokenizer
        self.model.save_pretrained(const.SAVE_MODEL_DIR)
        self.tokenizer.save_pretrained(const.SAVE_MODEL_DIR)
        logging.log(logging.INFO, "Model and tokenizer saved successfully at %s...", const.SAVE_MODEL_DIR)

if __name__ == "__main__":
    # variables
    MODEL_NAME = const.MODEL_CHECKPOINT
    DATASET_PATH = const.DATASET_PATH
    EPOCHS = const.EPOCHS
    LOSS_FUNCTION = cross_entropy_loss()

    # train the model
    bert = Bert(MODEL_NAME)
    bert.train(DATASET_PATH, EPOCHS, LOSS_FUNCTION)