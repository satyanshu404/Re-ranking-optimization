'''Script to fine-tune BERT model on the MS MARCO dataset
The dataset is expected to be in the TSV format with the following columns:
- text: The input text for the model
- label: The label for the input text

Change the hyperparameters in the BertConstants class from constants.py file.
'''
import os
import logging
from math import ceil
import torch
from tqdm import tqdm 
import pandas as pd
from constants import BertConstants as const
from losses import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset, Dataset
from torch.optim import AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = const.VISIBLE_DEVICES
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TokenData(Dataset):
    '''Dataset class for tokenized data'''
    def __init__(self, train_x: pd.DataFrame, train_y: pd.DataFrame, train_tokens: dict) -> None:
        '''Initialize the dataset'''
        self.text_data = train_x
        self.tokens = train_tokens
        self.labels = list(train_y)
        
    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        sample = {k: torch.tensor(v[idx]) for k, v in self.tokens.items()}
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

class TokenIterableData(IterableDataset): # pylint: disable = abstract-method
    '''IterableDataset for large datasets'''
    def __init__(self, data, labels, tokenizer, batch_size=const.BATCH_SIZE):
        '''Initialize the class'''
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def preprocess_batch(self, start_idx, end_idx):
        '''Preprocess the batch'''
        batch_data = list(self.data[start_idx:end_idx])
        tokens = self.tokenizer(batch_data, padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor(self.labels[start_idx:end_idx])
        tokens['labels'] = labels
        return tokens

    def __iter__(self):
        '''Iterate over the data'''
        for start_idx in range(0, len(self.data), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(self.data))
            yield self.preprocess_batch(start_idx, end_idx)

class Bert:
    """BERT model for sequence classification"""
    def __init__(self, model_checkpoint:str=const.MODEL_CHECKPOINT,
                 num_labels: int=const.NUMBER_OF_CLASSES,
                 device: torch.device="cpu") -> None:
        """Initialize the BERT model"""
        self.device = device
        self.seed = const.RANDOM_STATE
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.optimizer = AdamW(self.model.parameters(), lr=const.LEARNING_RATE)
        self.model.to(self.device)

        # loggings
        logging.info("MODEL INFORMATION: %s", '='*50)
        logging.info("Model and tokenizer loaded from: %s...", model_checkpoint)
        logging.info("Number of classes set to: %s...", num_labels)
        logging.info("Optimizer set to: %s...", self.optimizer.__class__.__name__)
        logging.info("Batch size set to: %s...", const.BATCH_SIZE)
        logging.info("Tokenizer batch size set to: %s...", const.TOKENIZATION_BATCH_SIZE)
        logging.info("Learning rate set to: %s...", const.LEARNING_RATE)
        logging.info("Visible device set to: %s...", const.VISIBLE_DEVICES)
        logging.info("Device set to: %s...", self.device)
        logging.info("Random seed set to: %s...", self.seed)
        logging.info("%s", '='*69)

    def read_data(self, data_path: str) -> pd.DataFrame:
        '''Read the data from the given tsv path only'''
        assert data_path.endswith(".tsv"), "Only TSV files are supported..."
        logging.info("Reading the data...")
        data = pd.read_csv(data_path, sep="\t")
        logging.info("Data shape: %s...", data.shape)
        return data
    
    def batch_tokenize(self, data: list, batch_size: int) -> dict:
        '''Tokenize the data in batches'''
        tokens = {"input_ids": [], "attention_mask": []}
        
        # Calculate total batches including remainder
        total_batches = ceil(len(data) / batch_size)
        
        # Tokenize in batches
        for i in tqdm(range(0, len(data), batch_size), desc="Tokenizing dataset: ", unit=" batch", total=total_batches):
            batch_data = data[i:i + batch_size]
            batch_tokens = self.tokenizer(batch_data, padding=True, truncation=True)  
            tokens["input_ids"].extend(batch_tokens["input_ids"])
            tokens["attention_mask"].extend(batch_tokens["attention_mask"])
        
        return tokens

    def prepare_data(self,
                     data: pd.DataFrame,
                     labels: pd.DataFrame,
                     test_size: float=const.TEST_SIZE,
                     batch_size: int=const.BATCH_SIZE,
                     num_workers: int=const.NUM_WORKERS) -> tuple:
        '''Prepare data using IterableDataset for large datasets'''
        logging.info("Preparing dataset for training...")

        # Split dataset
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=self.seed)
        logging.info("Train data shape: %s, Test data shape: %s", x_train.shape, x_test.shape)

        # Tokenize train and test data in batches
        logging.info("Tokenizing the data...")
        logging.info("Tokenizing train data...")
        train_tokens = self.batch_tokenize(list(x_train), batch_size=const.TOKENIZATION_BATCH_SIZE)
        logging.info("Tokenizing test data...")
        test_tokens = self.batch_tokenize(list(x_test), batch_size=const.TOKENIZATION_BATCH_SIZE)

        # Use TokenData for datasets
        logging.info("Creating DataLoader for training and testing...")
        train_dataset = TokenData(x_train, y_train, train_tokens)
        test_dataset = TokenData(x_test, y_test, test_tokens)

        # DataLoader with optimization
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                                pin_memory=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, 
                                pin_memory=True, num_workers=num_workers)

        logging.log(logging.INFO, "Train loader batch size: %s, dataset size: %s",
                    train_loader.batch_size, len(train_loader.dataset))
        logging.log(logging.INFO, "Test loader batch size: %s, dataset size: %s",
                    test_loader.batch_size, len(test_loader.dataset))

        return train_loader, test_loader

    def evaluator(self, test_loader: DataLoader, loss_fn) -> None:
        '''Evaluate the model'''
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False, total= len(test_loader)):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                pred = outputs.logits
                loss = loss_fn(pred, labels)

                total_loss += loss.item()
                correct += (pred.argmax(1) == labels).sum().item()

            average_loss = total_loss / len(test_loader)
            accuracy = correct / len(test_loader.dataset)
            logging.info("Test loss: %s, Test accuracy: %s", average_loss, accuracy)


    def trainer(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int, loss_fn) -> None:
        '''Train the model'''
        logging.info("Training the model...")
        for epoch in range(epochs):
            self.model.train()
            logging.info("Epoch %d: %s", epoch + 1, '='*50)
            total_train_loss = 0.0
            correct = 0
            for batch in tqdm(train_loader, desc="Training", leave=False, total= len(train_loader)):
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                pred = outputs.logits
                loss = loss_fn(pred, labels)
                
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                correct += (pred.argmax(1) == labels).sum().item()
            
            average_train_loss = total_train_loss / len(train_loader)
            accuracy = correct / len(train_loader.dataset)
            logging.info("Train loss: %s, Train accuracy: %s", average_train_loss, accuracy)

            # Call the evaluator function for the test set
            self.evaluator(test_loader, loss_fn)


    def train(self, file_path: str, loss_function, epochs: int=const.EPOCHS) -> None:
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
        logging.info("Training completed...")
        
        # Check if the directory exists
        if not os.path.exists(const.SAVE_MODEL_DIR):
            logging.info("No existing directory '%s' found, creating one...", const.SAVE_MODEL_DIR)
            os.makedirs(const.SAVE_MODEL_DIR)

        # Save the model and tokenizer
        save_path: str = os.path.join(const.SAVE_MODEL_DIR, const.SAVE_MODEL_PATH)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logging.info("Model and tokenizer saved successfully at %s...", const.SAVE_MODEL_DIR)

if __name__ == "__main__":
    # variables
    MODEL_NAME = const.MODEL_CHECKPOINT
    DATASET_PATH = const.DATASET_PATH
    EPOCHS = const.EPOCHS
    # LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    LOSS_FUNCTION = CrossEntropyLoss()   # custom loss function
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train the model
    bert = Bert(MODEL_NAME, device=DEVICE)
    bert.train(DATASET_PATH, LOSS_FUNCTION, EPOCHS)
