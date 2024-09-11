'''Script to fine-tune BERT model on the MS MARCO dataset
The dataset is expected to be in the TSV format with the following columns:
- text: The input text for the model
- label: The label for the input text

Change the hyperparameters in the BertConstants class from constants.py file.
'''
import os
import logging
import torch
import pandas as pd
from constants import BertConstants as const
from losses import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# Configure logging
logging.basicConfig(level=logging.INFO)
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
                 num_labels: int=const.NUMBER_OF_CLASSES,
                 device: torch.device="cpu") -> None:
        """Initialize the BERT model"""
        self.device = device
        self.model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.optimizer = AdamW(self.model.parameters(), lr=const.LEARNING_RATE)
        self.model.to(self.device)

        # loggings
        logging.log(logging.INFO, "MODEL INFORMATION: %s", '='*50)
        logging.log(logging.INFO, "Model and tokenizer loaded from: %s...", model_checkpoint)
        logging.log(logging.INFO, "Number of classes set to: %s...", num_labels)
        logging.log(logging.INFO, "Optimizer set to: %s...", self.optimizer.__class__.__name__)
        logging.log(logging.INFO, "Batch size set to: %s...", const.BATCH_SIZE)
        logging.log(logging.INFO, "Learning rate set to: %s...", const.LEARNING_RATE)
        logging.log(logging.INFO, "Device set to: %s...", self.device)
        logging.log(logging.INFO, "%s", '='*69)
    
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
        
        total_loss = 0.0
        correct = 0
        total_examples = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                pred = outputs.logits
                loss = loss_fn(pred, labels)

                total_loss += loss.item()
                correct += (pred.argmax(1) == labels).sum().item()
                total_examples += labels.size(0)
            average_loss = total_loss / len(test_loader)
            accuracy = correct / total_examples
            logging.log(logging.INFO, "Test loss: %s", average_loss)
            logging.log(logging.INFO, "Testing accuracy: %s", accuracy)

    
    def trainer(self,
                train_loader: DataLoader,
                test_loader: DataLoader,
                epochs: int,
                loss_fn) -> None:
        '''Train the model'''
        for epoch in range(epochs):
            self.model.train()
            logging.log(logging.INFO, "Epoch %d: %s", epoch + 1, '='*50)
            total_train_loss = 0.0
            
            for batch in train_loader:
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
            average_train_loss = total_train_loss / len(train_loader)
            logging.log(logging.INFO, "Training loss: %s", average_train_loss)

            # Call the evaluator function for the test set
            self.evaluator(test_loader, loss_fn)


    def train(self,
              file_path: str,
              loss_function,
              epochs: int=const.EPOCHS) -> None:
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
        save_path: str = os.path.join(const.SAVE_MODEL_DIR, const.SAVE_MODEL_PATH)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logging.log(logging.INFO, "Model and tokenizer saved successfully at %s...", const.SAVE_MODEL_DIR)

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