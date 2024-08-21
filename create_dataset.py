"""
This script generates a dataset for the re-ranking optimization model. 
The dataset is structured as follows:
- Each row consists of a query, a corresponding positive document, and a negative document.
- Positive documents are labeled with 1, while negative documents are labeled with -1.
- The dataset is balanced by ensuring an equal number of positive and negative documents.
"""
import os
import logging
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import CreateDatasetConstants as const

logging.basicConfig(level=logging.INFO)

class PairWizeFormatter:
    '''Class to format the dataset in a pair-wise manner'''
    def __init__(self, triples_path: str, save_path: str):
        '''Initialize the class'''
        self.triples_path = triples_path
        self.save_path = save_path

        parent_dir = os.path.dirname(self.save_path)
        if not os.path.exists(parent_dir):
            logging.log(logging.INFO, "No existing directory '%s' found, creating one...", parent_dir)
            os.makedirs(parent_dir)

    def read_data(self) -> pd.DataFrame:
        '''Read the data from the triples file'''
        data = pd.read_csv(self.triples_path, sep='\t', header=None)
        return data
    
    def prompt_template(self, query:str, doc1:str, doc2:str) -> str:
        '''Prompt the user to select the best document'''
        text = f"Query:{query}\nDocument 1:{doc1}\nDocuemt 2:{doc2}"
        return text
      
    def chunk_text(self, text:str) -> str:
        '''Split the text into chunks'''
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=const.CHUNK_SIZE,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        return chunks[0]
    
    def make_prompt(self, queries, response_a, response_b) -> list[str]:
        '''Generate the prompt for the dataset'''
        text = [
            self.prompt_template(self.chunk_text(str(query)), 
            self.chunk_text(str(response_a)), 
            self.chunk_text(str(response_b)) )
            for query, response_a, response_b in zip(queries, response_a, response_b)
        ]
        return text
    
    def create_dataset(self) -> None:
        '''Create the dataset'''
        # Read the data
        data = self.read_data()

        # Generate the prompt
        prompts_1: str = self.make_prompt(data[1], data[5], data[9])
        prompts_2: str = self.make_prompt(data[1], data[9], data[5])

        # Create the dataframe
        df1 = pd.DataFrame({
            "text": prompts_1,
            "label": [1]*len(prompts_1)
        })

        df2 = pd.DataFrame({
            "text": prompts_2,
            "label": [-1]*len(prompts_2)
        })

        # Concatenate the dataframes
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

        # Save the data
        df.to_csv(self.save_path, sep='\t', index=None)
        logging.log(logging.INFO, "Dataset saved at %s", self.save_path)

if __name__ == "__main__":
    logging.log(logging.INFO, "Creating the dataset...")
    try:
        formatter = PairWizeFormatter(const.TRIPLES_PATH, const.SAVE_PATH)
        formatter.create_dataset()
    except Exception as e:      # pylint: disable = broad-exception-caught
        logging.log(logging.ERROR, "An error occurred: %s", e)