"""
This script generates a dataset for fine tuining the re-ranking model.
The input data must be in the following format:
-  Each row should consist of a qid, docid, query, document.
-  There must be two files as input one for relevant and other for non-relevant documents.

The script will generate a dataset in the following format:
- Relevant doc A will be labeled as 1 and non-relevant doc B will be labeled as 0.
- The dataset will be balanced by ensuring an equal number of positive and negative documents.
"""
import os
import logging
import pandas as pd
from constants import CreateDatasetConstants as const

logging.basicConfig(level=logging.INFO)

class CreateDataset:
    '''Class to create the dataset for fine-tuning the model'''
    def __init__(self, rel_docs_path:str, non_rel_docs_path:str, save_path:str):
        '''Initialize the class'''
        
        self.relevant_docs_path = rel_docs_path
        self.non_relevant_docs_path = non_rel_docs_path
        self.save_path = save_path

        parent_dir = os.path.dirname(self.save_path)
        if not os.path.exists(parent_dir):
            logging.log(logging.INFO, "No existing directory '%s' found, creating one...", parent_dir)
            os.makedirs(parent_dir)

    def read_data(self, file_path: str) -> pd.DataFrame:
        '''Read the data from the file'''
        data = pd.read_csv(file_path, sep='\t')
        return data
    
    def prompt_template(self, query:str, document: str) -> str:
        '''Prompt the user to select the best document'''
        text = f"Query:{query}\nDocument:{document}"
        return text
      
    def make_prompt(self, queries, responses) -> list[str]:
        '''Generate the prompt for the dataset'''
        text = [
            self.prompt_template(str(query), str(response))
            for query, response in zip(queries, responses)
        ]
        return text
    
    def create_dataset(self) -> None:
        '''Create the dataset'''
        # Read the data
        logging.log(logging.INFO, "Reading the data...")
        relevant_docs = self.read_data(self.relevant_docs_path)
        relevant_docs = relevant_docs.sample(n=const.NUMBER_OF_INSTANCE_PER_CLASS, random_state=const.RANDOM_STATE)

        non_relevant_docs = self.read_data(self.non_relevant_docs_path)
        non_relevant_docs = non_relevant_docs.sample(n=const.NUMBER_OF_INSTANCE_PER_CLASS, random_state=const.RANDOM_STATE)

        assert 'query' and 'doc' in relevant_docs.columns, "Data does not have the required 'query' and 'doc' columns..."
        assert 'query' and 'doc' in non_relevant_docs.columns, "Data does not have the required 'query' and 'doc' columns..."

        # Generate the prompt
        logging.log(logging.INFO, "Generating the prompt...")
        prompts_1: str = self.make_prompt(relevant_docs['query'], relevant_docs['doc'])
        prompts_2: str = self.make_prompt(non_relevant_docs['query'], non_relevant_docs['doc'])

        # Create the dataframe
        logging.log(logging.INFO, "Creating the dataframe...")
        df1 = pd.DataFrame({
            "text": prompts_1,
            "label": [1]*len(prompts_1)
        })

        df2 = pd.DataFrame({
            "text": prompts_2,
            "label": [0]*len(prompts_2)
        })

        # Concatenate the dataframes
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

        logging.log(logging.INFO, "Dataset with 0 lables: %s", len(df[df['label']==0]))
        logging.log(logging.INFO, "Dataset with 1 lables: %s", len(df[df['label']==1]))

        # Save the data
        logging.log(logging.INFO, "Saving the dataset...")
        df.to_csv(self.save_path, sep='\t', index=None)
        logging.log(logging.INFO, "Dataset saved at %s", self.save_path)

if __name__ == "__main__":
    logging.log(logging.INFO, "Creating the dataset...")
    try:
        formatter = CreateDataset(const.RELEVANT_DOCS_PATH, const.NON_RELEVANT_DOCS_PATH, const.SAVE_PATH)
        formatter.create_dataset()
    except Exception as e:      # pylint: disable = broad-exception-caught
        logging.log(logging.ERROR, "An error occurred: %s", e)