'''
Script to read the directory of dataset and creates a new tsv file similar to msmarco dataset.

Input: Directory of the fair dataset
Output: TSV file with columns: qid, query, docid, doc, label, annotation
'''
import json
import logging
import os
import pandas as pd
from constants import FairnessDatasetCreationConstants as const


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO)

class CreateDataset:
    '''Read the directory of the dataset and create a new tsv file'''
    def __init__(self) -> None:
        '''Initialize the class'''
        pass  # pylint: disable = unnecessary-pass
    
    def read_directory(self, directory_path: str) -> list:
        ''' Read the directory '''
        files = os.listdir(directory_path)
        # Remove the .DS_Store file (useless)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        files = [os.path.join(directory_path, file) for file in files]
        return files
    
    def extract_file_name(self, file_name: str) -> str:
        ''' Extract the file name '''        
        if file_name == '':
            raise ValueError("The file name is empty")
        assert '.' in file_name, "Invalid file name"
        if(file_name[-1]=='/'):
            file_name = file_name[:-1]

        return file_name.split('/')[-1].split('.')[0].strip().lower()

    def get_all_files(self, directory_path: str) -> dict[str, list]:
        ''' Get all the files in the directory '''
        logging.info("Getting all the files in the directory...")
        directory = self.read_directory(directory_path)
        folders = []
        for file in directory:
            folders.extend(self.read_directory(file))
        

        files = []
        for folder in folders:
            files.extend(self.read_directory(folder))

        dict_ = {}
        for file in files:
            name = self.extract_file_name(file)
            if name not in dict_:
                dict_[name] = []
            dict_[name].append(file)
        return dict_
    
    def read_json_file(self, file_path: str) -> dict:
        ''' Read the json file '''
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def read_csv_file(self, file_path: str) -> pd.DataFrame:
        ''' Read the csv file '''
        df = pd.read_csv(file_path)
        return df
    
    def get_data(self, qid: int, file_paths: list[str]) -> pd.DataFrame:
        ''' Get the data from the files '''
        # get the json and csv data
        json_data, csv_data = None, None
        for file in file_paths:
            if file.endswith('.json'):
                json_data = self.read_json_file(file)
            elif file.endswith('.csv'):
                csv_data = self.read_csv_file(file)

        # get the query
        query = list(json_data.keys())[0]

        assert 'mapped_annotation' in csv_data.columns, "The csv file should have 'mapped_annotation' column"
        assert 'docid' in csv_data.columns, "The csv file should have 'docid' column"

        # get the docs and annotations
        rows = []
        for docs in json_data[query]:
            # assertions
            assert 'docid' in docs, "The json file must have 'docid'"
            assert 'title' in docs, "=The json file must have 'title'"
            assert 'body' in docs, "The json file must have 'body'"
            
            context = f"Title: {docs['title']}, Context: {docs['body']}"
            row = csv_data[csv_data['docid'] == docs['docid']].iloc[0]
            rel, senti = row['mapped_annotation'].split(',')

            row_dict = {
                "qid": int(qid),
                "docid": row['docid'],
                "query": query,
                "doc": context,
                "relevance": int(rel),
                "annotation": int(senti)
            }
            rows.append(row_dict)
        return pd.DataFrame(rows)
    
    def check_directory(self, directory: str) -> None:
        ''' Check if the directory exists '''
        dir_path = os.path.dirname(directory)
        
        if not os.path.exists(dir_path):
            logging.info("Error 404, %s dir not found...", dir_path)
            logging.info("Created the directory...")
            os.makedirs(dir_path, exist_ok=True)
    
    def create_dataset(self, directory: str, save_path: str) -> None:
        ''' Create the dataset '''
        logging.info("Creating the dataset...")
        logging.info("Reading the directory '%s'...", directory)
        files = self.get_all_files(directory)

        df = pd.DataFrame(columns=['qid', 'docid', 'query', 'doc', 'relevance', 'annotation'])

        logging.log(logging.INFO, "Reading the files from the directory...")
        for idx, (_, value) in enumerate(files.items()):
            data = self.get_data(idx+1, value)
            df = pd.concat([df, data], ignore_index=True)

        logging.info("Saving the dataset to '%s'...", save_path)
        self.check_directory(save_path)
        df.to_csv(save_path, sep='\t', index=False)
        logging.info("Dataset saved successfully!")

class CreateQrel:
    ''' Create the qrel file '''
    def __init__(self) -> None:
        ''' Initialize the class '''
        pass # pylint: disable = unnecessary-pass

    def create_qrel(self, dataset_path: str, save_path: str) -> None:
        ''' Create the qrel file '''
        logging.info("Creating the qrel file...")
        df = pd.read_csv(dataset_path, sep='\t')
        df = df[['qid', 'docid', 'relevance']]
        df = df.rename(columns={'relevance': 'label'})
        df['run'] = 'Q0'
        df = df[['qid', 'run', 'docid', 'label']]
        df['label'] = df['label'].astype('int')
        logging.info("Saving the qrel file to '%s'...", save_path)
        df.to_csv(save_path, sep=' ', index=False, header=False)
        logging.info("Qrel file saved successfully!")
    

if __name__ == "__main__":
    DIRECTORY = const.DIRECTORY_PATH
    SAVE_PATH = const.SAVE_PATH

    dataset = CreateDataset()
    dataset.create_dataset(DIRECTORY, SAVE_PATH)

    if not os.path.exists(const.QREL_PATH):
        logging.info("Error 404, Qrel file not found...")
        qrel = CreateQrel()
        qrel.create_qrel(SAVE_PATH, const.QREL_PATH)

        
        
