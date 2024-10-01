'''
Script to download the MS MARCO dataset files.
Script take two inputs as arguments: 
    1. dataset to download (documents or passages)
    2. data type to download (doc, train, dev, or test)

Note: All the links are stored at location data/msmarco-datset-download-links.json
'''
import os
import subprocess
import logging
import argparse
from constants import DownloadMSMARCOConstants as const

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset(datset_type: str, file_type: str):
    """Download the specified file type from the MS MARCO dataset."""
    data_dir = const.SAVE_DIR
    logging.info("Downloading %s files...", file_type)


    # Check if the file type is valid
    if datset_type not in const().URLS:
        logging.error("Invalid dataset type %s. Must be one of: documents, passages...", datset_type)
        return
    if file_type not in const().URLS[datset_type]:
        logging.error("Invalid file %s. Must be one of: docs, train, test.", file_type)
        return

    # Create the directory if it doesn't exist
    directory = os.path.join(data_dir, datset_type)
    directory = os.path.join(directory, file_type)
    subprocess.run(["mkdir", "-p", directory], check=False)
    logging.info("Download directory is set to %s...", directory)

    # Download the selected file type
    logging.info("Downloading %s files...", file_type)
    urls = const().URLS[datset_type]
    for url in urls[file_type]:
        logging.info("Downloading %s...", url)
        subprocess.run(["wget", url, "-P", directory], check=False)
    logging.info("All files downloaded successfully...")

if __name__ == "__main__":
    # Argument parser to specify the dataset and file type to download
    parser = argparse.ArgumentParser(description="Download MS MARCO dataset files.")
    parser.add_argument('--dataset_type', '-d', choices=['documents', 'passages'], required=True,
                        help="Specify which dataset to download (documents or passages).")
    parser.add_argument('--type', '-f', choices=['doc', 'train', 'dev', 'test'], required=True,
                        help="Specify which dataset type to download (doc, train, dev, or test).")
    
    # Parse the arguments
    args = parser.parse_args()

    # Validate the inputs and start the download process
    download_dataset(args.dataset_type, args.type)

