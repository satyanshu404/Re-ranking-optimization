'''Script to download the MS MARCO dataset'''
import subprocess
import logging
from constants import DownloadMSMARCOConstants as const

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the data directory
DATA_DIR = const.DOWNLOAD_PATH

# Create the directory if it doesn't exist
logging.log(logging.INFO, "Creating directory %s...", DATA_DIR)
subprocess.run(["mkdir", "-p", DATA_DIR], check=False)

# Download each file
logging.log(logging.INFO, "Downloading files...")
for url in const.URLS():
    subprocess.run(["wget", url, "-P", DATA_DIR], check=False)

