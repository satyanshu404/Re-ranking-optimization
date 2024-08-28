'''Script to map the complete MS MARCO dataset for the re-ranking model.'''
import csv
import gzip
import logging
from constants import MapAllMSMARCOConstants as const

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_queries(file_path):
    """Load query strings from the given file path."""
    queries_ = {}
    logging.info("Loading query strings...")
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            topic_id, query_string = row
            queries_[topic_id] = query_string
    return queries_

def load_doc_offsets(file_path):
    """Load document offsets from the given file path."""
    doc_offsets_ = {}
    logging.info("Loading document offsets...")
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            doc_id, _, offset = row
            doc_offsets_[doc_id] = int(offset)
    return doc_offsets_

def get_doc_content(doc_id, file, doc_offsets_):
    """Retrieve content for a given doc_id from the filehandle."""
    file.seek(doc_offsets_[doc_id])
    line = file.readline()
    assert line.startswith(doc_id + "\t"), f"Expected {doc_id}, found {line}"
    # Get the content of the document
    content = line.split("\t")[-1].strip()
    return content

def map_data(queries_, doc_offsets_, outfile):
    """Map all queries_ and documents based on qrels and generate the dataset."""

    with gzip.open(const.DOCDEV_TOP100_PATH, 'rt', encoding='utf8') as top100f, \
         gzip.open(const.DOCS_PATH, 'rt', encoding='utf8') as docs_file, \
         open(outfile, 'w', encoding='utf8') as out:

        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["qid", "docid", "query", "doc_content", "rank"])

        logging.info("Mapping data for training...")
        current_topic_id = None
        doc_count:int = 0
        count: int = 0 # Count the number of documents processed

        for line in top100f:
            row = line.split()
            topic_id, _, doc_id, rank, _, _ = row

            if topic_id != current_topic_id:
                current_topic_id = topic_id
                doc_count = 0
                count += 1
                logging.info("Processing query %s...", count)

            if doc_count < 10:
                # Retrieve query string and document content
                logging.info("Processing doc %s...", doc_count+1)
                query = queries_[topic_id]
                doc_content = get_doc_content(doc_id, docs_file, doc_offsets_)

                # Write to output file
                writer.writerow([topic_id, doc_id, query, doc_content, rank])
                doc_count += 1

if __name__ == "__main__":
    # Load data
    queries = load_queries(const.QUERIES_PATH)
    doc_offsets = load_doc_offsets(const.DOCS_LOOKUP_PATH)

    # Generate dataset
    logging.info("Mapping dataset...")
    map_data(queries, doc_offsets, const.SAVE_PATH)