'''
This Script maps the top-k q-d pairs from the top-100 docs to the relevant and non-relevant docs
Main Idea:
    - The doc in qrel is consider as relevant
    - The doc in top-100 but not in qrel is consider as non-relevant
The script will generate two files, one for relevant and the other for non-relevant documents.
'''
import csv
import random
import gzip
import logging
from collections import defaultdict
from constants import MapMSMARCOConstants as const

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Dynamic subset size configuration
SUBSET_SIZE = const.SUBSET_SIZE  # Use a constant for dynamic configuration

def load_queries(file_path):
    """Load query strings from the given file path."""
    logging.info("Loading query strings...")
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        return {row[0]: row[1] for row in csv.reader(f, delimiter="\t")}

def load_doc_offsets(file_path, required_doc_ids):
    """Load document offsets only for required document IDs from the given file path."""
    logging.info("Loading document offsets...")
    doc_offsets = {}
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        for row in csv.reader(f, delimiter="\t"):
            doc_id, _, offset = row
            if doc_id in required_doc_ids:
                doc_offsets[doc_id] = int(offset)
    return doc_offsets

def load_positive_doc_ids(file_path):
    """Load positive document IDs from qrels file."""
    logging.info("Loading positive document IDs...")
    qrel = defaultdict(list)
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        for row in csv.reader(f, delimiter=" "):
            topic_id, _, doc_id, rel = row
            if rel == "1":
                qrel[topic_id].append(doc_id)
    return qrel

def get_required_doc_ids(file_path, subset_size):
    """Get a set of all document IDs required from top100 file, limited to subset size."""
    logging.info("Collecting required document IDs...")
    all_required_doc_ids = set()
    current_qid = None
    current_doc_count = 0
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        for line in f:
            row = line.split()
            # Extract the query ID and document ID
            qid = row[0]
            doc_id = row[2]
            
            # Check if we're processing a new query ID
            if qid != current_qid:
                current_qid = qid
                current_doc_count = 0
            
            # If fewer than subset_size documents have been processed for this query, add the doc ID to the set
            if current_doc_count < subset_size:
                all_required_doc_ids.add(doc_id)
                current_doc_count += 1
    return all_required_doc_ids

def load_top_k(file_path, subset_size=SUBSET_SIZE):
    """Load top-k documents for each query from the given file path."""
    logging.info("Loading top-k documents...")
    top_k = defaultdict(list)
    current_qid = None
    current_doc_count = 0
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        for line in f:
            row = line.split()
            qid = row[0]
            doc_id = row[2]
            if qid != current_qid:
                current_qid = qid
                current_doc_count = 0
            if current_doc_count < subset_size:
                top_k[qid].append(doc_id)
                current_doc_count += 1
    return top_k

def load_docs(file_path, required_doc_ids, positive_doc_ids):
    """Load documents only for required and positive document IDs from the given file path."""
    logging.info("Loading required and positive documents...")
    # Combine required and positive doc IDs to create a set of all needed docs
    all_needed_doc_ids = required_doc_ids.union(set(doc_id for docs in positive_doc_ids.values() for doc_id in docs))
    
    docs = {}
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        for line in f:
            doc_id = line.split("\t", 1)[0]
            if doc_id in all_needed_doc_ids:
                content = f"Title: {line.split('\t')[-2]}, Context: {line.split('\t')[-1]}"
                docs[doc_id] = content
    return docs

def generate_files(relevant_outfile, non_relevant_outfile):
    """Generate separate files for relevant and non-relevant documents."""
    status = defaultdict(int)
    required_doc_ids = get_required_doc_ids(const.DOCTRAIN_TOP100_PATH, SUBSET_SIZE)
    positive_doc_ids = load_positive_doc_ids(const.DOCTRAIN_QRELS_PATH)

    # Load data
    queries = load_queries(const.QUERIES_PATH)
    top_k = load_top_k(const.DOCTRAIN_TOP100_PATH)
    all_docs = load_docs(const.DOCS_PATH, required_doc_ids, positive_doc_ids)

    with open(relevant_outfile, 'w', encoding="utf8") as rel_out, \
        open(non_relevant_outfile, 'w', encoding="utf8") as non_rel_out:
        
        rel_out.write("qid\tdocid\tquery\tdoc\n")
        non_rel_out.write("qid\tdocid\tquery\tdoc\n")

        logging.info("Processing lines...")
        for idx, (qid, doc_ids) in enumerate(top_k.items()):
            logging.info("Processing query %s/%s...", idx+1, len(top_k))
            
            if qid not in queries or qid not in positive_doc_ids:
                status['missing_data'] += 1
                continue
            
            # Select positive doc from the qrel file
            positive_doc_id = random.choice(positive_doc_ids[qid])
            if positive_doc_id not in all_docs:
                logging.warning("Document ID %s not found in all_docs for query %s...", positive_doc_id, qid)
                status['missing_docs'] += 1
                continue
            doc = all_docs[positive_doc_id]
            rel_out.write(f"{qid}\t{positive_doc_id}\t{queries[qid]}\t{doc}\n")

            # Remove the positive doc from the list of doc_ids if it's present
            if positive_doc_id in doc_ids:
                doc_ids.remove(positive_doc_id)

            # Filter available negative doc IDs
            available_negative_docs = [doc_id for doc_id in doc_ids if doc_id in all_docs]
            if not available_negative_docs:
                logging.warning("No negative documents available for query %s...", qid)
                status['missing_docs'] += 1
                continue

            negative_doc_id = random.choice(available_negative_docs)
            doc = all_docs[negative_doc_id]
            non_rel_out.write(f"{qid}\t{negative_doc_id}\t{queries[qid]}\t{doc}\n")
            status['kept'] += 1
    logging.log(logging.INFO, "Files generated successfully...")
    return status

if __name__ == "__main__":
    # Generate files and print statistics
    logging.info("Generating separate files for relevant and non-relevant documents...")
    stats = generate_files(const.RELEVANT_SAVE_PATH, const.NON_RELEVANT_SAVE_PATH)
    logging.info("Statistics:")
    for key, val in stats.items():
        print(f"{key}: {val}")
