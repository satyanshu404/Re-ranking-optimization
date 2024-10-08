'''
Script to map all the judged q-d pairs using qrels file of top-100 file.
The input to this script are:
    - path to the top-100, qrels, queries and docs files
The output of this script is:
    - a file with the query-doc pairs in the ranked for each query
'''
import gzip
import logging
from collections import defaultdict
import pandas as pd
from constants import MapAllMSMARCOConstants as const
from efficient_mapping_msmarco_for_training import load_queries, get_required_doc_ids, load_docs
const = const()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def unique_qids(file_path: str) -> set:
    '''Get unique query ids from the file'''
    qrels = pd.read_csv(file_path, sep=' ', names=['qid', 'Q0', 'docid', 'relevance'])
    return set(qrels['qid'].unique())

def map_docs(save_dir: str):
    """Generate separate files for relevant and non-relevant documents."""
    status = defaultdict(int)
    required_doc_ids = get_required_doc_ids(const.TOP100_PATH, const.SUBSET_SIZE)

    # Load data
    queries = load_queries(const.QUERIES_PATH)
    all_docs = load_docs(const.DOCS_PATH, required_doc_ids)
    # unique_query_ids = unique_qids(const.QRELS_PATH)

    current_qid = None
    queries_count = 0

    with open(save_dir, 'w', encoding="utf8") as save_file, \
         gzip.open(const.TOP100_PATH, 'rt', encoding="utf8") as top100_file:
        
        save_file.write("qid\tdocid\tquery\tdoc\trank\trelevance\n")

        logging.info("Processing lines...")
        for line in top100_file:
            qid, _, doc_id, rank, rel, _ = line.split(" ")
            qid, doc_id, rank, rel = qid.strip(), doc_id.strip(), rank.strip(), rel.strip()

            # if qid not in unique_query_ids:
            #     status['missing_data'] += 1
            #     continue

            # just for logging
            if qid != current_qid:
                current_qid = qid
                queries_count += 1
                logging.info("Processing query %s...", queries_count)
            
            # Check if query and doc_id are present in the data
            if qid not in queries or doc_id not in all_docs:
                status['missing_data'] += 1
                logging.warning("Missing data for query %s and doc %s", qid, doc_id)
                continue

            query = queries.get(qid, "").strip()
            doc = all_docs.get(doc_id, "").strip()

            save_file.write(f"{qid}\t{doc_id}\t{query}\t{doc}\t{rank}\t{rel}\n")
            status['kept'] += 1
    logging.log(logging.INFO, "Files generated successfully...")
    return status

if __name__ == "__main__":
    # Generate files and print statistics
    logging.info("Mapping all the q-d pairs in the top-100 file...")
    stats = map_docs(const.SAVE_PATH)
    logging.info("Statistics:")
    for key, val in stats.items():
        print(f"{key}: {val}")