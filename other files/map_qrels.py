'''Script to map the complete MS MARCO dataset for the re-ranking model.'''
import csv
import gzip
import logging
from random import choice, sample
from constants import MapQrelsConstants as const

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_queries(file_path):
    """Load query strings from the given file path."""
    queries_ = {}
    logging.info("Loading query strings...")
    with gzip.open(file_path, 'rt', encoding='utf8') as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) == 2:
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
            if len(row) == 3:
                doc_id, _, offset = row
                doc_offsets_[doc_id] = int(offset)
    return doc_offsets_

def get_doc_content(doc_id, file, doc_offsets_):
    """Retrieve content for a given doc_id from the filehandle."""
    try:
        offset = doc_offsets_[doc_id]
    except KeyError:
        logging.warning("Document ID %s not found in offsets.", doc_id)
        return None

    file.seek(offset)
    line = file.readline()
    if not line.startswith(doc_id + "\t"):
        logging.error("Expected %s, found %s", doc_id, line)
        return None

    # Get the content of the document
    content = line.split("\t")[-1].strip()
    return content

def map_data(queries_, doc_offsets_, outfile):
    """Map all queries and documents based on qrels and generate the dataset."""
    all_docs = set()
    with gzip.open(const.DOCS_PATH, 'rt', encoding='utf8') as docs_file:
        logging.info("Loading all documents...")
        for line in docs_file:
            doc_id = line.split("\t", 1)[0]
            all_docs.add(doc_id)

    relevant_docs_by_query = {}
    doc_count = 0
    logging.info("Loading relevant query-doc pairs...")
    with gzip.open(const.QRELS_PATH, 'rt', encoding='utf8') as qrels:
        for line in qrels:
            row = line.split()
            if len(row) < 4:
                continue

            topic_id, _, doc_id, relevance = row
            if relevance == "1":  # Relevant documents
                if topic_id not in relevant_docs_by_query:
                    relevant_docs_by_query[topic_id] = set()
                relevant_docs_by_query[topic_id].add(doc_id)
                doc_count += 1
    logging.log(logging.INFO, "Total relevant query-doc pairs: %d...", doc_count)

    with gzip.open(const.DOCS_PATH, 'rt', encoding='utf8') as docs_file, \
         open(outfile, 'w', encoding='utf8') as out:

        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["qid", "docid", "query", "doc", "relevance"])

        logging.info("Mapping data for training...")
        doc_count = 0
        selected_queries = sample(list(relevant_docs_by_query.keys()), min(const.MAX_POSITIVE_DOC_COUNT, const.TOP_K))

        # Generate positive samples
        for topic_id in selected_queries:
            relevant_docs = relevant_docs_by_query[topic_id]
            logging.info("Mapping a positive doc for query %s...", doc_count)

            for doc_id in relevant_docs:
                if doc_count >= const.MAX_POSITIVE_DOC_COUNT:
                    break
                query = queries_.get(topic_id, "Query not found")
                doc_content = get_doc_content(doc_id, docs_file, doc_offsets_)
                if doc_content:
                    writer.writerow([topic_id, doc_id, query, doc_content, 1])
                    doc_count += 1
            if doc_count >= const.MAX_POSITIVE_DOC_COUNT:
                break

        # Generate negative samples
        remaining_queries = set(relevant_docs_by_query.keys()) - set(selected_queries)
        negative_count = 0
        num_samples = min(len(remaining_queries), const.TOP_K)

        for topic_id in sample(sorted(remaining_queries), num_samples):
            logging.info("Mapping a negative doc for query %s...", negative_count)
            relevant_docs = relevant_docs_by_query[topic_id]
            available_docs = list(all_docs - relevant_docs)

            if negative_count >= const.TOP_K:
                break

            if len(available_docs) > 0:
                neg_doc_id = choice(available_docs)
                query = queries_.get(topic_id, "Query not found")
                doc_content = get_doc_content(neg_doc_id, docs_file, doc_offsets_)

                if doc_content:
                    writer.writerow([topic_id, neg_doc_id, query, doc_content, 0])
                    negative_count += 1

        logging.info("Mapping completed with %d positives and %d negatives.", doc_count, negative_count)

if __name__ == "__main__":
    # Load data
    queries = load_queries(const.QUERIES_PATH)
    doc_offsets = load_doc_offsets(const.DOCS_LOOKUP_PATH)

    # Generate dataset
    logging.info("Mapping dataset...")
    map_data(queries, doc_offsets, const.SAVE_PATH)
