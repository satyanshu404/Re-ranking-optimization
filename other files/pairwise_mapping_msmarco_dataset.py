'''
This script will generate a pair of positive and negative documents for each query.
We select 1st docs as positive and a random doc from 100 as negative.
The generated datset is used to train the model.
'''
import csv
import random
import gzip
import logging
from collections import defaultdict
from constants import MapMSMARCOConstants as const

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load query strings
logging.log(logging.INFO, "Loading query strings...")
querystring = {}
with gzip.open(const.QUERIES_PATH, 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for row_ in tsvreader:
        topicid, querystring_of_topicid = row_
        querystring[topicid] = querystring_of_topicid

# Load document offsets
logging.log(logging.INFO, "Loading document offsets...")
docoffset = {}
with gzip.open(const.DOCS_LOOKUP_PATH, 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for row_ in tsvreader:
        docid, _, offset = row_
        docoffset[docid] = int(offset)

# Load positive document IDs
logging.log(logging.INFO, "Loading positive document IDs...")
qrel = {}
with gzip.open(const.DOCTRAIN_QRELS_PATH, 'rt', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter=" ")
    for row_ in tsvreader:
        topicid, _, docid, rel = row_
        if rel == "1":
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]

def getcontent(doc_id, file):
    """Retrieve content for a given doc_id from filehandle file."""
    file.seek(docoffset[doc_id])
    line = file.readline()
    assert line.startswith(doc_id + "\t"), f"Looking for {doc_id}, found {line}"
    return line.rstrip()

def generate_triples(outfile, triples_to_generate):
    """Generate triples comprising Query, Positive, and Random documents."""
    status = defaultdict(int)
    unjudged_rank_to_keep = random.randint(1, 100)
    already_done_a_triple_for_topicid = None
    total_completed_triplets:int = const.TRIPLET_COMPLETED*100     #  total queries * 100 docs
    triples_to_generate -= const.TRIPLET_COMPLETED

    with gzip.open(const.DOCTRAIN_TOP100_PATH, 'rt', encoding='utf8') as top100f, \
         gzip.open(const.DOCS_PATH, 'rt', encoding="utf8") as file, \
         open(outfile, 'w', encoding="utf8") as out:
        
        logging.log(logging.INFO, "Processing line %s...", triples_to_generate)
        for line in top100f:
            # Skip all completed triples
            if total_completed_triplets > 0:
                total_completed_triplets -= 1
                continue

            row = line.split()
            if len(row) < 6:
                continue
            topic_id, _, unjudged_docid, rank, _, _ = row

            if already_done_a_triple_for_topicid == topic_id or int(rank) != unjudged_rank_to_keep:
                status['skipped'] += 1
                continue

            unjudged_rank_to_keep = random.randint(1, 100)
            already_done_a_triple_for_topicid = topic_id

            if topic_id not in querystring or topic_id not in qrel or unjudged_docid not in docoffset:
                status['missing_data'] += 1
                continue

            positive_docid = random.choice(qrel[topic_id])
            if unjudged_docid in qrel[topic_id]:
                status['docid_collision'] += 1
                continue

            status['kept'] += 1
            out.write(f"{topic_id}\t{querystring[topic_id]}\t" +
                      f"{getcontent(positive_docid, file)}\t" +
                      f"{getcontent(unjudged_docid, file)}\n")

            triples_to_generate -= 1
            if triples_to_generate <= 0:
                break
            logging.log(logging.INFO, "Processing line %s...", triples_to_generate)

    return status


if __name__ == "__main__":
    # Generate triples and print statistics
    logging.log(logging.INFO, "Generating triples...")
    stats = generate_triples(const.SAVE_PATH, const.NUMBER_OF_TRIPLES)
    logging.log(logging.INFO, "Statistics:")
    for key, val in stats.items():
        print(f"{key}: {val}")
