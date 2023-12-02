import argparse
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from read_data import read_collection, read_query

logger = logging.getLogger()
fh = logging.FileHandler("t2ranking_bm25.log",encoding="utf-8",mode="a")
formatter = logging.Formatter("%(asctime)s - %(name)s-%(levelname)s %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type = str, default = "hfl/chinese-roberta-wwm-ext",
               help = "base model for pretrained tokenizer")
    parser.add_argument("--collection_file", type = str, default = "../dataset/t2ranking/collection.tsv")
    parser.add_argument("--train_query", type = str, default = "../dataset/t2ranking/queries.train.tsv")
    args = parser.parse_args()
    return args
    
def main():
    args = get_args()

    # load tokenizer
    toknizer = AutoTokenizer.from_pretrained(args.base_model_name)
    logger.info("tokenizer loaded.")

    # load collection
    collection = read_collection(args.collection_file)
    corpus = collection['para'].tolist()
    docid_list = collection.index.tolist()
    logger.info("collection loaded.")

    # test code 
    corpus = corpus[:5]
    docid_list = docid_list[:5]
    logger.debug(f"corpus:\n{corpus}")
    logger.debug(f"docid_list:\n{docid_list}")
    tokenized_corpus = list(tqdm(map(toknizer.tokenize, corpus), total=len(corpus)))
    logger.info("corpus tokenized.")
    
    bm25 = BM25Okapi(tokenized_corpus, docid_list = docid_list)

    query = "新妈妈们产后如何快速恢复肚子"
    tokenized_query = toknizer.tokenize(query)
    doc_scores = bm25.get_scores(tokenized_query)
    logger.debug(doc_scores)
    top_n = bm25.get_top_n(query, corpus, 5)
    logger.debug(top_n)
    top_n_docid = bm25.get_top_n_docid(query, 100)
    logger.debug(top_n_docid)



if __name__ == "__main__":
    main()