import argparse
import logging
import os
from tqdm import tqdm
import multiprocessing
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from read_data import read_collection, read_query

logger = logging.getLogger(__name__)
fh = logging.FileHandler("t2ranking_bm25.log",encoding="utf-8",mode="a")
formatter = logging.Formatter("%(asctime)s - %(name)s-%(levelname)s %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="hfl/chinese-roberta-wwm-ext",
               help="base model for pretrained tokenizer")
    parser.add_argument("--collection_file", type=str, default="../dataset/t2ranking/collection.tsv")
    parser.add_argument("--train_query", type=str, default="../dataset/t2ranking/queries.train.tsv")
    parser.add_argument("--index_file", type=str, default="./t2ranking_bm25_index.json")
    args = parser.parse_args()
    return args
    
def main():
    args = get_args()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    logger.info("tokenizer loaded.")

    if args.index_file is not None and os.path.exists(args.index_file):
        # load index from index_file
        bm25 = BM25Okapi(logger=logger, index_file=args.index_file)

    else:
        # create index

        # load collection
        collection = read_collection(args.collection_file)
        corpus = collection['para'].tolist()
        docid_list = collection.index.tolist()
        logger.info("collection loaded.")

        # test code 
        # corpus = corpus[:5]
        # docid_list = docid_list[:5]
        # logger.debug(f"corpus:\n{corpus}")
        # logger.debug(f"docid_list:\n{docid_list}")
        tokenized_corpus = list(tqdm(map(tokenizer.tokenize, corpus), total=len(corpus)))


        logger.info("corpus tokenized.")
    
        bm25 = BM25Okapi(tokenized_corpus, docid_list=docid_list, logger=logger)

        if args.index_file is not None:
            bm25.save_index(args.index_file)

    # TODO: test t2ranking bm25
    # TODO: add code to read query file and retrieve relevant docs
    query = "洗衣机离合器打滑"
    tokenized_query = tokenizer.tokenize(query)
    # doc_scores = bm25.get_scores(tokenized_query)
    # logger.debug(doc_scores)
    top_n_docid = bm25.get_top_n_docid(tokenized_query, 100)
    logger.debug(top_n_docid)



if __name__ == "__main__":
    main()