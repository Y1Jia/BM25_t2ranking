#!/usr/bin/env python
import os
import math
import json
import numpy as np
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus=None, tokenizer=None, docid_list=None, logger=None, index_file=None):
        self.corpus_size = 0
        self.avgdl = 0  # average document length
        self.doc_freqs = [] # list of dict {word -> frequency of word (in that doc)}
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        self.docid_list = docid_list
        self.logger = logger

        if index_file is not None:
            assert os.path.exists(index_file)==True, "index_file not exists!"
            self.load_index(index_file)
        else: 
            assert corpus is not None, "corpus is None!"
            if tokenizer:
                corpus = self._tokenize_corpus(corpus)

            nd = self._initialize(corpus)
            self._calc_idf(nd)

    def _initialize(self, corpus):
        ''' get self.doc_freqs, self.doc_len, nd '''
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    def get_top_n_docid(self, query, n=100):
        assert self.corpus_size == len(self.docid_list), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [self.docid_list[i] for i in top_n]
    
    def save_index(self, index_file):
        index = {"corpus_size": self.corpus_size,
                 "avgdl": self.avgdl,
                 "doc_freqs": self.doc_freqs,
                 "idf": self.idf,
                 "doc_len": self.doc_len,
                 "docid_list": self.docid_list}
        
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)
        
        self.logger.info(f"save index to file {index_file}")

    def load_index(self, index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        
        try:
            self.corpus_size = index["corpus_size"]
            self.avgdl = index["avgdl"]
            self.doc_freqs = index["doc_freqs"]
            self.idf = index["idf"]
            self.doc_len = index["doc_len"]
            self.docid_list = index["docid_list"]
        except KeyError as e:
            self.logger.error("index doesn't contain key:", e)
        except Exception as e:
            self.logger.error("an error occour:", e)

        self.logger.info(f"load index from {index_file}")

class BM25Okapi(BM25):
    def __init__(self, corpus=None, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25, docid_list=None, logger=None, index_file=None):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer, docid_list, logger, index_file)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])    # term frequency of query word in each doc
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()


# BM25Adpt and BM25T are a bit more complicated than the previous algorithms here. Here a term-specific k1
# parameter is calculated before scoring is done

# class BM25Adpt(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
#
#
# class BM25T(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
