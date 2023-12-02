#!/usr/bin/env python
'''
read datafile in t2ranking
'''
import pandas as pd

def read_collection(collecton_file):
    '''
    collection.tsv contains two columns: pid text (with header = 0)
    return: a df with index = pid and a column para
    '''
    collection = pd.read_csv(collecton_file, sep = '\t', header = 0, quoting = 3, dtype={'text': str})
    collection.columns = ['pid', 'para']
    collection = collection.fillna("NA")
    collection.index = collection.pid
    collection.pop('pid')
    return collection

def read_query(query_file):
    '''
    qureies.train(or dev).tsv contains two columns: qid text (with header = 0)
    return: a df with index = qid and a column qry
    '''
    query = pd.read_csv(query_file, sep='\t', header=0, quoting=3)
    query.columns = ['pid', 'qry']
    query = query.fillna("NA")
    query.index = query.pid
    query.pop('pid')
    return query
