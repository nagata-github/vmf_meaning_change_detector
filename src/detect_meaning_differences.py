# -*- coding: utf-8 -*-

import sys, codecs
import argparse
import numpy as np

import torch
import transformers
from transformers import AutoTokenizer
from transformers import BertModel

import util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_corpus')
    parser.add_argument('target_corpus')

    parser.add_argument('-m',
                        '--bert_model',
                        help='Language model to get word vectors',
                        default='bert-large-uncased')

    parser.add_argument('-c',
                        '--cased',
                        help='Use this to consider upper/lower case distinction',
                        action='store_true')

    parser.add_argument('-b', '--batch_size', default=32, type=int)

    parser.add_argument('-f',
                        '--freq_threshold',
                        help='Words whose frequency is more than this value is considered',
                        default=10, type=int)


    parser.add_argument('-a',
                        '--all_sub_words',
                        help='Output all sub-words in a word',
                        action='store_true')
        
    args = parser.parse_args()

    return args


"""
To calculate mean norms of all tokens in the given corpus with their
frequencies.
"""
def cal_mean_norms_with_freqs(vectorizer, tokenizer, sentences, device='cpu'):

    token2vecs = util. tokenize_and_vectorize(vectorizer,
                                              tokenizer,
                                              sentences,
                                              device=device)
    token2freq = {}
    token2mean_norm = {}
    vector_size = None
    for token, vecs in token2vecs.items():
        token2freq[token] = len(vecs)
        sum_vec = np.sum(vecs, axis=0)
        vector_size = sum_vec.size
        mean_norm = np.linalg.norm(sum_vec)/float(len(vecs))
        token2mean_norm[token] = mean_norm.item()

    return token2mean_norm, vector_size, token2freq


def cal_scores(source_token2mean_norm,
              target_token2mean_norm, 
              source_token2freq, 
              target_token2freq,
              source_vector_size=1024,
              target_vector_size=1024,
              freq_threshold=10):

    # Obtaining common vocabulary set 
    target_vocab = set(target_token2mean_norm.keys())
    source_vocab = set(source_token2mean_norm.keys())
    common_vocab = target_vocab & source_vocab

    # Scoring
    token2score = {}
    for token in common_vocab:
        source_freq = source_token2freq[token]
        source_mean_norm = source_token2mean_norm[token]
        target_freq = target_token2freq[token]
        target_mean_norm = target_token2mean_norm[token]
        
        if target_freq <= freq_threshold or source_freq <= freq_threshold:
            continue

        source_kappa =\
            util.cal_concentration(source_token2mean_norm[token],
                                   vector_size=source_vector_size)
        target_kappa =\
            util.cal_concentration(target_token2mean_norm[token],
                                   vector_size=target_vector_size)

        # Score function
        # Note that the smaller the kappa is, the more meanings the token has
        if target_kappa > 0.0 and source_kappa > 0.0:
            score = np.log(target_kappa/source_kappa)
            token2score[token] = score

    return token2score


def output(token2score, source_token2freq, target_token2freq,
           delim='\t', digit=3):

    results = sorted(token2score.items(), key = lambda x: x[1], reverse=True)

    for token, score in results:
        # to exclude middle and end subwords
        score = round(score, digit)
        output = delim.join((token,
                             str(score),
                             str(source_token2freq[token]),
                             str(target_token2freq[token])))

        print(output)


def main():
    """
    This program detects words having wider meanings in the input source
    corpus than in the input target corpus.

        usage: python detect_meaning_differences.py SOURCE_CORPUS TARGET_CORPUS
    
    See 'def parse_args()' for other possible options.
    """

    args = parse_args()

    # Preparing data
    source_sentences = util.load_sentences(args.source_corpus, args.cased)
    target_sentences = util.load_sentences(args.target_corpus, args.cased)
    batched_source_sentences =\
        util.to_batches(source_sentences, batch_size=args.batch_size)
    batched_target_sentences =\
        util.to_batches(target_sentences, batch_size=args.batch_size)


    # Preparing BERT model and tokenizer
    device =\
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vectorizer = BertModel.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Calculating mean vectors with token frequencies
    source_token2mean_norm, source_vector_size, source_token2freq =\
        cal_mean_norms_with_freqs(vectorizer,
                                  tokenizer,
                                  batched_source_sentences,
                                  device=device)

    target_token2mean_norm, target_vector_size, target_token2freq =\
        cal_mean_norms_with_freqs(vectorizer,
                                  tokenizer,
                                  batched_target_sentences,
                                  device=device)

    token2score = cal_scores(source_token2mean_norm,
                             target_token2mean_norm, 
                             source_token2freq, 
                             target_token2freq,
                             source_vector_size=source_vector_size,
                             target_vector_size=source_vector_size,
                             freq_threshold=args.freq_threshold)

    output(token2score, source_token2freq, target_token2freq)

if __name__ == '__main__':
    main()
