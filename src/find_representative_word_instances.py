import sys, codecs
import random
import argparse
import numpy as np

import torch
from transformers import BertModel
from transformers import AutoTokenizer

import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_corpus')
    parser.add_argument('target_corpus')

    parser.add_argument('-t',
                        '--target_phrase',
                        help='Target phrase whose instances are scored')

    parser.add_argument('-m',
                        '--bert_model',
                        help='Language model to get word vectors',
                        default='bert-large-uncased')

    parser.add_argument('-c',
                        '--cased',
                        help='Use this to consider upper/lower case distinction',
                        action='store_true')

    parser.add_argument('-b', '--batch_size', default=32, type=int)

    parser.add_argument('-n',
                        '--topN',
                        help='To show top N results',
                        default=10,
                        type=int)

    parser.add_argument('-a',
                        '--all_subwords',
                        help='To use all subwords in a token for\
                              its word vector; otherwise only first subword',
                        action='store_true')
        
    args = parser.parse_args()

    return args


def cal_scores(source_vecs, target_vecs):

    # normalizing to norm 1
    source_vecs = [ v/np.linalg.norm(v) for v in source_vecs ]
    source_mean_vec = np.mean(source_vecs, axis=0)
    source_mean_norm = np.linalg.norm(source_mean_vec)
    normalized_source_mean_vec = source_mean_vec/source_mean_norm

    target_vecs = [ v/np.linalg.norm(v) for v in target_vecs ]
    target_mean_vec = np.mean(target_vecs, axis=0)
    target_mean_norm = np.linalg.norm(target_mean_vec)
    normalized_target_mean_vec = target_mean_vec/target_mean_norm

    diff_vec =\
        normalized_source_mean_vec/(1.0 - source_mean_norm**2)\
        - normalized_target_mean_vec/(1.0 - target_mean_norm**2)

    scores = []
    for i, v in enumerate(source_vecs):
        score = np.dot(v, diff_vec)
        scores.append(score)

    return scores


def output(results, topN=10, marker='*', delim='\t'):

    index = min(topN, len(results))

    for score, words, span in results[:index]:
        start, end = span
        words = ['[BOS]'] + words + ['[EOS]']
        before = ' '.join(words[:start+1])
        target = marker + ' '.join(words[start+1:end+1]) + marker
        after = ' '.join(words[end+1:])
        output = '\t'.join((str(score), before, target, after))
        print(output)
        

def main():
    """
    This program finds word instances having wider meanings in the input source
    corpus than in the input target corpus.

        usage: python find_representative_word_instances SOURCE_CORPUS TARGET_CORPUS WORD_TYP
    
    See 'def parse_args()' for other possible options.
    """
    args = parse_args()

    # Preparing data
    source_sentences, source_spans =\
        util.load_sentences_with_target_spans(args.source_corpus,
                                              args.target_phrase,
                                              cased=args.cased)
    target_sentences, target_spans =\
        util.load_sentences_with_target_spans(args.target_corpus,
                                              args.target_phrase,
                                              cased=args.cased)

    if len(source_spans)<1 or len(target_spans)<1:
        exit(0)


    batched_source_sentences = util.to_batches(source_sentences,
                                               batch_size=args.batch_size)
    batched_source_spans = util.to_batches(source_spans,
                                           batch_size=args.batch_size)
    batched_target_sentences = util.to_batches(target_sentences,
                                               batch_size=args.batch_size)
    batched_target_spans = util.to_batches(target_spans,
                                           batch_size=args.batch_size)


    # Preparing BERT model and tokenizer
    device =\
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vectorizer = BertModel.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # Obtaining word vectors for target phrase
    source_vecs =\
        util.tokenize_and_vectorize_with_spans(batched_source_sentences,
                                               batched_source_spans,
                                               vectorizer,
                                               tokenizer,
                                               all_subwords=args.all_subwords,
                                               device=device)
    target_vecs =\
        util.tokenize_and_vectorize_with_spans(batched_target_sentences,
                                               batched_target_spans,
                                               vectorizer,
                                               tokenizer,
                                               all_subwords=args.all_subwords,
                                               device=device)

    # Calculating scores
    scores = cal_scores(source_vecs, target_vecs)

    # For output
    results =\
        [ (s, source_sentences[i], source_spans[i]) for i, s in enumerate(scores) ]
    results = sorted(results, key = lambda x: x[0], reverse=True)

    output(results, topN=args.topN)
    

if __name__ == '__main__':
    main()
