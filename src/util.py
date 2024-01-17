# -*- coding: utf-8 -*-

import sys, codecs
import numpy as np

import torch
import transformers


def load_sentences(corpus_file, cased=False):
    """
    Parameters:
    ----------
    corpus_file:str

    cased:bool
        whether or not to consider upper/lower cases
        if False, case is ignored when the target phrase is searched
    

    Rreturn:
    -------=
    sentences: list[list[str]]
        words in sentences. That is, note that sentences are split into words
        here.
    """

    sentences = []
    with codecs.open(corpus_file,  'r', 'utf-8', 'ignore') as fp:
        for sentence in fp:
            sentence = sentence.rstrip()
            if cased  == False:
                sentence = sentence.lower()
            tokens = sentence.split(' ')
            sentences.append(tokens)

    return sentences


def load_sentences_with_target_spans(corpus_file, target_phrase, cased=False):
    """
    Parameters:
    ----------
    corpus_file:str

    target_phrase:str
        target phrase to be searched for the corpus
    
    cased: bool 
        if False, case is ignored when the target phrase is searched
    

    Rreturn:
    -------=
    sentences: list[list[str]]
        words in sentences
    
    spans: list[taple]
        list of spans for the target phrase in the sencences
    """

    if cased  == False:
        target_phrase = target_phrase.lower()
    words_of_target_phrase = target_phrase.split(' ')
    sentences = []
    target_spans = []
    with codecs.open(corpus_file, 'r', 'utf-8', 'ignore') as fp:
        for sentence in fp:
            sentence = sentence.rstrip()
            if cased  == False:
                sentence = sentence.lower()
            words = sentence.split(' ')
            spans = cal_spans(words, words_of_target_phrase)
            for s in spans:
                sentences.append(words)
                target_spans.append(s)

    return sentences, target_spans


def cal_spans(words, words_in_target_phrase):
    
    spans = []
    len_of_target_phrase = len(words_in_target_phrase)
    for i in range(len(words)-len_of_target_phrase+1):
        words_ = words[i:i+len_of_target_phrase]
        if words_ == words_in_target_phrase:
            spans.append((i, i+len_of_target_phrase))

    return spans


def tokenize_and_vectorize(vectorizer, tokenizer, batched_sentences, 
                           is_split_into_words=True,
                           device='cpu'):
    """
    To tokenize and vectorize batched sentences. The obtained vectors are all
    normalized so that their norms equal one. This is for the von Mises-Fisher
    distribution.


    Parameters
    ----------
    vectorizer: vectorizer (mostly, BERT-based)

    tokenizer: tokenizer that is compatible with the vectorizer.


    vetcor_size: int
        the dimension of the mean vector (and also all vectors in the space).
        its default is set to 1024 coming from 'bert-large' models


    Returns 
    -------
    token2vecs: {str:list[numpy array]} 
        dict. mapping token to its normalized word vectors
    
    """

    vectorizer.to(device)
    vectorizer.eval()
    token2vecs = {}
    with torch.no_grad():
        for sentences in batched_sentences:
            token_ids, mask_ids, token_idx2subword_idx =\
                to_ids_with_token_idx(tokenizer,
                                      sentences,
                                      device=device)

            output = vectorizer(token_ids, mask_ids)
            last_hidden_state = output.last_hidden_state

            for batch_i, sentence in enumerate(sentences):
                for token_i, subword_i in token_idx2subword_idx[batch_i].items():
                    vec = last_hidden_state[batch_i][subword_i]
                    vec = vec.to('cpu').detach().numpy().copy()
                    # normalizing for vMF distribution
                    vec = vec/np.linalg.norm(vec) 
                    token = sentence[token_i]
                    vecs = token2vecs.get(token, [])
                    vecs.append(vec)
                    token2vecs[token] = vecs

    return token2vecs


def tokenize_and_vectorize_with_indices(vectorizer,
                                       tokenizer,
                                       batched_sentences,
                                       is_split_into_words=True,
                                       device='cpu'):
    """
    To tokenize and vectorize batched sentences. The obtained vectors are all
    normalized so that their norms equal one. This is for the von Mises-Fisher
    distribution.


    Parameters
    ----------
    vectorizer: vectorizer (mostly, BERT-based)

    tokenizer: tokenizer that is compatible with the vectorizer.


    vetcor_size: int
        the dimension of the mean vector (and also all vectors in the space).
        its default is set to 1024 coming from 'bert-large' models


    Returns 
    -------
    token2vecs: {str:list[numpy array]} 
        dict. mapping token to its normalized word vectors
    
    """

    vectorizer.to(device)
    vectorizer.eval()
    token2vecs = {}
    token2indices = {}
    with torch.no_grad():
        n_sents = 0
        for sentences in batched_sentences:
            token_ids, mask_ids, token_idx2subword_idx =\
                to_ids_with_token_idx(tokenizer,
                                      sentences,
                                      device=device)

            output = vectorizer(token_ids, mask_ids)
            last_hidden_state = output.last_hidden_state

            for batch_i, sentence in enumerate(sentences):
                n_sents += 1
                for token_i, subword_i in token_idx2subword_idx[batch_i].items():
                    vec = last_hidden_state[batch_i][subword_i]
                    vec = vec.to('cpu').detach().numpy().copy()
                    # normalizing for vMF distribution
                    vec = vec/np.linalg.norm(vec) 
                    token = sentence[token_i]
                    vecs = token2vecs.get(token, [])
                    vecs.append(vec)
                    token2vecs[token] = vecs
                    indices = token2indices.get(token, [])
                    indices.append((n_sents-1, token_i))
                    token2indices[token] = indices

    return token2vecs, token2indices


def tokenize_and_vectorize_with_spans(batched_sentences, batched_target_spans,
                                      vectorizer, tokenizer,
                                      is_split_into_words=True,
                                      all_subwords=False,
                                      device='cpu'):
        
    target_vecs = []
    vectorizer.to(device)
    vectorizer.eval()
    with torch.no_grad():
        for sentences, target_spans in zip(batched_sentences,
                                           batched_target_spans):

            tokens, token_ids, mask_ids =\
                tokenize_with_ids(tokenizer, sentences, device=device)
            output = vectorizer(token_ids, mask_ids)
            last_hidden_state = output.last_hidden_state

            for batch_i, span in enumerate(target_spans):
                # word_ids shows alignment between original words and subwords
                word_ids = tokens.word_ids(batch_i) 
                start, end = align_original_and_sub_word_spans(span, word_ids)
                v =\
                    [ last_hidden_state[batch_i][j] for j in range(start, end) ]
                if all_subwords == True:
                    v = torch.cat(v)
                else:
                    v = v[0]
                v = v.to('cpu').detach().numpy().copy()
                target_vecs.append(v)

    return target_vecs


def align_original_and_sub_word_spans(span, word_ids):
    start, end = span

    subword_start = word_ids.index(start)
    end_indices = [ i for i, x in enumerate(word_ids) if x == end ]
    subword_end = subword_start + 1
    if len(end_indices) > 0:
        subword_end = end_indices[0]

    return subword_start, subword_end


def to_batches(instances, batch_size=32):
    num_batch = len(instances)//batch_size
    batches =\
        [ instances[n*batch_size:(n+1)*batch_size] for n in range(num_batch) ]

    rest = len(instances) - num_batch*batch_size
    if rest>0:
        batches.append(instances[num_batch*batch_size:num_batch*batch_size+rest])

    return batches



def cal_concentration(mean_norm, vector_size=1024):
    """
    Calculate the concentration parameter kappa of the von Mises-Fisher
    distribution, which is based on the norm of a vector


    Parameters
    ----------
    norm: float ([0, 1])
        norm of the mean vector, which ranges between 0 and 1 (because
        all vectors are normalized to have norm=1 in the von Mises-Fisher
        distribution. This function uses Banerjee et al. (2005)'s 
        approximation.

    vetcor_size: int
        the dimension of the mean vector (and also all vectors in the space).
        its default is set to 1024 coming from 'bert-large' models


    Returns 
    -------
    kappa: float
        the concentration parameter kappa of the von Mises-Fisher distribution
    
    """
    kappa = -1.0

    if mean_norm < 1.0:
        kappa = mean_norm*(float(vector_size) - mean_norm**2)/(1.0 - mean_norm**2)

    return kappa


def tokenize_with_ids(tokenizer, sentences, device='cpu'):
    """
    Parameters
    ----------
    tokenizer: BERT-based tokenizer

    """

    tokens = tokenizer(sentences,
                       return_tensors='pt',
                       is_split_into_words=True,
                       padding=True,
                       truncation=True)

    token_ids = tokens['input_ids'].to(device)
    mask_ids = tokens['attention_mask'].to(device)

    return tokens, token_ids, mask_ids


def to_ids_with_token_idx(tokenizer, sentences, device='cpu'):
    """
    This method returns token indices only for words that are NOT split into
    subwords. The returned token idices follow the BERT-indexing system, meaning
    that they are added by one (for the special token [CLS]) from the original
    index (of the origina sentence).

    Parameters
    ----------
    tokenizer: BERT-based tokenizer

    sentences: list[str]
    batched sentences consiting of tokens
    """

    tokens = tokenizer(sentences,
                       return_tensors='pt',
                       is_split_into_words=True,
                       padding=True,
                       truncation=True)

    token_ids = tokens['input_ids'].to(device)
    mask_ids = tokens['attention_mask'].to(device)

    target_token_idx2subword_idx = [ {} for _ in sentences ]
    for batch_i in range(len(sentences)):
        # mapping between subword index to token index in the original sentence
        subword_idx2token_idx = tokens.word_ids(batch_i)
        token_idx2subword_indices = {}
        for subword_i, token_i in enumerate(subword_idx2token_idx):
            if token_i != None:
                subword_indices = token_idx2subword_indices.get(token_i, [])
                subword_indices.append(subword_i)
                token_idx2subword_indices[token_i] = subword_indices

        for token_i, subword_indices in token_idx2subword_indices.items():
            # targeting only tokens consiting of one sub-word
            if len(subword_indices) == 1:   
                target_token_idx2subword_idx[batch_i][token_i] =\
                    subword_indices[0]

    return token_ids, mask_ids, target_token_idx2subword_idx
