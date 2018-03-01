import collections
from _collections import defaultdict
import pandas as pd
import numpy as np
import spacy 
import csv

nlp = spacy.load('en')

SpecialVocab = collections.namedtuple('SpecialVocab', ['sos', 'eos', 'unknown', 
                                        'padding'])
special_vocab = SpecialVocab(sos='SEQUENCE_START', eos='SEQUENCE_END',
                             unknown="UNK", padding='-PAD-')


def words_to_indices(seq, vocab_dict):
    '''
    @param seq: list of words/tokens
    '''
    word_indices = []
    for w in seq:
        if w in vocab_dict:
            v = vocab_dict.get(w)
            if v is None:
                print("Got None for %s"%w)
            word_indices.append(v)
        else:
            word_indices.append(vocab_dict.get(special_vocab.unknown))
            print("Couldn't find %s"%w)
    return word_indices

def build_vocab(sentsdf, min_freq=1):
    vocab_dict = defaultdict(int)
    for _, row in sentsdf.iterrows():
        for w in row.words:
            vocab_dict[w] += 1
                    
    #drop all words with less than min_freq
    vocabdf = pd.DataFrame({'word': list(vocab_dict.keys()), 
                            'freq': list(vocab_dict.values())})
    vocabdf = vocabdf[vocabdf.freq >= min_freq]
    
    #add special vocab, with padding at index 0
    vocab_ = [special_vocab.padding, special_vocab.unknown, 
              special_vocab.eos, special_vocab.sos] + vocabdf.word.values.tolist() 
    return vocab_

def pad_seq(seq, max_len):
    seq_len = len(seq)
    if len(seq) > max_len: 
        seq = seq[:max_len]
        seq_len = max_len
    else:
        seq = seq+[special_vocab.padding]*(max_len-seq_len) 
    return seq

def process_sequence(df, vocab_dict, max_len):
    '''
    1. Tokenize
    2. Pad
    3. Convert to wordindices
    4. Convert to relative distances from entity1 and entity2 for each word.
    '''
    word_indices = []
    ent1_dists = []
    ent2_dists = []
    for _, row in df.iterrows():
        seq = row.sent
        
        words = row.words
        
        padded_words = [special_vocab.sos] + words + [special_vocab.eos]
        padded_words = pad_seq(padded_words, max_len)
        wi = words_to_indices(padded_words, vocab_dict)
        
        e1_end = int(row.ent_1_end)#seq.index(row.ent_1)+len(row.ent_1) 
        e2_end = int(row.ent_2_end)#seq.index(row.ent_2)+len(row.ent_2)
        newseq = seq[:e1_end].strip()+" entity_1_end "+ seq[e1_end:e2_end].strip() + \
        " entity_2_end "+ seq[e2_end:].strip()

        newseq_words = [tok.text for tok in nlp.tokenizer(newseq)]
        newseq_words = [special_vocab.sos] + newseq_words + [special_vocab.eos]
        i1 = newseq_words.index('entity_1_end') - 1#TODO: Use head of entity-phrase instead of rightmost word
        i2 = newseq_words.index('entity_2_end') - 2
        
        ent1_dist = [i-i1 for i in range(len(padded_words))]
        ent2_dist = [i-i2 for i in range(len(padded_words))]
        
        word_indices.append(wi)
        ent1_dists.append(ent1_dist)
        ent2_dists.append(ent2_dist)        
        
    word_indices, ent1_dists, ent2_dists = \
        np.asarray(word_indices), np.asarray(ent1_dists), np.asarray(ent2_dists)
    
    ent1_dists += max_len
    ent2_dists += max_len
    
    return word_indices, ent1_dists, ent2_dists
    
            
def read_semeval2010_data(filename):
    data = {'rel':[], 'sent': [], 'ent_1':[], 'ent_2':[], 'words':[],
            'ent_1_start':[], 'ent_2_start':[], 'ent_1_end':[], 'ent_2_end':[]}
    etags = ['<e1>', '</e1>', '<e2>', '</e2>']
    with open(filename, 'r') as rf:
        for line in rf:
            _, sent = line.split('\t')
            
            rel = next(rf).strip().upper()
            next(rf) #comment
            next(rf)#blankline
            e1 = sent[sent.index('<e1>')+4:sent.index('</e1>')]
            e2 = sent[sent.index('<e2>')+4:sent.index('</e2>')]
            e1_start = sent.index('<e1>') - 1
            e2_start = sent.index('<e2>') - 1*4 - 1*5 - 1 #compensating for tag, and "
            e1_end = sent.index('</e1>') - 1*4 - 1
            e2_end = sent.index('</e2>') - 2*4 - 1*5 - 1
            
            for tag_ in etags:
                sent = sent.replace(tag_,"")
            sent = sent.strip().lower()[1:-1]
            words = [tok.text for tok in nlp.tokenizer(sent)]
            data['sent'].append(sent)
            data['ent_1'].append(e1)
            data['ent_2'].append(e2)
            data['rel'].append(rel)
            data['words'].append(words)
            data['ent_1_start'].append(e1_start)
            data['ent_1_end'].append(e1_end)
            data['ent_2_start'].append(e2_start)
            data['ent_2_end'].append(e2_end)
    df = pd.DataFrame.from_dict(data)
    return df
            
def read_embeddings(embeddings_path, vocab_, init_scale=0.25, 
                           dtype='float32', random_state=None):
    
    if random_state is None:
        random_state = np.random.RandomState(10)
        
    vocab_vec = pd.read_csv(embeddings_path, header=None, skiprows=[0],
                        sep=' ', index_col=0, quoting=csv.QUOTE_NONE)
    cols = ['col%d'%x for x in range(vocab_vec.shape[1])]
    vocab_vec.columns = cols
    
#     known_words = [w for w in vocab_  if w in vocab_vec.index]
#     known_mat = vocab_vec.ix[known_words,:]
#     known_mat.to_csv(embeddings_path+".aclaug", sep=' ', index_label='word')
    
    print("Vocab Size: %d"%len(vocab_), flush=True)
    unknown_words = [w for w in vocab_  if w not in vocab_vec.index]
    
    print("adding %d unknown words..."%len(unknown_words), flush=True)
    emb_dim = vocab_vec.shape[1]
    rnd_mat = random_state.uniform(-init_scale, init_scale, 
                    size=(len(unknown_words), emb_dim))
    rnd_df = pd.DataFrame(rnd_mat, index=unknown_words, columns=cols)
    vocab_vec = pd.concat([vocab_vec, rnd_df], axis=0)
    embeddings_mat = vocab_vec.ix[vocab_,:]
    return embeddings_mat

    