import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

stopwords_eng = set(stopwords.words('english'))
stopwords_deu = set(stopwords.words('german'))
stopwords_nld = set(stopwords.words('dutch'))

german_char_celex = {
    'ß': 'ss',
    'u' + '̈': 'ue',
    'o' + '̈': 'oe',
    'a' + '̈': 'ae',
}

# wiktionary
def get_affixes(path, lang='eng'):
    prefix = set()
    suffix = set()
    infix = set()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = re.split(r'\s{2,}|\t', line.strip())
            if len(line) < 2:
                continue
            if line[0] != lang: continue
            w = line[1].lower()
            if w[0] == '-' and  w[-1] == '-':
                infix.add(w[1:-1])
            else:
                if w[0] == '-':
                    suffix.add(w[1:])
                if w[-1] == '-':
                    prefix.add(w[0:-1])
    return prefix, suffix, infix

def get_etymology(path, lang='eng', etym_type=None, uppercase_filter=False):
    etym = defaultdict(list)
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = re.split(r'\s{2,}|\t', line.strip())
            if len(line) < 2:
                continue
            if line[0] != lang: continue
            if etym_type != None and line[4] != etym_type: continue
            w = line[1]
            if uppercase_filter and w != w.lower():
                continue
            w = w.lower()
            e = line[-1].split('|')[1:]
            e = [re.sub(r'\W+', '', l) for l in e]
            for i in range(len(e)):
                e_i = [u.lower() for u in e[0:i+1] if len(u) > 1]
                if ''.join(e_i) == w: etym[w].append(e_i)
    return etym

# celex
def read_morph_status(p):
    ret = defaultdict(set)
    with open(p, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\\')
            w = line[1].lower()
            status = line[3]
            ret[w].add(status)
    return ret

def read_immediate_segmentation(p, parse_pos):
    ret = {}
    with open(p, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\\')
            w = line[1].lower()
            status = line[3]
            parse = line[parse_pos]
            if status == 'C':
                ret[w] = parse.split('+')
    return ret

def read_celex_compounds(p, parse_pos, stopwords, prefix, suffix, uppercase_filter=False):
    ret = {}
    with open(p, 'r') as f:
        lines = f.readlines()
        lines = [line.strip().split('\\') for line in lines]
        
        # get possible constituents
        words = {line[1] for line in lines}
        if uppercase_filter:
            words = {w for w in words if w.lower() == w}
        words = {w.lower() for w in words if len(w) > 2}
        words -= set(stopwords)
        words -= set(prefix)
        words -= set(suffix)
        
        print(len(words))
        
        # get segmentation
        for line in lines:
            w = line[1].lower()
            status = line[3]
            parse = line[parse_pos].split('+')
            parse = [u.lower() for u in parse]
            if status == 'C' and len(parse) >= 2 and sum(u in words for u in parse) > 1:
                ret[w] = parse
    return ret

# chinese compound segmentation and clipping
def chi_seg(w_lst, vocab):
    vocab = set(vocab)
    ret = {}
    for w in w_lst:
        if len(w) == 2:
            # TODO: check for loanwords, proper nouns, named entities
            ret[w] = ([w[0], w[1]])
        elif len(w) == 3:
            if w[0:2] in vocab:
                ret[w] = ([w[0:2], w[-1]])
            elif w[1:3] in vocab:
                ret[w] = ([w[0], w[1:3]])
        else:
            for i in range(len(w)-1):
                if w[i+1:] in vocab and w[0:i+1] in vocab:
                    ret[w] = (w[0:i+1], w[i+1:])
                    break
    return ret

def chi_clip_identifier(w_lst, vocab):
    # just focus on 2 x 2 clipped into 2 for now
    vocab = {w for w in vocab if len(w) == 4}
    w_lst = [w for w in w_lst if len(w) == 2]
    ret = {}
    for w in w_lst:
        ret_w = []
        try:
            for ss in wn.synsets(w, lang='cmn'):
                for u in ss.lemma_names(lang='cmn'):
                    if u in vocab and not (w == u[0:2] or w == u[2:]):
                        for i in range(len(u)-1):
                            if w[0] in u[0:i+1] and w[1] in u[i+1:]:
                                ret_w.append(u)
            if len(ret_w) > 0:
                ret[w] = ret_w
        except nltk.corpus.reader.wordnet.WordNetError:
            print(w)
    return ret

# other
def convert_ger_chars(w):
    for k, v in german_char_celex.items():
        w = w.replace(k, v)
    return w

def is_comp(w, stopwords):
    # check if a combination contains a stopword
    p = w.split('-')
    if len(p) >= 2 and all(not w in stopwords for w in p):
        return True
    p = w.split(' ')
    return len(p) >= 2 and all(not w in stopwords for w in p)

def print_top_words(compound_forms, parse, i, k):
    counts = Counter([parse[w][i] for w in compound_forms])
    for p in counts.most_common(k):
        w, c = p
        examples = [u for u in compound_forms if parse[u][i] == w]
        print(p, ', '.join(examples[0:5]) + (', ...' if len(examples) > 5 else ''))
    return counts

def make_rank_plot(axes, words, counts, title):
    lst1 = [np.arange(1,1+len(words)), [counts[w] for w in words]]
    lst2 = [np.log10(np.arange(1,1+len(words))), np.log10([counts[w] for w in words])]
    lsts = [lst1, lst2]
    xlabels = ['rank', 'log rank']
    ylabels = ['family size', 'log family size']
    for i, ax in enumerate(axes):
        ax.plot(lsts[i][0], lsts[i][1])
        ax.set_ylabel(ylabels[i], fontsize=20)
        ax.set_xlabel(xlabels[i], fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title, fontsize=22)
    return axes

