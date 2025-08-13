import os
import pickle
import argparse
from collections import defaultdict
from itertools import product

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import iv

from helpers import *
from combination_pred_split import model_inputs, set_head_only, compute_cosine

def get_knn(lexicon, vs, target, k):
    sim = compute_cosine(vs, target)
    _, knn = torch.topk(sim, k, largest=True, sorted=True)
    return [lexicon[i] for i in knn.tolist()]

def write_csv(rows, fn):
    with open(fn, 'w') as f:
        rows = [','.join([str(x) for x in row]) for row in rows]
        f.write('\n'.join(rows))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_norm", type=int, help="0 = no norm, 1 = normalized", default=0)
    parser.add_argument("--schema_norm", type=int, help="0 = no norm, 1 = normalized", default=1)
    parser.add_argument("--gamma", type=float, nargs='*', help="weight on modifier in vector difference", default=[0.5])

    parser.add_argument("--total_splits", type=int, help="number of splits", default=5)
    parser.add_argument("--test_split", type=int, help="use the n-th split as test set", default=1)
    parser.add_argument("--val_split", type=int, help="use the n-th split as validation set", default=1)

    parser.add_argument("--gpu", type=int, help="gpu", default=0)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    # fixed seed
    seed = 0
    num_splits = args.total_splits
    test_split = args.test_split
    val_split = args.val_split

    langs = ['eng',] # ['eng', 'chi', 'ger']
    diff_norm = args.diff_norm
    schema_norm = args.schema_norm

    gamma_lst = args.gamma # 1 per language

    eng_word2vec_path = "../data/enwiki_20180420_nolg_300d.txt"
    ger_word2vec_path = "../data/dewiki_20180420_300d.txt" # TODO
    chi_word2vec_path = "../data/sgns.chi_wiki.word"

    vectors_cache = './cache/subpattern_w2v_vecs_{lang}_1.pkl'
    ger2eng_path = '../data/ger_translations.pkl'

    target_dist_csv = "./outputs/subpatterns/target_dist_{lang}_{num_splits}_{test_split}_{val_split}_{seed}.csv"
    head_dist_csv = "./outputs/subpatterns/head_dist_{lang}_{num_splits}_{test_split}_{val_split}_{seed}.csv"

    # get compounds
    compounds = {}
    for lang in langs:
        comps_lang = {}      
        if lang == 'eng':
            comps_lang, _ = pickle.load(open('../data/eng_compounds.pkl', 'rb'))
            comps_lang = {w: (p[1], p[0]) for w, p in comps_lang.items()} # reverse so that head first

            comps_lang = {w: p for w, p in comps_lang.items() if not is_word_informal_wn(w, 'eng') and is_wn_lemma(w, 'eng')}
            comps_lang = filter_stopwords(comps_lang, 'eng')
            comps_lang = filter_chemical(comps_lang, 'eng') # remove chemicals since they belong to their own naming system

        elif lang == 'ger':
            comps_lang, _ = pickle.load(open('../data/ger_compounds.pkl', 'rb'))
            comps_lang = {w: (p[1], p[0]) for w, p in comps_lang.items()} # reverse so that head first
            comps_lang = filter_stopwords(comps_lang, 'ger')
            comps_lang = filter_chemical_via_translation(comps_lang, pickle.load(open(ger2eng_path, 'rb'))) # remove chemical names

        elif lang == 'chi':
            # use two-character words for now
            comps_lang = get_wn_chi_2char()
            comps_lang = {w: p for w, p in comps_lang.items() if not is_word_informal_wn(w, 'cmn')}
            comps_lang = filter_chemical(comps_lang, 'cmn') # remove chemical names

        comps_lang = remove_overlap(comps_lang)
        compounds[lang] = comps_lang
        print('num comps', len(compounds[lang]))

    # get lexicons; use wordnet except for german
    lexicons = {}
    for lang in langs:
        lexicons[lang] = set()
        for p1, p2 in compounds[lang].values():
            lexicons[lang].add(p1)
            lexicons[lang].add(p2)
        if lang == 'eng':
            lexicons[lang] |= {w for w in wn.all_lemma_names(lang='eng') if w.isalpha() and w == w.lower() and len(w) > 1}
        elif lang == 'chi':
            lexicons[lang] |= {w for w in wn.all_lemma_names(lang='cmn') if w.isalpha() and all(is_chi_char(c) for c in w)}
        elif lang == 'ger':
            pass
        print(f'starting lexicon size = {len(lexicons[lang])}')

    # get embeddings
    vectors = {}
    dims = {}
    for lang in langs:
        print(lang)
        vectors_cache_lang = vectors_cache.format(lang=lang, normalized=1)

        if os.path.exists(vectors_cache_lang):
            vectors[lang], dims[lang] = pickle.load(open(vectors_cache_lang, 'rb'))
        else:
            vocab = sorted(set(compounds[lang].keys()) | set(lexicons[lang]))
            normalize = True
            if lang == 'eng':
                vectors[lang], dims[lang] = get_embeddings(eng_word2vec_path, vocab, normalize)
            elif lang == 'ger':
                vectors[lang], dims[lang] = get_embeddings(ger_word2vec_path, vocab, normalize)
            elif lang == 'chi':
                vectors[lang], dims[lang] = get_embeddings(chi_word2vec_path, vocab, normalize)
            pickle.dump((vectors[lang], dims[lang]), open(vectors_cache_lang, 'wb'))

    # compute mrr
    for l, lang in enumerate(langs):
        print(f'{lang}')
        d = dims[lang]
        vecs = vectors[lang]
        comps = compounds[lang]

        # get existing lexicon
        lexicon = {w for w in lexicons[lang] if w in vecs}
        lexicon = sorted(lexicon)
        print(f'existing lexicon size = {len(lexicon)}')

        # split compounds into target and input
        # TODO: read split from file
        comp_words = set(comps.keys()) & set(vecs.keys()) # comp has vector
        comp_words = {w for w in comp_words if comps[w][0] in vecs and comps[w][1] in vecs} # comp parts have vector
        comp_words = sorted(comp_words, key=lambda w: w[::-1])

        N_max = 10000
        if len(comp_words) > N_max:
            # TODO: remove if running time is not a concern
            np.random.seed(seed)
            np.random.shuffle(comp_words)
            comp_words = sorted(comp_words[0:N_max], key=lambda w: w[::-1])
        
        split_size = len(comp_words) // num_splits
        np.random.seed(seed)
        np.random.shuffle(comp_words)
        print(comp_words[0:5])
        splits = [comp_words[i*split_size:(i+1)*split_size] for i in range(num_splits)]

        targets = splits[test_split]
        inputs = np.concatenate([x for i, x in enumerate(splits) if i != test_split and i != val_split])
        print(f'num target compounds = {len(targets)}, num existing compounds = {len(inputs)}')


        # set up model inputs
        print('setting up model inputs...')
        head_subset = sorted({comps[w][0] for w in targets})
        head2idx = {h: i for i, h in enumerate(head_subset)}
        c_ws, cs, h_family, m_family, w2idx = model_inputs(targets, inputs, comps, lexicon, vecs)
        c_ws = torch.tensor(c_ws, device=device, dtype=torch.double, requires_grad=False)
        cs = torch.tensor(cs, device=device, dtype=torch.double, requires_grad=False)

        params = {
            'lang': lang,
        } # params for p(c | m, h, D)
        params = set_head_only(params, head_subset, h_family, comps, vecs, gamma_lst[l], 1, True if diff_norm == 1 else False, \
                                                                                            True if schema_norm == 1 else False, device)
        mu_lst = params['mu_lst']

        # compute dist between compound and head
        data_lst = []
        for i, w in enumerate(targets):
            h, m = comps[w]
            if len(h_family[h]) > 0:
                data_row = []
                data_row.append(w)
                data_row.append(h)
                data_row.append(len(h_family[h]))

                dist_w_c = 1 - torch.dot(mu_lst[head2idx[h]][0], cs[i])
                dist_w_mu = 1 - torch.dot(mu_lst[head2idx[h]][1], cs[i])
                data_row.append(dist_w_c.item())
                data_row.append(dist_w_mu.item())

                data_lst.append(data_row)

        headers = ['compound', 'head', 'head family size', '\"dist(h, c)\"', '\"dist(mu_h, c)\"']
        fn = target_dist_csv.format(lang=lang,num_splits=num_splits,test_split=test_split,val_split=val_split,seed=seed)
        write_csv([headers] + data_lst, fn)

        # compute dist between c_h and mu_h
        data_lst = []
        lexicon_subset = [w for w in lexicon if not w in comps]
        c_ws_subset = c_ws[[i for i, w in enumerate(lexicon) if not w in comps]]
        for i, h in enumerate(head_subset):
            if len(h_family[h]) > 0:
                data_row = []
                data_row.append(h)
                data_row.append(len(h_family[h]))

                dist_c_mu = 1 - torch.dot(mu_lst[i][0], mu_lst[i][1])
                data_row.append(dist_c_mu.item())

                c_knn = get_knn(lexicon_subset, c_ws_subset, mu_lst[i][0], 100)
                mu_knn = get_knn(lexicon_subset, c_ws_subset, mu_lst[i][1], 100)
                data_row.append("\"" + ', '.join(c_knn) + "\"")
                data_row.append("\"" + ', '.join(mu_knn) + "\"")

                data_lst.append(data_row)

        headers = ['head', 'head family size', '\"dist(h, mu_h)\"', 'h neighbours', 'mu_h neighbours']
        fn = head_dist_csv.format(lang=lang,num_splits=num_splits,test_split=test_split,val_split=val_split,seed=seed)
        write_csv([headers] + data_lst, fn)


