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

# setting up
def model_inputs(targets, inputs, comps, lexicon, vecs):
    c_ws = [vecs[w] for w in lexicon]
    w2idx = {w: i for i, w in enumerate(lexicon)}

    cs = []
    for i, w in enumerate(tqdm(targets)):
        cs.append(vecs[w])

    h_family = {h: [] for h in lexicon}
    m_family = {m: [] for m in lexicon}
    for i, w in enumerate(tqdm(inputs)):
        h, m = comps[w]
        h_family[h].append(w)
        m_family[m].append(w)

    return np.array(c_ws), np.array(cs), h_family, m_family, w2idx

# setup p(c | m, h, D)
def set_baseline(params):
    lang = params['lang']
    return params

def set_head_only(params, lexicon, h_family, comps, vecs, gamma, theta, diff_norm, schema_norm, device):
    p_z_lst = []
    mu_lst = []
    for i, w in enumerate(lexicon):
        # composition
        mu_h_lst = [vecs[w]]
        p_z_lst_h = [theta]

        # family-level schema
        if len(h_family[w]) > 0:
            diffs = [(1-gamma)*vecs[c] - gamma*vecs[comps[c][1]] for c in h_family[w]]
            # norm each difference
            if diff_norm:
                diffs = [diff / np.linalg.norm(diff) for diff in diffs]

            mu_h = np.mean(diffs, axis=0)
            # norm the vector
            if schema_norm:
                mu_h /= np.linalg.norm(mu_h)
            mu_h_lst.append(mu_h)
            p_z_lst_h.append(len(diffs))

        mu_h_lst = [torch.tensor(v, device=device, dtype=torch.double, requires_grad=False) for v in mu_h_lst]
        mu_lst.append(mu_h_lst)
        p_z_lst.append(p_z_lst_h)

    params['p_z_lst'] = p_z_lst
    params['mu_lst'] = mu_lst

    lang = params['lang']

    return params

def set_sigma2(model_type, params, targets, cs, c_ws, comps, w2idx, lbd, d, device):
    mu_lst = params['mu_lst'] if 'mu_lst' in params else []
    p_z_lst = params['p_z_lst'] if 'p_z_lst' in params else []

    sigma2_range = np.arange(0.001, 0.101, 0.001)
    best_sigma2 = 0
    best_nll = np.inf
    for sigma2 in sigma2_range:
        nll = 0

        for i, w in enumerate(targets):
            h, m = comps[w]
            j, m_idx = w2idx[h], w2idx[m]

            if model_type == 0:
                comp_array = lbd * c_ws[j] + (1 - lbd) * c_ws[m_idx]
                comp_array = torch.nn.functional.normalize(comp_array, p=2, dim=0)

                dot_prod = compute_cosine(comp_array, cs[i])
                nll += exp_cosine(dot_prod, sigma2)

            elif model_type in {1, 2, 3}:
                comp_array = []
                p_z = []
                for k, mu_h_j in enumerate(mu_lst[j]):
                    comp_array.append(lbd * mu_h_j + (1 - lbd) * c_ws[m_idx])
                    p_z.append(p_z_lst[j][k])
                comp_array = torch.stack(comp_array)
                p_z = torch.tensor(p_z, device=device, dtype=torch.double, requires_grad=False)
                p_z /= torch.sum(p_z)

                comp_array = torch.nn.functional.normalize(comp_array, p=2, dim=-1)
                dot_prod = compute_cosine(comp_array, cs[i])

                nll += -torch.logsumexp(torch.log(p_z) + dot_prod / sigma2, dim=0)

        nll = vMF_const(d, 1/sigma2) + nll/len(targets)
        print(f'sigma2 = {sigma2}, avg nll = {nll}')

        if nll < best_nll:
            best_nll = nll
            best_sigma2 = sigma2

    print(f'best {best_sigma2}')
    print('--------------------')
    return best_sigma2

# (negative) log density functions
def compute_cosine(mu, c):
    return torch.sum(mu * c, axis=-1)

def exp_cosine(dot_prod, sigma2):
    return -1/sigma2 * dot_prod

def vMF_const(d, tau):
    return d/2*np.log(2 * np.pi) - (d/2 - 1)*np.log(tau) + np.log(iv(d/2-1, tau))

# eval
def mrr(idxes):
    idxes = np.array(idxes)
    return np.mean(1 / (idxes + 1))

def select_topk_comps(nll_array, k):
    k_m = min(k, nll_array.size()[1])
    top = torch.topk(nll_array, k=k_m, dim=-1, largest=False)
    top2 = torch.topk(torch.flatten(top.values), k=k, largest=False)

    top_m = torch.flatten(top.indices)[top2.indices]
    top_h = top2.indices // k_m
    return torch.stack((top_h, top_m), dim=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=int, help="0 = comp-only, 1 = one schema, 2 = exemplar, \
                                                        3 = custom schemas, 10 = prior only", default=0)
    parser.add_argument("--num_top", type=int, help="number of top retrievals to record", default=5)
    parser.add_argument("--diff_norm", type=int, help="0 = no norm, 1 = normalized", default=0)
    parser.add_argument("--schema_norm", type=int, help="0 = no norm, 1 = normalized", default=1)

    parser.add_argument("--smooth_min", type=int, help="minimum value for pseudocount", default=0.01)
    parser.add_argument("--smooth_max", type=int, help="maximum value for pseudocount", default=100)

    parser.add_argument("--lbd", type=float, help="weight on head word", default=0.5)
    parser.add_argument("--gamma", type=float, nargs='*', help="weight on modifier in vector difference", default=[0.5])
    parser.add_argument("--theta", type=float, nargs='*', help="weight of composition", default=[1.])

    parser.add_argument("--total_splits", type=int, help="number of splits", default=5)
    parser.add_argument("--test_split", type=int, help="use the n-th split as test set", default=1)
    parser.add_argument("--val_split", type=int, help="use the n-th split as validation set", default=1)
    parser.add_argument("--test", type=int, help="0 = val, 1 = test", default=0)

    parser.add_argument("--gpu", type=int, help="gpu", default=0)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    # fixed seed
    seed = 0
    num_splits = args.total_splits
    test_split = args.test_split
    val_split = args.val_split
    testing = args.test

    langs = ['eng', 'chi', 'ger']
    model_type = args.model_type
    num_top = args.num_top
    diff_norm = args.diff_norm
    schema_norm = args.schema_norm

    lbd = args.lbd
    gamma_lst = args.gamma # 1 per language
    theta_lst = args.theta # 1 per language

    pseudocount_range = 10 ** np.arange(np.log10(args.smooth_min), np.log10(args.smooth_max)+1, 1)
    alpha_range =  pseudocount_range # head pseudocount
    beta_range = pseudocount_range # modifier pseudocount

    eng_word2vec_path = "../data/enwiki_20180420_nolg_300d.txt"
    ger_word2vec_path = "../data/dewiki_20180420_300d.txt" # TODO
    chi_word2vec_path = "../data/sgns.chi_wiki.word"

    vectors_cache = './cache/combination_w2v_vecs_{lang}_1.pkl'
    output_csv = './outputs/splits/combination_pred_{lang}_{schema_norm}_{diff_norm}_{model}_{alpha}_{beta}_' + \
                 '{lbd}_{gamma}_{theta}_{num_splits}_{testing}_{test_split}_{val_split}_{seed}.csv' # TODO: change folder when no prior

    ger2eng_path = '../data/ger_translations.pkl'

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

    # get lexicons; just use all constituents for now
    lexicons = {}
    for lang in langs:
        lexicons[lang] = set()
        for p1, p2 in compounds[lang].values():
            lexicons[lang].add(p1)
            lexicons[lang].add(p2)
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

        targets = splits[test_split] if testing == 1 else splits[val_split]
        inputs = np.concatenate([x for i, x in enumerate(splits) if i != test_split and i != val_split])
        print(f'num target compounds = {len(targets)}, num existing compounds = {len(inputs)}')


        # set up model inputs
        print('setting up model inputs...')
        c_ws, cs, h_family, m_family, w2idx = model_inputs(targets, inputs, comps, lexicon, vecs)
        c_ws = torch.tensor(c_ws, device=device, dtype=torch.double, requires_grad=False)
        cs = torch.tensor(cs, device=device, dtype=torch.double, requires_grad=False)
        N_h = torch.tensor([len(h_family[w]) for w in lexicon], device=device, dtype=torch.double, requires_grad=False)
        N_m = torch.tensor([len(m_family[w]) for w in lexicon], device=device, dtype=torch.double, requires_grad=False)
        K = len(lexicon)

        params = {
            'lang': lang,
        } # params for p(c | m, h, D)
        if model_type == 0:
            params = set_baseline(params)
        elif model_type == 1:
            params = set_head_only(params, lexicon, h_family, comps, vecs, gamma_lst[l], theta_lst[l], \
                                   True if diff_norm == 1 else False, True if schema_norm == 1 else False, device)
        else:
            pass
        params['sigma2'] = set_sigma2(model_type, params, splits[val_split], cs, c_ws, comps, w2idx, lbd, d, device)

        sigma2 = 1
        if 'sigma2' in params:
            sigma2 = torch.tensor(params['sigma2'], device=device, dtype=torch.double, requires_grad=False)
        p_z_lst, mu_lst = [], []
        if 'mu_lst' in params and 'p_z_lst' in params:
            mu_lst = params['mu_lst']
            p_z_lst = params['p_z_lst']

        # compute ranks
        print(f'compute ranks for attetsed compounds...')
        idxes = {(alpha, beta): [] for alpha, beta in product(alpha_range, beta_range)} 
        top_retrievals = {(alpha, beta): [] for alpha, beta in product(alpha_range, beta_range)}

        for i, w in enumerate(targets):
            h, m = comps[w]

            # p(c |m, h, D)
            nll_st = torch.zeros([K, K], device=device, dtype=torch.double, requires_grad=False) # head i, modifier j
            if model_type == 0:
                for j, c_j in enumerate(c_ws):
                    comp_array = lbd * c_j + (1 - lbd) * c_ws
                    comp_array = torch.nn.functional.normalize(comp_array, p=2, dim=-1)

                    dot_prod = compute_cosine(comp_array, cs[i])
                    nll_c = exp_cosine(dot_prod, sigma2)
                    nll_st[j] += nll_c

            elif model_type in {1, 2, 3}:
                for j, mu_h_lst in enumerate(mu_lst):
                    comp_array = []
                    p_z = []
                    for k, mu_h_j in enumerate(mu_h_lst):
                        comp_array.append(lbd * mu_h_j + (1 - lbd) * c_ws)
                        p_z.append(p_z_lst[j][k])
                    comp_array = torch.stack(comp_array) # num clusters * lexicon
                    p_z = torch.tensor(p_z, device=device, dtype=torch.double, requires_grad=False)
                    p_z /= torch.sum(p_z)

                    comp_array = torch.nn.functional.normalize(comp_array, p=2, dim=-1)
                    dot_prod = compute_cosine(comp_array, cs[i])

                    nll_c = -torch.logsumexp(torch.log(p_z.reshape(-1,1)) + dot_prod / sigma2, dim=0)
                    nll_st[j] += nll_c
                
            
            # p(m, h | D)
            for alpha, beta in product(alpha_range, beta_range): 
                nll_h = -torch.log(alpha + N_h)
                nll_m = -torch.log(beta + N_m)
                nll_array = ((nll_st + nll_m).T + nll_h).T

                # get rank of attested compound
                attested_nll = nll_array[w2idx[h]][w2idx[m]].item()
                rank = torch.sum(nll_array < attested_nll) + (torch.sum(nll_array == attested_nll)-1)/2 # use fractional rank
                idxes[(alpha, beta)].append(rank.cpu().item())

                print(targets[i], alpha, beta, idxes[(alpha, beta)][-1]/K**2, torch.min(nll_array).item(), attested_nll)

                # get top retrievals
                top_rets_w = select_topk_comps(nll_array, num_top)
                top_rets_w = [(lexicon[h.item()], lexicon[m.item()]) for h, m in top_rets_w] # head-modifier
                top_retrievals[(alpha, beta)].append(top_rets_w)
                # print(top_rets_w)

        # print output
        print('printing output...')
        best_choice, best_mrr = ((-1, -1)), 0 
        for alpha, beta in product(alpha_range, beta_range):
            idxes_ab = idxes[(alpha, beta)]
            top_retrievals_ab = top_retrievals[(alpha, beta)]
            mrr_cur = mrr(idxes_ab)
            print(f'alpha={alpha}, beta={beta}, mrr = {mrr_cur}')

            if mrr_cur > best_mrr:
                best_choice = (alpha, beta)
                best_mrr = mrr_cur

            output_csv_name = output_csv.format(lang=lang,model=model_type,schema_norm=schema_norm,diff_norm=diff_norm, \
                                                alpha=alpha,beta=beta,lbd=lbd,gamma=gamma_lst[l],theta=theta_lst[l],\
                                                num_splits=num_splits,testing=testing,test_split=test_split,val_split=val_split,seed=seed)
            with open(output_csv_name, 'w') as f:
                f.write('compound,head,modifier,head family size,rank,top retrievals\n')
                for i, w in enumerate(targets):
                    h, m = comps[w]
                    top_retrievals_w = ', '.join(['-'.join(x) for x in top_retrievals_ab[i]])
                    f.write(f'{w},{h},{m},{len(h_family[h])},{idxes_ab[i]},\"{top_retrievals_w}\"\n')

        print(f'best alpha, beta = {best_choice[0]}, {best_choice[1]}; mrr = {best_mrr}')

