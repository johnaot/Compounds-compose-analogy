import re

import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from tqdm import tqdm

stopwords_eng = set(stopwords.words('english'))
stopwords_deu = set(stopwords.words('german'))

# process compounds
def is_chi_char(w):
    chi_regex = u'[\u4e00-\u9fff]'
    return re.search(chi_regex, w)

def is_word_number_wn(w, lang):
    for s in wn.synsets(w, lang=lang):
        if any(l.isdigit() for l in s.lemma_names()):
            return True
    return False

def is_word_informal_wn(w, lang):
    is_slang = lambda defn: ('slang' in defn or 'informal term' in defn or 'street name' in defn)
    return (not 'slang' in w) and any([is_slang(s.definition()) for s in wn.synsets(w, lang=lang)])

def is_wn_lemma(w, lang):
    return len(wn.synsets(w, lang=lang)) > 0

def filter_stopwords(comps, lang):
    stopwords = set()
    if lang == 'eng':
        stopwords = stopwords_eng
    elif lang == 'ger':
        stopwords = stopwords_deu
    return {w: (p[0], p[1]) for w, p in comps.items() if not any(u in stopwords for u in p)}

def filter_chemical(comps, lang):
    organic = wn.synsets('organic_compound')[0]
    chem_group = wn.synsets('group')[1]
    keywords = set()
    for ss in [organic, chem_group]:
        for s in ss.closure(lambda s: s.hyponyms()):
            keywords |= {w for w in s.lemma_names(lang=lang) if w.isalpha()}

    # remove false positives
    if lang == 'eng':
        keywords -= {'tar', 'oil', 'wax', 'fat', 'gas', 'balm', 'lard', 'pulp', 'base', \
                     'crude', 'pitch', 'sugar', 'amber', 'canola', 'gluten', 'grease', 'napalm'}
    elif lang == 'cmn':
        keywords -= {'油', '脂', '火油', '鱼油', '鳕油', '鲸油', '粗糖', '汽油', '兽脂', '树脂', '油脂', '鹅脂', '牛脂' \
                     '木薯', '矿粉', '矿浆', '明胶', '骨胶', '火棉', '烟碱', '糖类', '精油', '琥珀', '菜油', '乳脂' \
                     '多糖', '尸毒', '地蜡', '脂肪', '血糖', '果糖', '煤油', '残油', '骨油', '抗体', '补体', '板油', \
                     '樟脑', '蔗糖', '糊粉', '鲸脂', '渣油', '淀粉',} # TODO: add longer words

    # add WordNet false negatives
    if lang == 'eng':
        keywords |= {'phenyl', 'diol', 'silane', 'quinoline', 'choline', 'thiophene', 'pyrazole', 'corundum', 'indole', \
                     'hydroxy', 'amidite', 'glycerol', 'piperazine', 'adenine', 'cyclopropenone', 'diphenyl', 'diazo', \
                     'geranyl', 'fibrosis', 'myelo', 'peptidyl', 'prolyl', 'butyric', 'naphthoquinone', 'benzo', 'formyl', \
                     'glutathione', 'thiazepine', 'quinoxaline', 'daunorubicin', 'geraniol', 'prostaglandin', 'nitroso', \
                     'guanidine', 'inositol', 'aryl', 'boronic', 'borane', 'catechol', 'butyrate', 'sulfate', 'oxalate', \
                     'pyrazine', 'methoxy', 'pyridine', 'cyano', 'imino', 'peroxy', 'phenoxy', 'ethinyl', 'dihydro', \
                     'estradiol', 'myricetin', 'palmitoyl', 'carnitine', 'fluoro', 'boric', 'lyase', 'diaryl', 'benzoic', \
                     'flavone', 'cholecalciferol', 'olivenite', 'adenosyl', 'lipscombite', 'trichloro'}

    keywords_tuple = tuple(sorted(keywords))
    return {c: (p[0], p[1]) for c, p in comps.items() if (not c in keywords) and \
            (not any(w.startswith(keywords_tuple) or w.endswith(keywords_tuple) for w in p))}

def filter_chemical_via_translation(comps, translations):
    comps_translated = {}
    recorded_comps = {}
    for w, p in comps.items():
        if (w in translations) and (p[0] in translations) and (p[1] in translations):
            w_eng = translations[w].lower()
            p1_eng = translations[p[0]].lower()
            p2_eng = translations[p[1]].lower()
            comps_translated[w_eng] = (p1_eng, p2_eng)
            recorded_comps[w] = w_eng

    comps_translated_filtered = filter_chemical(comps_translated, 'eng')
    # print({w for w in comps if (w in recorded_comps) and (not recorded_comps[w] in comps_translated_filtered)})

    return {w: p for w, p in comps.items() if (not w in recorded_comps) or (recorded_comps[w] in comps_translated_filtered)}

def get_wn_chi_2char():
    comps = {} # head is first
    for w in wn.all_lemma_names(lang='cmn'):
        if len(w) == 2 and is_chi_char(w[0]) and  is_chi_char(w[1]) and \
           not is_word_number_wn(w[0], 'cmn') and not is_word_number_wn(w[1], 'cmn'):
            try:
                if wn.synsets(w, lang='cmn')[0].pos() == 'v':
                    comps[w] = (w[0], w[1]) # assume verbs are left-headed
                else:
                    comps[w] = (w[1], w[0]) # assume others are right-headed
            except:
                print(w)
    return comps

def remove_overlap(comps):
    comps_set = set(comps.keys())
    comps_copy = {}
    for w, p in comps.items():
        w1, w2 = p
        if (not w1 in comps_set) and (not w2 in comps_set):
            comps_copy[w] = p
    return comps_copy

# translation preprocessing
def process_eng_translation(d):
    ret = {}
    for k, v in d.items():
        v2 = ' '.join([w for w in v.lower().split() if not w in stopwords_eng]) # lowercase and remove stopwords
        if len(v2) > 0:
            ret[k] = v2
    return ret

# load vectors
def vec_normalize(v):
    v /= np.linalg.norm(v)
    return v

def read_vectors(p, nomralize=True):
    vecs = dict()
    d = 0
    with open(p, 'r') as f:
        line = f.readline()
        n, d = line.strip().split()
        n, d = int(n), int(d)
        for i in tqdm(range(n)):
            line = f.readline()
            line = line.strip().split()
            w = line[0]
            if len(line[1:]) == d:
                v = np.array([float(x) for x in line[1:]])
                vecs[w] = vec_normalize(v) if nomralize else v
    return vecs, d


def get_embeddings(vectors_path, vocab, nomralize=True):
    vectors_all, d = read_vectors(vectors_path, nomralize)
    vectors = {w: vectors_all[w] for w in vocab if w in vectors_all}
    print(len(vectors), d)
    return vectors, d

# count type frequency
def count_exemplars(compounds, alpha, beta):
    targets = set(compounds.keys())
    c1_count = {c1: alpha for c1, _ in compounds.values()} # assume c1 is head
    c2_count = {c2: beta for _, c2 in compounds.values()}

    for w, p in compounds.items():
        c1, c2 = p
        c1_cat[c1] += 1
        c2_cat[c2] += 1
    return c1_count, c2_count

