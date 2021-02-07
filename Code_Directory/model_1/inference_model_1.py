import pickle

# ------------------------------Load pre-trained files----------------------------#

weights_path = r'training_model_weights.pkl'

with open(weights_path, 'rb') as f:
    optimal_params = pickle.load(f)
pre_trained_weights = optimal_params[0]
pre_trained_weights

dict_path = r'dictionaries\training_words_tags_dict.pkl'
with open(dict_path, 'rb') as f:
    words_tags_dict = pickle.load(f)

dict_path = r'dictionaries\training_suffixes_dict.pkl'
with open(dict_path, 'rb') as f:
    suffixes_dict = pickle.load(f)

dict_path = r'dictionaries\training_prefixes_dict.pkl'
with open(dict_path, 'rb') as f:
    prefixes_dict = pickle.load(f)

dict_path = r'dictionaries\training_trigram_dict.pkl'
with open(dict_path, 'rb') as f:
    trigram_dict = pickle.load(f)

dict_path = r'dictionaries\training_bigram_dict.pkl'
with open(dict_path, 'rb') as f:
    bigram_dict = pickle.load(f)

dict_path = r'dictionaries\training_unigram_dict.pkl'
with open(dict_path, 'rb') as f:
    unigram_dict = pickle.load(f)

dict_path = r'dictionaries\training_prev_word_curr_tag_dict.pkl'
with open(dict_path, 'rb') as f:
    prev_word_curr_tag_dict = pickle.load(f)

dict_path = r'dictionaries\training_next_word_curr_tag_dict.pkl'
with open(dict_path, 'rb') as f:
    next_word_curr_tag_dict = pickle.load(f)

dict_path = r'dictionaries\training_is_number_dict.pkl'
with open(dict_path, 'rb') as f:
    is_number_dict = pickle.load(f)

dict_path = r'dictionaries\training_first_word_capitalized_dict.pkl'
with open(dict_path, 'rb') as f:
    first_word_capitalized_dict = pickle.load(f)

dict_path = r'dictionaries\training_mid_word_capitalized_dict.pkl'
with open(dict_path, 'rb') as f:
    mid_word_capitalized_dict = pickle.load(f)

dict_path = r'dictionaries\training_entire_word_capitalized_dict.pkl'
with open(dict_path, 'rb') as f:
    entire_word_capitalized_dict = pickle.load(f)

dict_path = r'dictionaries\training_is_company_name_dict.pkl'
with open(dict_path, 'rb') as f:
    is_company_name_dict = pickle.load(f)

# --------------------------create features indexes-----------------------#

CWORD = 0
CTAG = 1
PWORD = 2
PTAG = 3
NWORD = 4
PPTAG = 5


class features_indexes_class():

    def __init__(self, dictionaries):

        # Init and fill all features dictionaries
        self.words_tags_dict = dictionaries[0]
        self.trigram_dict = dictionaries[1]
        self.bigram_dict = dictionaries[2]
        self.unigram_dict = dictionaries[3]
        self.prefixes_dict = dictionaries[4]
        self.suffixes_dict = dictionaries[5]
        self.prev_word_curr_tag_dict = dictionaries[6]
        self.next_word_curr_tag_dict = dictionaries[7]
        self.is_number_dict = dictionaries[8]
        self.first_word_capitalized_dict = dictionaries[9]
        self.mid_word_capitalized_dict = dictionaries[10]
        self.entire_word_capitalized_dict = dictionaries[11]
        self.is_company_name_dict = dictionaries[12]

    def represent_input_with_features(self, history):
        """
                Extract feature vector in per a given history
                :param history: tuple{cword, ctag, pword, ptag, nword, ntag, pptag}
                    Return a list with all features that are relevant to the given history
            """
        cword = history[CWORD]
        ctag = history[CTAG]
        pword = history[PWORD]
        ptag = history[PTAG]
        nword = history[NWORD]
        pptag = history[PPTAG]
        features = set()

        if (cword, ctag) in self.words_tags_dict:
            features.add(self.words_tags_dict[(cword, ctag)])

        suff_dict = {"suffix(1)": cword[-1:],
                     "suffix(2)": cword[-2:], "suffix(3)": cword[-3:],
                     "suffix(4)": cword[-4:]}
        pref_dict = {"prefix(1)": cword[:1],
                     "prefix(2)": cword[:2], "prefix(3)": cword[:3],
                     "prefix(4)": cword[:4]}
        for suffix in suff_dict.values():
            if (suffix, ctag) in self.suffixes_dict:
                features.add(self.suffixes_dict[(suffix, ctag)])
        for prefix in pref_dict.values():
            if (prefix, ctag) in self.prefixes_dict:
                features.add(self.prefixes_dict[(prefix, ctag)])

        if (pptag, ptag, ctag) in self.trigram_dict:
            features.add(self.trigram_dict[(pptag, ptag, ctag)])

        if (ptag, ctag) in self.bigram_dict:
            features.add(self.bigram_dict[(ptag, ctag)])

        if ctag in self.unigram_dict:
            features.add(self.unigram_dict[ctag])

        if (pword, ctag) in self.prev_word_curr_tag_dict:
            features.add(self.prev_word_curr_tag_dict[(pword, ctag)])

        if (nword, ctag) in self.next_word_curr_tag_dict:
            features.add(self.next_word_curr_tag_dict[(nword, ctag)])

        if ctag in self.is_number_dict:
            features.add(self.is_number_dict[ctag])

        if ctag in self.first_word_capitalized_dict:
            features.add(self.first_word_capitalized_dict[ctag])

        if ctag in self.mid_word_capitalized_dict:
            features.add(self.mid_word_capitalized_dict[ctag])

        if ctag in self.entire_word_capitalized_dict:
            features.add(self.entire_word_capitalized_dict[ctag])

        if ctag in self.is_company_name_dict:
            features.add(self.is_company_name_dict[ctag])

        return list(features)

    def get_q_prob(self, weights, possible_tags, history):
        """
            Calc q probability part for the Viterbi algo, with respect to a given history
            :param weights: weights as calculated in training part
            :param possible_tags: all possible tags as found in training part
            :param history: tuple{cword, ctag, pword, ptag, nword, ntag, pptag}
                Return a dictionary with propabilities for the given history
        """
        sum_exp = 0
        v_dict = defaultdict(float)
        for tag in possible_tags:
            history[CTAG] = tag
            curr_tag_indexes_list = self.represent_input_with_features(history)
            curr_exp = np.exp(weights[curr_tag_indexes_list].sum())
            v_dict[tag] = curr_exp
            sum_exp += curr_exp
        q_dict = {v: exp / sum_exp for v, exp in v_dict.items()}

        return q_dict


# -----------------------------Viterbi algo--------------------------------#

import numpy as np
from collections import defaultdict


def memm_viterbi(sentence, weights, possible_tags, special_words_dict, dics_obj, B):
    """
		Viterbi implementation - using Beam Search to improve runtime
		:param weights: weights as calculated in training part 
		:param possible_tags: all possible tags as found in training part 
		:param special_words_dict: dictionary with special words that requires special Sv 
		:param dics_obj: object that hold all the features dictionaries and calc q probability
		:param B: hyper-parameter for beam search  
			return a list of all inferenced tags for the given sentence
    """
    num_of_words = len(sentence)
    tags_infer = [None] * num_of_words
    len_k = num_of_words
    phi = defaultdict(float)
    bp = defaultdict(float)

    phi[(0, "*", "*")] = 1  # initializaion

    # forward
    for k, curr_word in enumerate(sentence):
        k = k + 1
        phi_tuples = []
        q_dict = defaultdict()
        prev_w = sentence[k - 2] if k > 1 else '*'
        next_w = sentence[k] if k < len_k else 'STOP'
        curr_possible_tags = possible_tags
        # treatment for special words
        if curr_word in special_words_dict.keys():
            curr_possible_tags = special_words_dict[curr_word]
        check_curr_word = curr_word[1:] if curr_word[0] == '-' else curr_word
        check_curr_word = check_curr_word.replace('.', '').replace(':', '').replace(',', '')
        if check_curr_word.isdigit():
            curr_possible_tags = ['CD']
        if curr_word == 'yen':
            curr_possible_tags = ['NN']
            S_u = ['CD']

        for v_index, v in enumerate(curr_possible_tags):
            if k == 1:
                S_u = ['*']
                S_t = {'*': set('*')}
            for u_index, u in enumerate(S_u):
                best_prob = 0
                best_t = 0
                curr_S_t = S_t[u]
                for t_index, t in enumerate(curr_S_t):
                    # calc q probability
                    curr_history = [curr_word, v, prev_w, u, next_w, t]
                    if (u, t) not in q_dict.keys():
                        q_dict[(u, t)] = dics_obj.get_q_prob(weights, possible_tags, curr_history)
                    curr_q = q_dict[(u, t)][v]
                    curr_prob = phi[(k - 1, t, u)] * curr_q

                    if curr_prob > best_prob:
                        best_prob = curr_prob
                        best_t = t
                phi_tuples.append((v, u, best_t, best_prob))

        phi_tuples = sorted(phi_tuples, key=lambda phi_tuples: phi_tuples[3], reverse=True)
        S_u, S_t = set(), defaultdict(set)
        for b, curr_tuple in enumerate(phi_tuples):
            selected_v, selected_u, selected_t, selected_phi = curr_tuple
            if b == B:
                break
            phi[(k, selected_u, selected_v)] = selected_phi
            bp[(k, selected_u, selected_v)] = selected_t
            S_u.add(selected_v)
            S_t[selected_v].add(selected_u)

    # Backtracking
    # bp for t(n-1), t(n)
    best_phi = 0
    best_u, best_v = None, None
    for v_index, v in enumerate(S_u):
        for u_index, u in enumerate(S_t[v]):
            curr_phi = phi[(len_k, u, v)]
            if curr_phi > best_phi:
                best_phi = curr_phi
                best_u = u
                best_v = v
    tags_infer[-1] = best_v

    if len(sentence) > 1:
        tags_infer[-2] = best_u

        # bp for t(1),...,t(n-2)
        for k in np.arange(num_of_words - 2, 0, -1):
            a = bp[(k + 2, tags_infer[k], tags_infer[k + 1])]
            tags_infer[k - 1] = a

    return tags_infer


# ------------------------------------------Run code---------------------------------------#

dics = [words_tags_dict, trigram_dict, bigram_dict, unigram_dict, prefixes_dict, suffixes_dict, prev_word_curr_tag_dict,
        next_word_curr_tag_dict, is_number_dict, first_word_capitalized_dict, mid_word_capitalized_dict,
        entire_word_capitalized_dict, is_company_name_dict]
dics_obj = features_indexes_class(dics)

tags_path = r'training_possible_tags.pkl'
with open(tags_path, 'rb') as f:
    possible_tags = pickle.load(f)  # possible_tags is a list

import time
import re

special_words_dict = {'-LRB-': ['-LRB-'], '-RRB-': ['-RRB-'], '#': ['#'], '$': ['$'], '.': ['.'], ',': [','],
                      ':': [':'], ';': [':'],
                      '``': ['``'], "''": ["''"], '-': [':'], '--': [':'], '&': ['CC'], '%': ['NN', 'JJ'],
                      'ago': ['IN'], 'CDs': ['NNPS'],
                      'down': ['IN']}

# for the conf matrix
str2index = {}
for i, tag in enumerate(possible_tags):
    str2index[tag] = i
res_dict = defaultdict(list)
for tag in possible_tags:
    res_dict[tag] = [0] * len(possible_tags)

iter_count = 0
word_count = 0
good_word_count = 0
test_path = r"test1.wtag"
start = time.time()
with open(test_path) as f:
    for line in f:
        start_iter = time.time()
        tokens_tags = [tt.split('_') for tt in line.rstrip().split(' ')]
        sentence = [w for w, _ in tokens_tags]
        true_tags = [t for _, t in tokens_tags]
        viterbi_tags = memm_viterbi(sentence, pre_trained_weights, possible_tags, special_words_dict, dics_obj, B=10)
        for i in range(len(true_tags)):
            word_count += 1
            if true_tags[i] == viterbi_tags[i]:
                good_word_count += 1

        # for conf matrix
        for true_tag, viterbi_tag in zip(true_tags, viterbi_tags):
            res_dict[true_tag][str2index[viterbi_tag]] += 1

        iter_count += 1
print("total time of inference: ", time.time() - start)
print("accuracy: ", good_word_count / word_count)

# --------------------Confusion matrix-------------------------#
import pandas as pd

# extract top 10 rows and then only relevant columns to the 10 rows
conf_matrix = pd.DataFrame.from_dict(res_dict, orient='index', columns=possible_tags)
conf_matrix['total'] = conf_matrix.sum(axis=1)
conf_matrix['total_mistake'] = df.sum(axis=1)
for index, row in conf_matrix.iterrows():
    row['total_mistake'] -= row[index] + row['total']
    
conf_matrix = conf_matrix.sort_values(by='total_mistake', ascending=False)
conf_matrix = conf_matrix.head(10)
conf_matrix = conf_matrix.loc[:, (conf_matrix != 0).any(axis=0)]
conf_matrix_trans = conf_matrix.T
print("Top 10 mistakes: \n", conf_matrix_trans)

# calc accuracy
df = pd.DataFrame.from_dict(res_dict, orient='index', columns=possible_tags)
df_as_array = df.values
diagonal_sum = df_as_array.diagonal().sum()
total_sum = df[['total']].values.sum()
accuracy = (float(diagonal_sum) / float(total_sum))
print("accuracy from confustion matrix is: ", accuracy)


