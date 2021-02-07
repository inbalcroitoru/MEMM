CWORD = 0
CTAG = 1
PWORD = 2
PTAG = 3
NWORD = 4
PPTAG = 5

import pickle
from collections import OrderedDict
import numpy as np
from collections import defaultdict
import time
import re


# load model data- features dictionaries, trained weights and possible tags list
class generate_comp():
    def __init__(self):
        self.model_weights = []
        self.possible_tags = []
        self.words_tags_dict = OrderedDict()  # f100
        self.suffixes_dict = OrderedDict()  # f101
        self.prefixes_dict = OrderedDict()  # f102
        self.trigram_dict = OrderedDict()  # f103
        self.bigram_dict = OrderedDict()  # f104
        self.unigram_dict = OrderedDict()  # f105
        self.prev_word_curr_tag_dict = OrderedDict()  # f106
        self.next_word_curr_tag_dict = OrderedDict()  # f107
        self.is_number_dict = OrderedDict()  # capture words that are numbers
        self.first_word_capitalized_dict = OrderedDict()  # capture first words in sentence that contain capital letters
        self.mid_word_capitalized_dict = OrderedDict()  # capture mid- sentence words that contain capital letters
        self.entire_word_capitalized_dict = OrderedDict()  # capture words that are all capitalized
        # relevant to model 1 only
        self.is_company_name_dict = OrderedDict()  # capture words that are company names
        # relevant to model 2 only
        self.capital_number_dict = OrderedDict()  # capture words that are all capitalized and contain number\s
        self.contain_hyphen_dict = OrderedDict()  # capture words that contain hyphen

    def load_files(self, model_num):
        """
        Load all files that are relevant for inference, including weights, possible tags list and features dictionaries
        :param model_num: num of model- 1 or 2
        :return: no return value
        """
        weights_path = r'Code_Directory\model_{}\model_weights.pkl'.format(model_num)
        with open(weights_path, 'rb') as f:
            optimal_params = pickle.load(f)
        self.model_weights = optimal_params[0]

        tags_path = r'Code_Directory\model_{}\possible_tags.pkl'.format(model_num)
        with open(tags_path, 'rb') as f:
            self.possible_tags = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\words_tags_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.words_tags_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\suffixes_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.suffixes_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\prefixes_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.prefixes_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\trigram_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.trigram_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\bigram_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.bigram_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\unigram_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.unigram_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\prev_word_curr_tag_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.prev_word_curr_tag_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\next_word_curr_tag_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.next_word_curr_tag_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\is_number_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.is_number_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\first_word_capitalized_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.first_word_capitalized_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\mid_word_capitalized_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.mid_word_capitalized_dict = pickle.load(f)

        dict_path = r'Code_Directory\model_{}\dictionaries\entire_word_capitalized_dict.pkl'.format(model_num)
        with open(dict_path, 'rb') as f:
            self.entire_word_capitalized_dict = pickle.load(f)

        if model_num == 1:
            dict_path = r'Code_Directory\model_{}\dictionaries\is_company_name_dict.pkl'.format(model_num)
            with open(dict_path, 'rb') as f:
                self.is_company_name_dict = pickle.load(f)

        if model_num == 2:
            dict_path = r'Code_Directory\model_{}\dictionaries\capital_number_dict.pkl'.format(model_num)
            with open(dict_path, 'rb') as f:
                self.capital_number_dict = pickle.load(f)

            dict_path = r'Code_Directory\model_{}\dictionaries\contain_hyphen_dict.pkl'.format(model_num)
            with open(dict_path, 'rb') as f:
                self.contain_hyphen_dict = pickle.load(f)

    def represent_input_with_features(self, history, model_num):
        """
            Extract feature vector in per a given history
            :param history: tuple{cword, ctag, pword, ptag, nword, ntag, pptag}
            :param model_num: num of model- 1 or 2
                Return a list with all features that are relevant to the given history
        """
        cword = history[CWORD]
        ctag = history[CTAG]
        pword = history[PWORD]
        ptag = history[PTAG]
        nword = history[NWORD]
        pptag = history[PPTAG]
        features = set()

        if model_num == 2:
            cword_shaped = cword.lower()
        else:  # model_num == 1
            cword_shaped = cword

        if (cword_shaped, ctag) in self.words_tags_dict:
            features.add(self.words_tags_dict[(cword_shaped, ctag)])

        suff_dict = {"suffix(1)": cword_shaped[-1:],
                     "suffix(2)": cword_shaped[-2:], "suffix(3)": cword_shaped[-3:],
                     "suffix(4)": cword_shaped[-4:]}
        pref_dict = {"prefix(1)": cword_shaped[:1],
                     "prefix(2)": cword_shaped[:2], "prefix(3)": cword_shaped[:3],
                     "prefix(4)": cword_shaped[:4]}
        if model_num == 2 and len(cword_shaped) >= 10:
            suff_dict["suffix(5)"] = cword_shaped[-5:]
            suff_dict["suffix(6)"] = cword_shaped[-6:]
            pref_dict["prefix(5)"] = cword_shaped[:5]
            pref_dict["prefix(6)"] = cword_shaped[:6]

        for suffix in suff_dict.values():
            if (suffix, ctag) in self.suffixes_dict:
                features.add(self.suffixes_dict[(suffix, ctag)])
        for prefix in pref_dict.values():
            if (prefix, ctag) in self.prefixes_dict:
                features.add(self.prefixes_dict[(prefix, ctag)])

        if (cword_shaped, ctag) in self.prev_word_curr_tag_dict:
            features.add(self.prev_word_curr_tag_dict[(cword_shaped, ctag)])

        if (cword_shaped, ctag) in self.next_word_curr_tag_dict:
            features.add(self.next_word_curr_tag_dict[(cword_shaped, ctag)])

        if (pptag, ptag, ctag) in self.trigram_dict:
            features.add(self.trigram_dict[(pptag, ptag, ctag)])

        if (ptag, ctag) in self.bigram_dict:
            features.add(self.bigram_dict[(ptag, ctag)])

        if ctag in self.unigram_dict:
            features.add(self.unigram_dict[ctag])

        if ctag in self.is_number_dict:
            features.add(self.is_number_dict[ctag])

        if ctag in self.first_word_capitalized_dict:
            features.add(self.first_word_capitalized_dict[ctag])

        if ctag in self.mid_word_capitalized_dict:
            features.add(self.mid_word_capitalized_dict[ctag])

        if ctag in self.entire_word_capitalized_dict:
            features.add(self.entire_word_capitalized_dict[ctag])
        if model_num == 1:
            if ctag in self.is_company_name_dict:
                features.add(self.is_company_name_dict[ctag])

        if model_num == 2:
            if ctag in self.capital_number_dict:
                features.add(self.capital_number_dict[ctag])

            if ctag in self.contain_hyphen_dict:
                features.add(self.contain_hyphen_dict[ctag])

        return list(features)

    def get_q_prob(self, weights, history, model_num):
        """
        	Calc q probability part for the Viterbi algo, with respect to a given history
        	:param weights: weights as calculated in training part
        	:param possible_tags: all possible tags as found in training part
        	:param history: tuple{cword, ctag, pword, ptag, nword, ntag, pptag}
        		Return a dictionary with propabilities for the given history
        """
        sum_exp = 0
        v_dict = defaultdict(float)
        for tag in self.possible_tags:
            history[CTAG] = tag
            curr_tag_indexes_list = self.represent_input_with_features(history, model_num)
            curr_exp = np.exp(weights[curr_tag_indexes_list].sum())
            v_dict[tag] = curr_exp
            sum_exp += curr_exp
        q_dict = {v: exp / sum_exp for v, exp in v_dict.items()}

        return q_dict


# ------------------------Viterbi function-----------------#
def memm_viterbi(model_num, sentence, weights, special_words_dict, dics_obj, B):
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
        curr_possible_tags = dics_obj.possible_tags
        # treatment for special words
        if curr_word in special_words_dict.keys():
            curr_possible_tags = special_words_dict[curr_word]
        # check if curr word is digit like- 9, 1.2, 18:30, -90.9 -> if so, v_tag is CD
        check_curr_word = curr_word[1:] if curr_word[0] == '-' else curr_word
        check_curr_word = check_curr_word.replace('.', '').replace(':', '').replace(',', '')
        if check_curr_word.isdigit():
            curr_possible_tags = ['CD']
        if model_num == 1:
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
                        q_dict[(u, t)] = dics_obj.get_q_prob(weights, curr_history, model_num)
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


# ------------------------------Run Code for both models------------------------------#
for model_num in [1, 2]:
    dics_obj = generate_comp()
    dics_obj.load_files(model_num)

    special_words_dict = {'-LRB-': ['-LRB-'], '-RRB-': ['-RRB-'], '#': ['#'], '$': ['$'], '.': ['.'], ',': [','],
                          ':': [':'], ';': [':'],
                          '``': ['``'], "''": ["''"], '-': [':'], '--': [':'], '&': ['CC'], '%': ['NN', 'JJ']}
    if model_num == 1:
        special_words_dict['ago'] = ['IN']
        special_words_dict['CDs'] = ['NNPS']
        special_words_dict['down'] = ['IN']
    if model_num == 2:
        special_words_dict['x'] = ['SYM']

    good_cnt = 0
    word_cnt = 0
    comp_path = r'Code_Directory\model_{}\comp{}.words'.format(model_num, model_num)
    output = []
    with open(comp_path) as test_f:
        for line in test_f:
            sentence = line.rstrip().split(' ')

            viterbi_tags = memm_viterbi(model_num, sentence, dics_obj.model_weights, special_words_dict, dics_obj, B=15)

            sentence_str = ' '.join([word + '_' + tag for word, tag in zip(sentence, viterbi_tags)])
            output.append(sentence_str + '\n')

        output[-1] = output[-1].rstrip()
        with open(r"Code_Directory\comp_m{}_204783351.wtag".format(model_num), 'w') as res_f:
            res_f.writelines(output)

    print("Done generating for model ", model_num)
