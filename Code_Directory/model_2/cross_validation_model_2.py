
from collections import OrderedDict
from collections import defaultdict
import re

#--------------------------------feature_statistics_class- calc counts for each feature-----------------------#

class feature_statistics_class():

    def __init__(self):
        self.possible_tags = set()   # set of all possible tags from the train set 

        # Init all features dictionaries
        self.words_tags_count_dict = OrderedDict()               #f100
        self.suffixes_count_dict = OrderedDict()                 #f101
        self.prefixes_count_dict = OrderedDict()                 #f102
        self.trigram_count_dict = OrderedDict()                  #f103
        self.bigram_count_dict = OrderedDict()                   #f104
        self.unigram_count_dict = OrderedDict()                  #f105
        self.prev_word_curr_tag_count_dict = OrderedDict()       #f106
        self.next_word_curr_tag_count_dict = OrderedDict()       #f107
        self.is_number_count_dict = OrderedDict()                #capture words that are numbers 
        self.first_word_capitalized_count_dict = OrderedDict()   #capture first words in sentence that contain capital letters  
        self.mid_word_capitalized_count_dict = OrderedDict()     #capture mid- sentence words that contain capital letters  
        self.entire_word_capitalized_count_dict = OrderedDict()  #capture words that are all capitalized   
        self.capital_number_count_dict = OrderedDict()           #capture words that are all capitalized and contain number\s  
        self.contain_hyphen_count_dict = OrderedDict()           #capture words that contain hyphen
        #self.is_company_name_count_dict = OrderedDict()          #capture words that are company names 
        

    def get_word_tag_pair_count(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with counter of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)    
                for word_idx in range(len(splited_words)):
                    if splited_words[word_idx] == '': #remove word that is an empty string 
                        del splited_words[word_idx]
                        continue
                    curr_word, curr_tag = splited_words[word_idx].split("_")     
                    curr_word = curr_word.lower()
                    if (curr_word, curr_tag) not in self.words_tags_count_dict:
                        self.words_tags_count_dict[(curr_word, curr_tag)] = 1
                    else:
                        self.words_tags_count_dict[(curr_word, curr_tag)] += 1
                    self.possible_tags.add(curr_tag) 

    
    def get_tag_all_ngrams_count(self, file_path):
        """
            Extract out of text all n-grams tags when n is in {1,2,3}
            :param file_path: full path of the file to read
                return all ngrams tags with counter of appearance 
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)    
                for word_idx in range(len(splited_words)):
                    if splited_words[word_idx] == '': #remove word that is an empty string 
                        del splited_words[word_idx]
                        continue 
                    t = splited_words[word_idx].split("_")[1]
                    prev_tags = []
                    for index in [1,2]:
                        if index <= word_idx:
                                prev_tags.append(splited_words[word_idx-index].split("_")[1])
                        else:
                            prev_tags.append("*")                    
                    trigram_tags = (prev_tags[1], prev_tags[0], t) #tuple includes (t-2, t-1, t)
                    bigram_tags = (prev_tags[0], t) #tuple includes (t-1, t)
                    unigram_tags = t #includes (t)
                    for ngram_tags, ngram_count_dict in zip([trigram_tags, bigram_tags, unigram_tags], [self.trigram_count_dict, self.bigram_count_dict, self.unigram_count_dict]):
                        if ngram_tags not in ngram_count_dict:
                            ngram_count_dict[ngram_tags] = 1
                        else:
                            ngram_count_dict[ngram_tags] += 1
    
    
    def get_neighbor_word_curr_tag_count(self, file_path):
        """
            Extract out of text all neighbor_word\curr_tag pairs- neighbor can be prev word or next word
            :param file_path: full path of the file to read
                return all neighbor_word\curr_tag pairs with counter of appearance 
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)    
                #if a word in the line is an empty string- delete it from the line             
                splited_words = [word for word in splited_words if word != '']
                for word_idx in range(len(splited_words)):
                    curr_t = splited_words[word_idx].split("_")[1]
                    
                    #prev word curr tag
                    if word_idx > 0:
                        prev_word = splited_words[word_idx - 1].split("_")[0]
                    else:
                        prev_word = '*'
                    #next word curr tag 
                    if word_idx < len(splited_words)-1: #if curr word isn't the punqtuation at the end of the line 
                        next_word = splited_words[word_idx + 1].split("_")[0]
                    else:
                        next_word = 'STOP'
                         
                    for neighbor_word, neighbor_count_dict in zip([prev_word.lower(), next_word.lower()], [self.prev_word_curr_tag_count_dict, self.next_word_curr_tag_count_dict]):
                        if (neighbor_word,curr_t) not in neighbor_count_dict:
                            neighbor_count_dict[(neighbor_word, curr_t)] = 1
                        else:
                            neighbor_count_dict[(neighbor_word, curr_t)] += 1
                            
    def get_prefixes_suffixes_count(self, file_path):
        """
            Extract out of text all prefixes/suffixes & curr_tag pairs (prefixes/suffixes are with len<=4)
            :param file_path: full path of the file to read
                return all prefixes/suffixes & curr_tag pairs with counter of appearance 
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)   
                #if a word in the line is an empty string- delete it from the line             
                splited_words = [word for word in splited_words if word != '']
                for word_idx in range(len(splited_words)):
                    curr_w, curr_tag = splited_words[word_idx].split("_")
                    curr_w = curr_w.lower()
                    suffixes_dict = {"suffix(1)": curr_w[-1:],
                                    "suffix(2)": curr_w[-2:], "suffix(3)": curr_w[-3:],
                                    "suffix(4)": curr_w[-4:]}
                    prefixes_dict = {"prefix(1)": curr_w[:1],
                                    "prefix(2)": curr_w[:2], "prefix(3)": curr_w[:3],
                                    "prefix(4)": curr_w[:4]}
                    if len(curr_w) >= 10:
                        suffixes_dict["suffix(5)"] = curr_w[-5:]
                        suffixes_dict["suffix(6)"] = curr_w[-6:]
                        prefixes_dict["prefix(5)"] = curr_w[:5]
                        prefixes_dict["prefix(6)"] = curr_w[:6]
                        
                    for val in set(suffixes_dict.values()):
                        if (val, curr_tag) not in self.suffixes_count_dict:
                            self.suffixes_count_dict[(val, curr_tag)] = 1
                        else:
                            self.suffixes_count_dict[(val, curr_tag)] += 1
                    for val in set(prefixes_dict.values()):
                        if (val, curr_tag) not in self.prefixes_count_dict:
                            self.prefixes_count_dict[(val, curr_tag)] = 1
                        else:
                            self.prefixes_count_dict[(val, curr_tag)] += 1  

                            
    def get_is_number_count(self, file_path):
        """
            Extract out of text all number/tag pairs
            :param file_path: full path of the file to read
                return all number/tag pairs with counter of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)    
                for word_idx in range(len(splited_words)):
                    if splited_words[word_idx] == '': #remove word that is an empty string 
                        del splited_words[word_idx]
                        continue
                    curr_word, curr_tag = splited_words[word_idx].split("_")    
                    curr_word = curr_word[1:] if curr_word[0]=='-' else curr_word
                    curr_word = curr_word.replace('.', '').replace(':','').replace(',', '')
                    
                    if curr_word.isdigit():    
                        if curr_tag not in self.is_number_count_dict:
                            self.is_number_count_dict[curr_tag] = 1
                        else:
                            self.is_number_count_dict[curr_tag] += 1
                            
                            
    def get_capital_count(self, file_path):
        """
            Extract out of text all tags that belonds to capitalized words- seperated into 3 cases: 
                first word in sentence, mid-sentence words, and entire word capitalized. 
            :param file_path: full path of the file to read
                return all tags of capital words with counter of appearance 
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)    
                for word_idx in range(len(splited_words)):
                    if splited_words[word_idx] == '': #remove word that is an empty string 
                        del splited_words[word_idx]
                        continue
                    curr_word, curr_tag = splited_words[word_idx].split("_")  
                    if any (c.isupper() for c in curr_word):
                        if word_idx == 0: #this is the first word in the sentence 
                            if curr_tag not in self.first_word_capitalized_count_dict:
                                self.first_word_capitalized_count_dict[curr_tag] = 1
                            else:
                                self.first_word_capitalized_count_dict[curr_tag] += 1
                        else: #this is mid-sentence word 
                            if curr_tag not in self.mid_word_capitalized_count_dict:
                                self.mid_word_capitalized_count_dict[curr_tag] = 1
                            else:
                                self.mid_word_capitalized_count_dict[curr_tag] += 1
                    if curr_word.isupper(): #if entire word is capitalized 
                        if any (c.isdigit() for c in curr_word): #word conatins capitals and numbers 
                            if curr_tag not in self.capital_number_count_dict:
                                self.capital_number_count_dict[curr_tag] = 1
                            else:
                                self.capital_number_count_dict[curr_tag] += 1 
                        else:    #word conatins capitals and not numbers 
                            if curr_tag not in self.entire_word_capitalized_count_dict:
                                self.entire_word_capitalized_count_dict[curr_tag] = 1
                            else:
                                self.entire_word_capitalized_count_dict[curr_tag] += 1 
                                
    def get_contain_hyphen_count(self, file_path):
        """
            Extract out of text all words that contain hypehn
            :param file_path: full path of the file to read
                return all words that contain hyphen with counter of appearance
        """
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)   
                for word_idx in range(len(splited_words)):
                    if splited_words[word_idx] == '': #remove word that is an empty string 
                        del splited_words[word_idx]
                        continue
                    curr_word, curr_tag = splited_words[word_idx].split("_")   
                    
                    if any(c=='-' for c in curr_word):    
                        if curr_tag not in self.contain_hyphen_count_dict:
                            self.contain_hyphen_count_dict[curr_tag] = 1
                        else:
                            self.contain_hyphen_count_dict[curr_tag] += 1  
 
    def get_is_company_name_count(self, file_path):
        """
            Extract out of text all words are followed by a company suffix 
            :param file_path: full path of the file to read
                return all words that are followed by a company suffix, with counter of appearance
        """
        company_suffixes = ['Co.', 'Inc.', 'Ltd.', 'Corp.', 'Co', 'Inc', 'Ltd', 'Corp'] 
        
        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)   
                for word_idx in range(len(splited_words)):
                    if splited_words[word_idx] == '': #remove word that is an empty string 
                        del splited_words[word_idx]
                        continue
                    curr_word, curr_tag = splited_words[word_idx].split("_")    
                    if curr_word[0].isupper():
                        three_next_words = splited_words[word_idx+1:word_idx+4]
                        three_next_words = [w.split('_')[0] for w in three_next_words]
                        for word in three_next_words:
                            if word in company_suffixes:
                                if curr_tag not in self.is_company_name_count_dict:
                                    self.is_company_name_count_dict[curr_tag] = 1
                                else:
                                    self.is_company_name_count_dict[curr_tag] += 1 
                                break

  
                                                        
    def get_all_features_counts(self, train_path):
        self.get_word_tag_pair_count(train_path)
        self.get_tag_all_ngrams_count(train_path)
        self.get_neighbor_word_curr_tag_count(train_path)
        self.get_prefixes_suffixes_count(train_path)
        self.get_is_number_count(train_path)
        self.get_capital_count(train_path)
        self.get_contain_hyphen_count(train_path)
        #self.get_is_company_name_count(train_path)


#---------------------feature2id class- calc dict for each feature where count>= threshold-----------#

CWORD = 0
CTAG = 1
PWORD = 2
PTAG = 3
NWORD = 4
PPTAG = 5

class feature2id_class():

    def __init__(self, feature_statistics, threshold):
        self.feature_statistics = feature_statistics  # statistics class, for each featue gives empirical counts
        self.threshold = threshold                    # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0                     # Total number of features accumulated 

        # Init all features dictionaries
        self.words_tags_dict = OrderedDict()
        self.trigram_dict = OrderedDict()
        self.bigram_dict = OrderedDict()
        self.unigram_dict = OrderedDict()
        self.prefixes_dict = OrderedDict()
        self.suffixes_dict = OrderedDict()
        self.prev_word_curr_tag_dict = OrderedDict()
        self.next_word_curr_tag_dict = OrderedDict()
        self.is_number_dict = OrderedDict()
        self.first_word_capitalized_dict = OrderedDict() 
        self.mid_word_capitalized_dict = OrderedDict()    
        self.entire_word_capitalized_dict = OrderedDict()
        self.capital_number_dict = OrderedDict()
        self.contain_hyphen_dict = OrderedDict()          
        #self.is_company_name_dict = OrderedDict() 
        

    def get_all_feature_indexes(self):
        """
            Call to function get_feature_indexes() for each feature 
        """
        self.words_tags_dict = self.get_feature_indexes(self.feature_statistics.words_tags_count_dict, self.words_tags_dict)
        self.suffixes_dict = self.get_feature_indexes(self.feature_statistics.suffixes_count_dict, self.suffixes_dict)
        self.prefixes_dict = self.get_feature_indexes(self.feature_statistics.prefixes_count_dict, self.prefixes_dict)        
        self.trigram_dict = self.get_feature_indexes(self.feature_statistics.trigram_count_dict, self.trigram_dict)
        self.bigram_dict = self.get_feature_indexes(self.feature_statistics.bigram_count_dict, self.bigram_dict)
        self.unigram_dict = self.get_feature_indexes(self.feature_statistics.unigram_count_dict, self.unigram_dict)
        self.prev_word_curr_tag_dict = self.get_feature_indexes(self.feature_statistics.prev_word_curr_tag_count_dict, self.prev_word_curr_tag_dict)
        self.next_word_curr_tag_dict = self.get_feature_indexes(self.feature_statistics.next_word_curr_tag_count_dict, self.next_word_curr_tag_dict)
        self.is_number_dict = self.get_feature_indexes(self.feature_statistics.is_number_count_dict, self.is_number_dict)
        self.first_word_capitalized_dict = self.get_feature_indexes(self.feature_statistics.first_word_capitalized_count_dict, self.first_word_capitalized_dict)
        self.mid_word_capitalized_dict = self.get_feature_indexes(self.feature_statistics.mid_word_capitalized_count_dict, self.mid_word_capitalized_dict)
        self.entire_word_capitalized_dict = self.get_feature_indexes(self.feature_statistics.entire_word_capitalized_count_dict, self.entire_word_capitalized_dict)
        self.capital_number_dict = self.get_feature_indexes(self.feature_statistics.capital_number_count_dict, self.capital_number_dict)
        self.contain_hyphen_dict = self.get_feature_indexes(self.feature_statistics.contain_hyphen_count_dict, self.contain_hyphen_dict)
        #self.is_company_name_dict = self.get_feature_indexes(self.feature_statistics.is_company_name_count_dict, self.is_company_name_dict)

        
    def get_feature_indexes(self, count_dict, feature2id_dict):
        """
            Extract index for each feature that appear more times in text than the 'threshold'
            :param file_path: full path of the file to read
            :param count_dict: a dict from feature_statistics_class
            :param feature2id_dict: the fitting dict from feature2id class
                return all keys with index of appearance
        """
        
        for key, count in count_dict.items():
            if count >= self.threshold:
                feature2id_dict[key] = self.n_total_features
                self.n_total_features += 1
        
        return feature2id_dict
        

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

        cword_lower = cword.lower()

        if (cword_lower, ctag) in self.words_tags_dict:
            features.add(self.words_tags_dict[(cword_lower, ctag)])

        suff_dict = {"suffix(1)": cword_lower[-1:],
                         "suffix(2)": cword_lower[-2:], "suffix(3)": cword_lower[-3:],
                         "suffix(4)": cword_lower[-4:]}
        pref_dict = {"prefix(1)": cword_lower[:1],
                        "prefix(2)": cword_lower[:2], "prefix(3)": cword_lower[:3],
                        "prefix(4)": cword_lower[:4]}
        if len(cword_lower) >= 10:
            suff_dict["suffix(5)"] = cword_lower[-5:]
            suff_dict["suffix(6)"] = cword_lower[-6:]
            pref_dict["prefix(5)"] = cword_lower[:5]
            pref_dict["prefix(6)"] = cword_lower[:6]
                        
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

        if (pword.lower(), ctag) in self.prev_word_curr_tag_dict:
            features.add(self.prev_word_curr_tag_dict[(pword.lower(), ctag)])

        if (nword.lower(), ctag) in self.next_word_curr_tag_dict:
            features.add(self.next_word_curr_tag_dict[(nword.lower(), ctag)])

        if ctag in self.is_number_dict:
            features.add(self.is_number_dict[ctag])

        if ctag in self.first_word_capitalized_dict:
            features.add(self.first_word_capitalized_dict[ctag])
        
        if ctag in self.mid_word_capitalized_dict:
            features.add(self.mid_word_capitalized_dict[ctag])
        
        if ctag in self.entire_word_capitalized_dict:
            features.add(self.entire_word_capitalized_dict[ctag])    

        if ctag in self.capital_number_dict:
            features.add(self.capital_number_dict[ctag])  

        if ctag in self.contain_hyphen_dict:
            features.add(self.contain_hyphen_dict[ctag]) 

        #if ctag in self.is_company_name_dict:
        #    features.add(self.is_company_name_dict[ctag])
            
        return list(features)

    
    def create_history_feature_dict(self, file_path, possible_tags):
        """
            Extract history from text and add its feature vector to a dictionary that will be returned from the function 
            :param file_path: full path of the file to read
            :param possible_tags: list of all possible tags in the train set 
                Return a dict with {keys: history, values: relevant features indexes} 
                    and a dict with{keys: history, values: list of lists of relevant features indexes to each possible tag}
        """
        history_feature_dict = OrderedDict()
        f_h_all_tags_dict = defaultdict(list)

        with open(file_path) as f:
            for line in f:
                splited_words = re.split('\n| ',line)  
                #if a word in the line is an empty string- delete it from the line             
                splited_words = [word for word in splited_words if word != '']

                for word_idx in range(len(splited_words)):
                    curr_history = []
                    curr_history.append(splited_words[word_idx].split("_")[0])    #CWORD 
                    curr_history.append(splited_words[word_idx].split("_")[1])     #CTAG

                    if word_idx > 0:
                        curr_history.append(splited_words[word_idx - 1].split("_")[0])    #PWORD
                        curr_history.append(splited_words[word_idx - 1].split("_")[1])    #PTAG
                    else:
                        curr_history.append('*')    #PWORD
                        curr_history.append('*')    #PTAG

                    if word_idx < len(splited_words)-1: #if curr word isn't the punqtuation at the end of the line 
                        curr_history.append(splited_words[word_idx + 1].split("_")[0])    #NWORD
                    else:
                        curr_history.append('STOP')   #NWORD

                    if word_idx > 1: 
                        curr_history.append(splited_words[word_idx-2].split("_")[1])    #PPTAG
                    else:
                        curr_history.append('*')   #PPTAG

                    history_feature_dict[tuple(curr_history)] = self.represent_input_with_features(curr_history)
                    for tag in possible_tags:
                        tag_history = curr_history.copy()
                        tag_history[CTAG] = tag 
                        f_h_all_tags_dict[tuple(curr_history)].append(self.represent_input_with_features(tag_history))
        return history_feature_dict, f_h_all_tags_dict 


#-------------------auxilary functions-------------------#

import numpy as np

def get_sum_of_f(num_of_features, f_h_dict):
    """
        Calc the sum of all f(xi, yi) for all i in the train set
        :param f_h_dict: dictionary with {keys: history, values: relevant features indexes} 
        Return an int with the sum of all f(xi, yi)
    """
    sum_of_f = np.zeros(num_of_features)
    for f_vec in f_h_dict.values():
        for f_idx in f_vec:
            sum_of_f[f_idx] += 1
    return sum_of_f 



def calc_objective_per_iter(w_i, sum_of_f, param_lambda, f_h_all_tags_dict, num_of_features):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param w_i: weights vector in iteration i 
        :param sum_of_f: calculation of the sum all features together 
        :param param_lambda: hyper-parameter for regularization 
        :param f_h_all_tags_dict: dictionary with {keys: history, values: list of lists where each outer list fit to a history
            and each inner list is relevant features indexes for curr history and specific tag from all possible tags} 
        :param num_of_features: total num of features 
        The function returns the Max Entropy likelihood (objective) and the objective gradient
    """
    normalization_term, expected_counts = calc_normalization_term(w_i, f_h_all_tags_dict, num_of_features)
    
    likelihood = np.dot(w_i,sum_of_f) - normalization_term - 0.5*param_lambda*np.inner(w_i, w_i)
    grad = sum_of_f - expected_counts - param_lambda*w_i

    
    return (-1)*likelihood, (-1)*grad


def calc_normalization_term(w_i, f_h_all_tags_dict, num_of_features):
    """
        Calculate the normalization term of the likelihood
        :param w_i: weights vector in iteration i 
        :param f_h_all_tags_dict: dictionary with {keys: history, values: list of lists where each outer list fit to a history
            and each inner list is relevant features indexes for curr history and specific tag from all possible tags} 
        :param num_of_features: total num of features 
		The function returns the normalization term of the likelihood (objective) 
    """ 

    #total calc version
    outer_L_sum = 0 #this is the normalization term of L 
    outer_grad_sum = np.zeros(num_of_features, dtype=np.float32) #this is the gradient expected count of each k  
    for f_curr_h_all_tags in f_h_all_tags_dict.values():  #f_curr_h_all_tags is list of lists 
        inner_L_sum = 0 
        inner_gradient_sum = np.zeros(num_of_features, dtype=np.float32) 
        curr_k_list=set() #hold the relevant features indexes of current history, without duplicates 
        for f_curr_h_curr_tag in f_curr_h_all_tags: #f_curr_h_curr_tag is a list of indexes 
            #calc L part
            curr_exp = np.exp(w_i[f_curr_h_curr_tag].sum()) 
            inner_L_sum += curr_exp
            
            #calc gradient term 
            for k in f_curr_h_curr_tag:
                inner_gradient_sum[k] += curr_exp
                curr_k_list.add(k)
        if inner_L_sum==0:
            print("divide by zero")
        outer_L_sum += np.log(inner_L_sum) #the base is e 
        one_over_inner = 1.0/inner_L_sum
        for k in curr_k_list:
            outer_grad_sum[k] += one_over_inner * inner_gradient_sum[k] 
    
    return outer_L_sum, outer_grad_sum


#----------creation of the 10 folds. No need to run - allready created------#


from random import seed
from random import randrange
#
# # Split a dataset into k folds
# def cross_validation_split(dataset, folds=10):
#     dataset_split = [None] * len(dataset)
#     dataset_copy = list(dataset)
#     fold_size = int(len(dataset) / folds)
#     for group_num in range(folds):
#         cnt = 0
#         while cnt < fold_size:
#             index = randrange(len(dataset_copy))
#             curr_elem=dataset_copy.pop(index)
#             dataset_split[curr_elem] = group_num
#             cnt += 1
#     return dataset_split
#
# # test cross validation split
# seed(1)
# indexes_list = range(250)
# indexes_groups = cross_validation_split(indexes_list, 10)
#
# train2_path = r"train2.wtag"
# with open(train2_path, 'r') as f_data:
#     for line_num, line in enumerate(f_data):
#         i = indexes_groups[line_num]
#         curr_f = open(r"test_group_%i.wtag" %i,'a')
#         curr_f.write(line)
#         curr_f.close()
#
#
# #create train files
# for i in range(10):
#     curr_train_f = open(r"train_group_%i.wtag" %i,'a')
#     for j in range(10):
#         if j != i:
#             curr_test_f = open(r"test_group_{}.wtag".format(j),'r')
#             curr_train_f.writelines(curr_test_f.readlines())
#             curr_test_f.close()
#     curr_train_f.close()


# # inference

# In[7]:


#--------------------------pre-inference-------------------------------------#
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
        self.capital_number_dict = dictionaries[12]
        self.contain_hyphen_dict = dictionaries[13]
        #self.is_company_name_dict = dictionaries[14]

        
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

            if (cword.lower(), ctag) in self.words_tags_dict:
                features.add(self.words_tags_dict[(cword.lower(), ctag)])
            cword_lower = cword.lower()
            suff_dict = {"suffix(1)": cword_lower[-1:],
                             "suffix(2)": cword_lower[-2:], "suffix(3)": cword_lower[-3:],
                             "suffix(4)": cword_lower[-4:]}
            pref_dict = {"prefix(1)": cword_lower[:1],
                            "prefix(2)": cword_lower[:2], "prefix(3)": cword_lower[:3],
                            "prefix(4)": cword_lower[:4]}
            if len(cword_lower) >= 10:
                suff_dict["suffix(5)"] = cword_lower[-5:]
                suff_dict["suffix(6)"] = cword_lower[-6:]
                pref_dict["prefix(5)"] = cword_lower[:5]
                pref_dict["prefix(6)"] = cword_lower[:6]
            
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

            if (pword.lower(), ctag) in self.prev_word_curr_tag_dict:
                features.add(self.prev_word_curr_tag_dict[(pword.lower(), ctag)])

            if (nword.lower(), ctag) in self.next_word_curr_tag_dict:
                features.add(self.next_word_curr_tag_dict[(nword.lower(), ctag)])

            if ctag in self.is_number_dict:
                features.add(self.is_number_dict[ctag])

            if ctag in self.first_word_capitalized_dict:
                features.add(self.first_word_capitalized_dict[ctag])

            if ctag in self.mid_word_capitalized_dict:
                features.add(self.mid_word_capitalized_dict[ctag])

            if ctag in self.entire_word_capitalized_dict:
                features.add(self.entire_word_capitalized_dict[ctag]) 

            if ctag in self.capital_number_dict:
                features.add(self.capital_number_dict[ctag])  
                
            if ctag in self.contain_hyphen_dict:
                features.add(self.contain_hyphen_dict[ctag]) 

            #if ctag in self.is_company_name_dict:
            #    features.add(self.is_company_name_dict[ctag])                
                
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


#-----------------------------Viterbi algo---------------------#


#viterbi with beam search

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
        if curr_word in special_words_dict.keys():
            curr_possible_tags = special_words_dict[curr_word]
        #check if curr word is digit like- 9, 1.2, 18:30, -90.9 -> if so, v_tag is CD 
        check_curr_word = curr_word[1:] if curr_word[0]=='-' else curr_word
        check_curr_word = check_curr_word.replace('.', '').replace(':','').replace(',','')
        if check_curr_word.isdigit():
            curr_possible_tags = ['CD']
            
        for v_index, v in enumerate(curr_possible_tags):
            if k == 1:
                S_u = ['*']
                S_t = {'*' : set('*')}
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

#------------------------------------------Run code-----------------------------------------------------#

from scipy.optimize import fmin_l_bfgs_b
import pickle
import time 


special_words_dict = {'-LRB-':['-LRB-'], '-RRB-':['-RRB-'], '#':['#'], '$':['$'], '.':['.'], ',':[','], ':':[':'], ';':[':'],
                      '``':['``'], "''":["''"], '-':[':'], '--':[':'], '&':['CC'], '%':['NN', 'JJ'], 'x':['SYM']}

folds = 10
accuracy = [0] * folds 
inference_time = [0] * folds
weights_time = [0] * folds 
for fold_num in range(folds):
#----------------------------------------------------------train part--------------------------------------------------------------#   

    train_path = r'cross_validation\train_group_{}.wtag'.format(fold_num)
    
    # Statistics
    statistics = feature_statistics_class()
    statistics.get_all_features_counts(train_path)

        
    # feature2id
    threshold = 1
    feature2id = feature2id_class(statistics, threshold)
    feature2id.get_all_feature_indexes()
    possible_tags = list(statistics.possible_tags) #change to list in order to permanent the order of elements 
    f_h_dict, f_h_tags_dict = feature2id.create_history_feature_dict(train_path, possible_tags)
    
    
    # aux parameters 
    param_lambda = 1
    num_of_features = feature2id.n_total_features
        
    sum_of_f = get_sum_of_f(num_of_features, f_h_dict)

    # 'args' holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter' except for w_i 
    args = (sum_of_f, param_lambda, f_h_tags_dict, num_of_features)
    w_0 = np.random.uniform(low=-0.001, high=0.001, size=num_of_features) #np.zeros(num_of_features, dtype=np.float32)
    print("Done calculating features and aux parameters. Starting fmin_l_bfgs_b")
    start = time.time()
    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50) 
    end = time.time()
	print("Done training. Total time of fmin_l_bfgs_b is: ", end-start)
    
    weights = optimal_params[0]

 #---------------------------------------------------------inference part--------------------------------------------------------------#   

 
    dics = [feature2id.words_tags_dict, feature2id.trigram_dict, feature2id.bigram_dict, feature2id.unigram_dict,
            feature2id.prefixes_dict, feature2id.suffixes_dict, feature2id.prev_word_curr_tag_dict ,
            feature2id.next_word_curr_tag_dict ,feature2id.is_number_dict, feature2id.first_word_capitalized_dict,
            feature2id.mid_word_capitalized_dict, feature2id.entire_word_capitalized_dict, feature2id.capital_number_dict, 
            feature2id.contain_hyphen_dict] #, feature2id.is_company_name_dict]
    dics_obj = features_indexes_class(dics)
    
    import time 
    import re

    good_cnt = 0
    word_cnt = 0
    iter_count = 0
    mistakes = []
    test_path = r'cross_validation\test_group_{}.wtag'.format(fold_num)
    start = time.time()
    with open(test_path) as test_f:
        for line in test_f:
            splited_words = re.split('_| |\n', line)
            sentence = [x for i, x in enumerate(splited_words) if i % 2 == 0 and x != ""]
            true_tags = [x for i, x in enumerate(splited_words) if i % 2 != 0]  
            viterbi_tags = memm_viterbi(sentence, weights, possible_tags, special_words_dict, dics_obj, B=3)
            for i in range(len(true_tags)):
                word_cnt += 1
                if true_tags[i] == viterbi_tags[i]:
                    good_cnt += 1


            iter_count += 1
    accuracy[fold_num] = good_cnt / word_cnt 


#----------------accuracy calc-------------------#
sum = 0
for i in range(10):
    sum += accuracy[i]
    print("accuracy for fold ",i, " is: ", accuracy[i])
sum = sum/10
print("average accuracy for 10 fold cross validation is: ", sum)

