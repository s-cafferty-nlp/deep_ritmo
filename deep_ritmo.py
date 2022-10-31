import pandas as pd
import spacy
import epitran
import re
import unicodedata
import transformers
import spacy_transformers
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from itertools import combinations

class Preprocess():
    def __init__(self,
                 epitran_model = 'spa-Latn',
                 spacy_model = spacy.load('es_dep_news_trf'),
                 custom_vowels = 'aeiouáéíóúäëïöüàèìòùAEIOUÁÉÍÓÚÄËÏÓÜÀÈÌÒÙ',
                 
                ):
        self.epitran_model = epitran_model
        self.epi = epitran.Epitran(self.epitran_model)
        self.vowels = custom_vowels
        self.nlp = spacy_model

    def clean_word(self, word):
        entry = "".join(c for c in word if unicodedata.category(c) not in ["No", "Lo"])
        entry = re.sub('[^A-Za-zÀ-ÿ]+', '', word)
        return entry
    
    def clean_word_list(self, word_list):
        clean_word_list = []
        for word in word_list:
            clean_word_list.append(self.clean_word(word))
        return clean_word_list

    def count_vowels(self, text):
        v_count = 0
        for c in text:
            v_count += 1 if c in set(self.vowels) else 0
        return False if v_count == 0 else True
    
    def ngramify_entry(self, text, n=2):
        return (' ').join([text[i:i+n] for i in range(len(text)-n+1)])
    
    def get_pos(self,text):
        pos = []
        word_pos_doc = self.nlp(text)
        for token in word_pos_doc[:1]:           
            pos.append(token.pos_)       
        return pos[0] 
    
    def get_ipa(self, text): 
        if isinstance(text,str):
            text = self.clean_word(text)
            return self.epi.transliterate(self.clean_word(text))
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [self.epi.transliterate(self.clean_word(w)) for w in text]
    
    def get_sampa(self, text): 
        if isinstance(text,str):
            text = self.clean_word(text)
            return ('').join(self.epi.xsampa_list(text))
        if isinstance(text,list):
            text = self.clean_word_list(text)
            return [('').join(self.epi.xsampa_list(w)) for w in text]

    def weigh_syllables(self, word):
        word = word.split()
        # word.reverse()
        # word = [(len(word) - (idx)) * (' ' + s) for idx, s in enumerate(word)]
        word.reverse()

        return (' ').join(word).strip()

class DeepRitmo(Preprocess):
    def __init__(self,
                tokenizer = None,
                model = None,
                epitran_model='spa-Latn',
                custom_vowels='aeiouáéíóúäëïöüàèìòùAEIOUÁÉÍÓÚÄËÏÓÜÀÈÌÒÙ',
                spacy_model = spacy.load('es_dep_news_trf'),
               ):
        super().__init__(epitran_model, spacy_model, custom_vowels)
        self.tokenizer = tokenizer
        self.model = model
        self.word_list = []
        self.clean_word_list_ = []
        self.preprocessed_word_list = []
        self.vowels = custom_vowels
        self.model_words = []
        self.vocab_matrix = []

    def add_word_list(self, word_list):
        self.word_list = word_list
        self.clean_word_list_ = self.clean_word_list(word_list)
        self.preprocessed_word_list = [self.preprocess_word(w) for w in tqdm(self.clean_word_list_)]
        self.vocab_matrix = self.get_vocab_matrix(self.preprocessed_word_list)

    def preprocess_word(self, word):
        word = self.clean_word(word)
        preprocessed_word = self.get_pos(word) + ' ' + self.weigh_syllables(self.ngramify_entry(self.get_sampa(word)))
        return preprocessed_word

    def get_vector(self, processed_word):
        inputs = self.tokenizer(processed_word, return_tensors='pt')
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        return last_hidden_states[0,0,:].detach().numpy()

    def get_vocab_matrix(self, processed_word_list, embedding_dim=768):
        M = np.zeros([len(processed_word_list),embedding_dim])
        for idx, word in tqdm(enumerate(processed_word_list)):
            M[idx,:] = self.get_vector(word)
        return M

    def generate_random_vectors(self, vector_size, n_vectors=16):
        return np.random.randn(vector_size, n_vectors)

    def train_lsh(self, X, n_vectors, seed=None):    
        if seed is not None:
            np.random.seed(seed)

        dim = X.shape[1]
        random_vectors = self.generate_random_vectors(dim, n_vectors)  

        bin_indices_bits = X.dot(random_vectors) >= 0
        powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
        bin_indices = bin_indices_bits.dot(powers_of_two)

        table = defaultdict(list)
        for idx, bin_index in enumerate(bin_indices):
            table[bin_index].append(idx)

        model = {'table': table,
                'random_vectors': random_vectors,
                'bin_indices': bin_indices,
                'bin_indices_bits': bin_indices_bits}
        return model
    
    def search_nearby_bins(self, query_bin_bits, table, search_radius=3, candidate_set=None):

        if candidate_set is None:
            candidate_set = set()

        n_vectors = query_bin_bits.shape[0]
        powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

        for different_bits in combinations(range(n_vectors), search_radius):
            index = list(different_bits)
            alternate_bits = query_bin_bits.copy()
            alternate_bits[index] = np.logical_not(alternate_bits[index])

            nearby_bin = alternate_bits.dot(powers_of_two)
            if nearby_bin in table:
                candidate_set.update(table[nearby_bin])

        return candidate_set

    def get_nearest_neighbors(self, X, query_vector, model, max_search_radius=3):
        table = model['table']
        random_vectors = model['random_vectors']

        bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

        candidate_set = set()
        for search_radius in range(max_search_radius + 1):
            candidate_set = self.search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)

        candidate_list = list(candidate_set)
        candidates = X[candidate_list]
        distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()
        
        distance_col = 'distance'
        nearest_neighbors = pd.DataFrame({
            'id': candidate_list, distance_col: distance
        }).sort_values(distance_col).reset_index(drop=True)
        return nearest_neighbors

    def query_all_vocab(self, entry, weight=True):
        query_vector = self.get_vector(self.preprocess_word(entry)).reshape(1, -1)
        model = self.train_lsh(self.vocab_matrix, n_vectors=16, seed=143)
        results = self.get_nearest_neighbors(self.vocab_matrix, query_vector, model, max_search_radius=5)
        
        results['score'] = 1-results['distance']
        results['text'] = np.array(self.clean_word_list_)[list(results['id'])]
      
        results = results[['text','score']].copy()
        return results
