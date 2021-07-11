import pandas as pd
import os
from utils.data_utils import pad_sequences
from drmm_args import document_seq_length, query_seq_length
from tqdm import tqdm
from gensim.corpora.dictionary import Dictionary
import numpy as np


class Data2idx(object):
    def __init__(self, ):
        path = os.path.join(os.path.dirname(__file__), '../data/vocab.txt')
        print("test")
        self.vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
        self.char2idx = {char: index for index, char in enumerate(self.vocab)}

    # def idx(self, word, ):  # OOV items
    #     if word not in self.vocab:
    #         self.char2idx[word] = len(self.vocab)
    #         self.vocab.append(word)
    #     return self.char2idx[word]
    def load_data(self, file, frac=1.0, n=None):
        path = os.path.join(os.path.dirname(__file__), '../' + file)
        df = pd.read_csv(path).sample(n=n, frac=frac)
        querys = df['sentence1'].values.tolist()
        documents = df['sentence2'].values.tolist()
        label = df['label'].values
        return querys, documents, label

    # char->index
    def load_char_idx(self, querys, documents, query_seq_length, document_seq_length):
        print("test0")
        q_list, d_list = [], []
        char2idx = self.char2idx
        for query, document in tqdm(zip(querys, documents)):
            q_l = [char2idx[word.strip()] for word in query if word.strip() and word in char2idx.keys()]
            d_l = [char2idx[word.strip()] for word in document if word.strip() and word in char2idx.keys()]
            q_list.append(q_l)
            d_list.append(d_l)
        print("test1")
        q_list = pad_sequences(q_list, maxlen=query_seq_length)
        d_list = pad_sequences(d_list, maxlen=document_seq_length)

        return q_list, d_list

    # 加载char_index训练数据
    def load_word_idx(self,  w2indx, maxlen,*args):
        ''' Words become integers
        '''

        result =[]
        for sentences in args:
            data = []
            for sentence in sentences:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  # freqxiao10->0
                data.append(new_txt)
            data = pad_sequences(data, maxlen=maxlen)
            result.append(data)
        return result  # word=>index

    def create_dictionaries(self,
                            model=None):
        ''' Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries

        '''
        if model is not None:
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab.keys(),
                                allow_update=True)
            #  freqxiao10->0 所以k+1
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}
            w2vec = {word: model[word] for word in w2indx.keys()}

            return w2indx, w2vec
        else:
            print('No data provided...')

    def get_embedding(self, index_dict, word_vectors, vocab_dim):  # ,combined,y):

        n_symbols = len(index_dict) + 1
        embedding_weights = np.zeros((n_symbols, vocab_dim))
        for word, index in index_dict.items():
            embedding_weights[index, :] = word_vectors[word]
        # x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
        # y_train = to_categorical(y_train,num_classes=3)
        # y_test = to_categorical(y_test,num_classes=3)
        # print x_train.shape,y_train.shape
        return n_symbols, embedding_weights  # ,x_train,y_train,x_test,y_test


if __name__ == '__main__':
    data2idx = Data2idx(query_seq_length, document_seq_length)
    p_c_index, h_c_index, label = data2idx.load_char_idx('data/train.csv', frac=None, n=100)
    print(p_c_index)
