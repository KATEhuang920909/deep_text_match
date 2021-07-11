import pandas as pd
import os
from utils.data_utils import pad_sequences
from drmm_args import document_seq_length, query_seq_length


class Data2idx(object):
    def __init__(self, query_seq_length, document_seq_length):
        self.query_seq_length = query_seq_length
        self.document_seq_length = document_seq_length
        path = os.path.join(os.path.dirname(__file__), '../data/vocab.txt')
        print("test")
        self.vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
        self.char2idx = {char: index for index, char in enumerate(self.vocab)}

    def idx(self, word, ):  # OOV items
        if word not in self.vocab:
            self.char2idx[word] = len(self.vocab)
            self.vocab.append(word)
        return self.char2idx[word]

    # char->index
    def char_index(self, querys, documents, query_seq_length, document_seq_length):

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
    def load_char_data(self, file, frac=1.0, n=None):
        path = os.path.join(os.path.dirname(__file__), '../' + file)
        df = pd.read_csv(path).sample(n=n, frac=frac)
        querys = df['sentence1'].values
        documents = df['sentence2'].values
        label = df['label'].values
        q_c_index, d_c_index = self.char_index(querys, documents, self.query_seq_length, self.document_seq_length)
        return q_c_index, d_c_index, label


if __name__ == '__main__':
    data2idx = Data2idx(query_seq_length, document_seq_length)
    p_c_index, h_c_index, label = data2idx.load_char_data('data/train.csv', frac=None, n=100)
    print(p_c_index)
