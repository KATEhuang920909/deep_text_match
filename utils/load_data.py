import pandas as pd
import os
from utils.data_utils import shuffle, pad_sequences
import jieba
import re
from gensim.models import Word2Vec
import numpy as np
from args import drmm_args


def text2vec(txt):
    txt = list(txt)
    txt_vec = bert_encoder(txt)

    return txt_vec


def data2vec(data, columns):
    # train_data = pd.read_csv(path + "train_data.csv")
    # test_data = pd.read_csv(path + "test_data.csv")
    # valid_data = pd.read_csv(path + "valid_data.csv")
    # return train_data, valid_data, test_data
    data[columns] = data[columns].apply(lambda x: ''.join(x.split('0v0')))
    data[columns] = data[columns].apply(lambda x: text_clean(x))
    data = data[columns].apply(lambda x: text2vec(list(x))).values
    return data


def bert_process(vec, length):
    if len(vec) > length:
        vec = vec[0:length]
    elif len(vec) < length:
        zero = np.zeros(length)
        length = length - len(vec)
        for i in range(length):
            vec = np.vstack((vec, zero))
    return vec


# def load_data(path):
#     train_data = pd.read_csv(path + "trainset/recruit_folder.csv", header=0, names=["岗位编号", "求职者编号", "标签"])
#     test_data = pd.read_csv(path + "testset/recruit_folder.csv", header=0, names=["岗位编号", "求职者编号", "标签"])
#     train_data = train_data.sample(frac=1.0)
#     train_data, valid_data = train_test_split(train_data, test_size=0.2)
#     train_data = train_data.reset_index(drop=True)
#     valid_data = valid_data.reset_index(drop=True)
#     # valid_data = pd.read_csv(path + "valid_data.csv")
#     return train_data, valid_data, test_data


# 加载字典
def load_char_vocab():
    path = os.path.join(os.path.dirname(__file__), '../data/vocab.txt')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


# 加载词典
def load_word_vocab():
    path = os.path.join(os.path.dirname(__file__), '../output/word2vec/word_vocab.tsv')
    vocab = [line.strip() for line in open(path, encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


# 静态w2v
def w2v(word, model):
    try:
        return model.wv[word]
    except:
        return np.zeros(drmm_args.word_embedding_len)


# 字->index
def char_index(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=drmm_args.query_seq_length)
    h_list = pad_sequences(h_list, maxlen=drmm_args.document_seq_length)

    return p_list, h_list


# 词->index
def word_index(p_sentences, h_sentences):
    word2idx, idx2word = load_word_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word.lower()] for word in p_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]
        h = [word2idx[word.lower()] for word in h_sentence if len(word.strip()) > 0 and word.lower() in word2idx.keys()]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=drmm_args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=drmm_args.max_char_len)

    return p_list, h_list


def w2v_process(vec):
    if len(vec) > drmm_args.max_word_len:
        vec = vec[0:drmm_args.max_word_len]
    elif len(vec) < drmm_args.max_word_len:
        zero = np.zeros(drmm_args.word_embedding_len)
        length = drmm_args.max_word_len - len(vec)
        for i in range(length):
            vec = np.vstack((vec, zero))
    return vec


# 加载char_index训练数据
def load_char_data(file, data_size=None):
    path = os.path.join(os.path.dirname(__file__), '../' + file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    # [1,2,3,4,5] [4,1,5,2,0]
    p_c_index, h_c_index = char_index(p, h)

    return p_c_index, h_c_index, label


# 加载char_index与静态词向量的训练数据
def load_char_word_static_data(file, data_size=None):
    model = Word2Vec.load('../output/word2vec/word2vec.model')

    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    return p_c_index, h_c_index, p_w_vec, h_w_vec, label


# 加载char_index与动态词向量的训练数据
def load_char_word_dynamic_data(path, data_size=None):
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_char_index, h_char_index = char_index(p, h)

    p_seg = map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p)
    h_seg = map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h)

    p_word_index, h_word_index = word_index(p_seg, h_seg)

    return p_char_index, h_char_index, p_word_index, h_word_index, label


# 加载char_index、静态词向量、动态词向量的训练数据
def load_all_data(path, data_size=None):
    model = Word2Vec.load('../output/word2vec/word2vec.model')
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:data_size]
    h = df['sentence2'].values[0:data_size]
    label = df['label'].values[0:data_size]

    p, h, label = shuffle(p, h, label)

    p_c_index, h_c_index = char_index(p, h)

    p_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), p))
    h_seg = list(map(lambda x: list(jieba.cut(re.sub("[！，。？、~@#￥%&*（）.,:：|/`()_;+；…\\\\\\-\\s]", "", x))), h))

    p_w_index, h_w_index = word_index(p_seg, h_seg)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_w_vec = list(map(lambda x: w2v(x, model), p_seg))
    h_w_vec = list(map(lambda x: w2v(x, model), h_seg))

    p_w_vec = list(map(lambda x: w2v_process(x), p_w_vec))
    h_w_vec = list(map(lambda x: w2v_process(x), h_w_vec))

    # 判断是否有相同的词
    same_word = []
    for p_i, h_i in zip(p_w_index, h_w_index):
        dic = {}
        for i in p_i:
            if i == 0:
                break
            dic[i] = dic.get(i, 0) + 1
        for index, i in enumerate(h_i):
            if i == 0:
                same_word.append(0)
                break
            dic[i] = dic.get(i, 0) - 1
            if dic[i] == 0:
                same_word.append(1)
                break
            if index == len(h_i) - 1:
                same_word.append(0)

    return p_c_index, h_c_index, p_w_index, h_w_index, p_w_vec, h_w_vec, same_word, label


if __name__ == '__main__':
    load_all_data('../input/train.csv', data_size=100)
