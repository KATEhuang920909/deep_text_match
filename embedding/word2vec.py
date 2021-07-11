from gensim.models import Word2Vec
# from gensim.models import KeyedVectors
# from ..utils import input_helpers
import numpy as np
import pandas as pd
import jieba


def word2vec_train(data, pre_train_path=None, need_finetune=False):  # 全量训练/微调
    """
    data形状：[tokensize_sentence1,tokensize_sentence2...]
    1.是否有pre_model
    2.是否需要fine tune
    :param data:
    :return:
    """

    if pre_train_path:
        model = Word2Vec.load(pre_train_path)

        if need_finetune:
            model.train(data, total_examples=1, epochs=1)
            model.save(r'D:\learning\text_match\deep_text_match\output\word2vec.model')

    else:
        model = Word2Vec(data, size=256, window=5, min_count=1, workers=4)
        model.save(r'D:\learning\text_match\deep_text_match\output\word2vec\word2vec.model')
        # np.save('../pre_train_model/word2vec.npy',model,allow_pickle=True)
        model.wv.save_word2vec_format(r'D:\learning\text_match\deep_text_match\output\word2vec\word2vec.vector',
                                      binary=False)


if __name__ == '__main__':
    train = pd.read_csv(r"D:\learning\text_match\deep_text_match\data\train.csv")
    test = pd.read_csv(r"D:\learning\text_match\deep_text_match\data\test.csv")
    dev = pd.read_csv(r"D:\learning\text_match\deep_text_match\data\dev.csv")
    data = pd.concat([train, dev, test], ignore_index=True)
    sentences_1_bag = data.sentence1.values.tolist()
    sentences_2_bag = data.sentence2.values.tolist()
    sentences_1_bag = [jieba.lcut(k.strip()) for k in sentences_1_bag]
    sentences_2_bag = [jieba.lcut(k.strip()) for k in sentences_2_bag]
    print(sentences_1_bag[:2])
    print("\n")
    print(sentences_2_bag[:2])
    data_tokenize = sentences_1_bag + sentences_2_bag

    word2vec_train(data_tokenize)
