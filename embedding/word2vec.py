from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from ..utils import input_helpers
import numpy as np


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
            model.save('../pre_train_model/word2vec.model')

    else:
        model = Word2Vec(data, size=256, window=5, min_count=1, workers=4)
        model.save('../pre_train_model/word2vec.model')
        # np.save('../pre_train_model/word2vec.npy',model,allow_pickle=True)
        model.wv.save_word2vec_format('../pre_train_model/word2vec.vector', binary=False)


if __name__ == '__main__':
    data_loader = input_helpers.InputHelper()
    data_loader.load_file()
    print(data_loader.x_train[0])
    print(data_loader.x_test[0])
    data_tokenize = data_loader.data_token(data_loader.x_train + data_loader.x_test)
    print(data_tokenize[0])
    word2vec_train(data_tokenize)
