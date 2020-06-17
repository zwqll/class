import logging
import time
import os
import numpy as np
from gensim.models import word2vec
from tokenizer import Tokenizer
from text_config import *


class SentenceIter:
    def __init__(self, train_data, model_name, tokenizer):
        assert isinstance(tokenizer, Tokenizer)

        self.train_data = train_data
        self.model_name = model_name
        self.tokenizer = tokenizer

    def __iter__(self):
        for _, line in enumerate(self.train_data):
            try:
                line = line.strip()
                yield self.tokenizer.get_word_seq(line)
            except Exception as e:
                print(e)


class Word2vecTrainer:
    def __init__(self, config, tokenizer):
        assert isinstance(tokenizer, Tokenizer)
        assert isinstance(config, Word2VecTrainConfig)
        assert len(config.word2vec_save_path) > 0

        # 如果词向量存储目录不存在则创建
        try:
            w2v_save_dir = os.path.dirname(config.word2vec_save_path)
            if not os.path.exists(w2v_save_dir):
                os.mkdir(w2v_save_dir)
        except Exception as e:
            print(e)

        self.tokenizer = tokenizer
        self.config = config

    def train(self, train_data, model_name):
        """
        训练词向量
        :param train_data: 句子序列
        :param model_name: 模型名字
        :return:
        """
        t1 = time.time()
        sentence_iter = SentenceIter(train_data, model_name, tokenizer=self.tokenizer)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        print('开始训练词向量')
        try:
            model = word2vec.Word2Vec(sentence_iter, size=self.config.dim,
                                      window=5, min_count=1, workers=6, sg=self.config.sg)
        except Exception as e:
            print(e)
            return

        model.wv.save_word2vec_format(self.config.word2vec_save_path, binary=False)
        print('词向量训练完成')

        # 修改词向量格式
        self._rewrite_vec_file()

        if self.config.use_tencent:
            # 使用腾讯词向量替换
            try:
                self._rewrite_word2vec_file_with_redis_tencent_vectors()
            except Exception as e:
                print(e)
        print('-------------------------------------------')
        print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))

    def _rewrite_vec_file(self):
        vec = []
        with open(self.config.word2vec_save_path, 'r', encoding="utf-8") as f:
            f.readline()
            for line in f.readlines():
                vec.append(line)

        with open(self.config.word2vec_save_path, 'w', encoding="utf-8") as f:
            for item in vec:
                f.write(item)

            # 写入未知词向量
            unknown_vec_str = '<UNK>'
            unknown_vec = np.random.rand(200)
            for i in range(0, 200):
                unknown_vec_str += ' ' + str(unknown_vec[i])
            f.write(unknown_vec_str)

        print('词向量重写完成')

    def _rewrite_word2vec_file_with_redis_tencent_vectors(self):
        our_vectors = {}
        print('开始更新词向量')
        change_count = 0
        start_time = time.time()
        with open(self.config.word2vec_save_path, 'r', encoding="utf-8") as f:
            for line in f:
                tmp = line.strip().split(' ')
                word = tmp[0]
                vector = ' '.join(tmp[1:])
                if self.config.tencent_word2vec_dal.exists_word(word):
                    new_vector = self.config.tencent_word2vec_dal.get_vectors(word)
                    our_vectors[word] = new_vector
                    change_count += 1
                else:
                    our_vectors[word] = vector

        with open(self.config.word2vec_save_path, 'w', encoding="utf-8") as f:
            for key in our_vectors:
                f.write(key + ' ' + our_vectors[key] + '\n')

                our_vectors[tmp[0]] = ' '.join(tmp[1:])

        print('词向量更新完成')
        print('更新数量：{}/{}'.format(change_count, len(our_vectors)))
        print('用时：{}'.format(time.time() - start_time))


if __name__ == '__main__':
    # 1. 定义分词器
    t = Tokenizer(use_single_char=True)

    # 2. 获取原始句子语料
    test_train_data_file = u'D:\\项目\\AI助手\\数据\\经销商\\location_corpus_800.txt'
    data_list = []
    with open(test_train_data_file, 'r', encoding='utf-8') as train_data_f:
        for corpus in train_data_f.readlines():
            data_list.append(corpus.replace('\n', ''))

    # 3. 训练词向量
    conf = Word2VecTrainConfig()
    conf.word2vec_save_path = './test_vec.txt'
    conf.use_tencent = False

    trainer = Word2vecTrainer(config=conf, tokenizer=t)
    trainer.train(train_data=data_list, model_name='vec_test')
