import jieba
import jieba.posseg
import pickle
import os


class Tokenizer:
    """
    自定义分词器，可以通过文件恢复，为了确保每个模型使用特定且唯一的分词方式
    """
    def __init__(self, user_dict_path='', entity_dict_path='', stop_words_path='',
                 user_dict=(), entity_dict=(), stop_words=(), use_single_char=False):
        """
        初始化分词器，用词典初始化
        :param user_dict_path: 用户词典路径
        :param entity_dict_path: 实体词典路径
        :param stop_words_path: 停用词路径
        :param user_dict: 用户词典集合
        :param entity_dict: 实体词典集合
        :param stop_words: 停用词集合
        """
        assert isinstance(user_dict_path, str)
        assert isinstance(entity_dict_path, str)
        assert isinstance(stop_words_path, str)
        assert isinstance(user_dict, tuple)
        assert isinstance(entity_dict, tuple)
        assert isinstance(stop_words, tuple)

        self.use_single_char = use_single_char
        # 初始化结巴分词器
        self.tokenizer = jieba.Tokenizer()
        try:
            if os.path.exists(user_dict_path):
                self.tokenizer.load_userdict(user_dict_path)
            if os.path.exists(entity_dict_path):
                self.tokenizer.load_userdict(entity_dict_path)
            for word in user_dict:
                self.tokenizer.add_word(word)
            for word in entity_dict:
                self.tokenizer.add_word(word)
        except Exception as e:
            print(e)
        self.pos_tokenizer = jieba.posseg.POSTokenizer(tokenizer=self.tokenizer)

        # 初始化停用词表
        self.stop_words = []
        try:
            if os.path.exists(stop_words_path):
                with open(stop_words_path, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        word = line.replace('\r', '').replace('\n', '').replace('\t', '').replace(' ', '')
                        self.stop_words.append(word)

            self.stop_words.extend(list(stop_words))
            self.stop_words = list(set(self.stop_words))
        except Exception as e:
            print(e)

    def get_word_seq(self, sentence):
        """
        对句子进行分词，分词结果为词的序列（统一分词方法，避免不同阶段分词结果不同）
        :return: 分词结果
        :param sentence: 需要分词的句子
        :return:
        """
        word_seq = self.tokenizer.cut(sentence)
        rtn = []
        for word in word_seq:
            if word not in self.stop_words:
                rtn.append(word)

        if self.use_single_char:
            for word in list(sentence):
                if word not in self.stop_words:
                    rtn.append(word)
        return rtn

    def get_word_seq_pos(self, sentence):
        """
        对句子进行分词，分词结果为词/词性的序列（统一分词方法，避免不同阶段分词结果不同）
        :return: 分词结果
        :param sentence: 需要分词的句子
        :return:
        """
        word_seq = self.pos_tokenizer.cut(sentence)
        rtn = []
        for word in word_seq:
            if word not in self.stop_words:
                rtn.append(word)
        return rtn

    @staticmethod
    def save(tokenizer, save_path):
        try:
            assert isinstance(tokenizer, Tokenizer)

            with open(save_path, 'wb') as f:
                pickle.dump(tokenizer, f)
        except Exception as e:
            print(e)

    @staticmethod
    def load(save_path):
        try:
            assert os.path.isfile(save_path)

            with open(save_path, 'rb') as f:
                tokenizer = pickle.load(f)
        except Exception as e:
            print(e)
        return tokenizer


if __name__ == '__main__':
    user_dict_ = ('我去',)
    stop_words_ = ('上',)
    t = Tokenizer(user_dict=user_dict_, stop_words=stop_words_, use_single_char=True)
    print(t.get_word_seq('我去上学校'))
