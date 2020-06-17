import numpy as np


class EmbeddingLoader:
    def __init__(self, embedding_file, w2v_dal):
        assert w2v_dal is not None
        assert 'exists_word' in dir(w2v_dal)
        assert 'get_vectors' in dir(w2v_dal)

        self.embedding = {}
        try:
            with open(embedding_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    # 防止文件中有空行，增加空行判断
                    if line != "" and line != "\r\n" and line != "\r" and line != "\n":
                        items = line.split(' ')
                        word = items[0]
                        vec = np.array([float(val) for val in items[1:]])
                        self.dim = len(items) - 1
                        self.embedding[word] = vec
        except Exception as e:
            print(e)
        self.w2v_dal = w2v_dal

    def get_dim(self):
        return self.dim

    def get_embedding(self, word):
        if word in self.embedding.keys():
            return self.embedding[word]
        elif self.w2v_dal.exists_word(word):  # todo 查找腾讯词向量
            vec_str = self.w2v_dal.get_vectors(word)
            vec = vec_str.split(' ')
            return np.array([float(val) for val in vec])
        else:
            return np.random.rand(self.dim)
