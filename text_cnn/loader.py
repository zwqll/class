from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import codecs
import random
import pickle


def split(full_list, shuffle=False, ratio=0.2):
    """
    拆分训练集
    :param full_list: 完整训练集
    :param shuffle: 是否打乱顺序
    :param ratio: 验证集所占比例
    :return: 验证集、训练集
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


def read_train_data(train_data, tokenizer):
    """
    将训练集解析成label、word_seq返回
    :param train_data: 训练集
    :param tokenizer: 分词器
    :return: label、word_seq
    """
    contents, labels = [], []

    for line in train_data:
        try:
            line = line.rstrip()
            line = line.replace(' ', '').replace('\n', '').replace('\r\n', '')

            content, label = line.split('\t')
            labels.append(label)

            words = tokenizer.get_word_seq(content)
            contents.append(words)
        except Exception as e:
            print(e)
    return labels, contents


def build_mapping(train_data, word2id_path, ctg2id_path, tokenizer):
    """
    根据全量训练集构建word2id、
    :param train_data: 全量训练集
    :param word2id_path: 词-id映射存储路径
    :param ctg2id_path: 分类-id映射存储路径
    :param tokenizer: 分词器
    :return: ctg2id，word2id
    """
    # 读取标签及分词结果
    labels, data_train = read_train_data(train_data, tokenizer)

    # 创建ctg2id mapping
    labels = list(set(labels))
    ctg2id = dict(zip(labels, range(len(labels))))
    try:
        with codecs.open(ctg2id_path, 'wb') as f:
            pickle.dump(ctg2id, f)
    except Exception as e:
        print(e)

    # 创建word2id mapping
    all_words = []
    for content in data_train:
        all_words.extend(content)

    # 根据词频给词排序后做映射（个人感觉可有可无）
    counter = Counter(all_words)
    count_pairs = counter.most_common()
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)

    word2id = dict(zip(words, range(len(words))))
    try:
        with codecs.open(word2id_path, 'wb') as f:
            pickle.dump(word2id, f)

    except Exception as e:
        print(e)
    return ctg2id, word2id


def read_word_embedding(word2id, word2vec_save_path):
    """
    读取词向量矩阵
    :param word2vec_save_path: 词向量保存路径
    :param word2id: word2id映射
    :return: 二维词向量矩阵（numpy类型），外层index为word2id的值；内层为word的向量
    """
    # 获取词向量长度
    try:
        with open(word2vec_save_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # 防止文件中有空行，增加空行判断
                if line != "" and line != "\r\n" and line != "\r" and line != "\n":
                    items = line.split(' ')
                    dim = len(items) - 1
                    break
    except Exception as e:
        print(e)

    embeddings = np.zeros((len(word2id) + 1, dim))
    try:
        with open(word2vec_save_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # 防止文件中有空行，增加空行判断
                if line != "" and line != "\r\n" and line != "\r" and line != "\n":
                    items = line.split(' ')
                    word = items[0]
                    vec = np.array([float(val) for val in items[1:]])
                    if word in word2id.keys():
                        embeddings[word2id[word]] = vec
    except Exception as e:
        print(e)
    return embeddings


def process_for_train(data, word_to_id, cat_to_id, max_length=20, tokenizer=None):
    """
    预处理，将输入文本转化成数字矩阵
    :param data: 输入的文本集合（数据-分类键值对）
    :param word_to_id: word2id映射
    :param cat_to_id: ctg2id映射
    :param max_length: 最大句子长度
    :param tokenizer: 分词器
    :return:
    """
    label_list, word_seq_list = read_train_data(data, tokenizer)
    word_id_seq_list, label_id_list = [], []
    for i in range(len(word_seq_list)):
        word_id_seq_list.append([word_to_id[x] for x in word_seq_list[i] if x in word_to_id.keys()])
        label_id_list.append(cat_to_id[label_list[i]])
    # 输入数据转为多维数组，词转为id，超过max_length截断，小于max_length填充
    x_pad = kr.preprocessing.sequence.pad_sequences(word_id_seq_list, max_length, padding='post', truncating='post')
    y_pad = kr.utils.to_categorical(label_id_list, len(cat_to_id))

    return x_pad, y_pad


def process_for_predict(sentence, word_to_id, max_length=20, tokenizer=None):
    data_id = []
    id_seq = []
    seg_list = tokenizer.get_word_seq(sentence)
    for i in range(len(seg_list)):
        # 对于未训练过的新词，赋予新的id，后续会未其生成随机词向量
        if seg_list[i] in word_to_id.keys():
            id_seq.append(word_to_id[seg_list[i]])
        else:
            id_seq.append(0)
    data_id.append(id_seq)

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    return x_pad


def process_with_embedding(data, cat_to_id=None, max_length=20, tokenizer=None, embedding_loader=None, is_train=True):
    """
    预处理，将输入文本转化成数字矩阵
    :param data: 输入的文本集合（数据-分类键值对）
    :param cat_to_id: ctg2id映射
    :param max_length: 最大句子长度
    :param tokenizer: 分词器
    :param embedding_loader: 获取词向量的操作类需要有get_dim及get_embedding接口
    :param is_train: 是否是训练（如果不是则为预测，不返回ctg）
    :return:
    """
    assert embedding_loader is not None
    assert 'get_dim' in dir(embedding_loader)
    assert 'get_embedding' in dir(embedding_loader)

    # 分词
    if is_train:
        assert cat_to_id is not None and isinstance(cat_to_id, dict)
        label_list, word_seq_list = read_train_data(data, tokenizer)
    else:
        word_seq_list = [tokenizer.get_word_seq(data)]

    # 获取词向量
    dim = embedding_loader.get_dim()
    shape = (len(word_seq_list), max_length, dim)
    x_pad = np.zeros(shape)
    for i, sentence in enumerate(word_seq_list):
        for j, word in enumerate(sentence):
            if j >= max_length:
                break
            x_pad[i][j] = embedding_loader.get_embedding(word)

    if is_train:
        # 获取标签id
        label_id_list = []
        for i in range(len(word_seq_list)):
            label_id_list.append(cat_to_id[label_list[i]])
        y_pad = kr.utils.to_categorical(label_id_list, len(cat_to_id))
        return x_pad, y_pad
    return x_pad


def process_train_data(data, word_to_id=None, cat_to_id=None, x_max_length=20,
                       tokenizer=None, load_embedding=False, embedding_loader=None):
    # TODO 重构process方法，为了后续方便维护
    pass


def process_test_data(sentence, word_to_id=None, max_length=20,
                      tokenizer=None, load_embedding=False, embedding_loader=None):
    # TODO 重构process方法，为了后续方便维护
    pass


def x_padding(to_padding_x, max_length):
    """
    训练集填充
    :return: padding结果
    """
    return kr.preprocessing.sequence.pad_sequences(to_padding_x, max_length, padding='post', truncating='post')


def y_padding(to_padding_y, max_length):
    """
    label填充
    :return: padding结果
    """
    return kr.utils.to_categorical(to_padding_y, max_length)


def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
