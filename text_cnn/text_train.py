from __future__ import print_function
import time
import os
import traceback
from text_model import *
from text_config import *
from loader import *
from w2v_dal import W2VDAL


class TextCNNTrainer:
    def __init__(self, config, tokenizer):
        """
        初始化trainer
        :param config: 训练参数，参考TextConfig
        :param tokenizer: 分词器
        """
        assert isinstance(config, TextConfig)
        assert config.word2vec_file_path != ''

        self.config = config
        self.tokenizer = tokenizer
        self.model = None

    def train(self, train_data, model_name):
        print('Configuring CNN model...')
        prev = self.config.base_data_path + model_name + '/' + model_name

        # 把语料分词，做成词典
        cat_to_id, word_to_id = build_mapping(train_data,
                                              word2id_path=prev + self.config.word2id_filename,
                                              ctg2id_path=prev + self.config.ctg2id_filename,
                                              tokenizer=self.tokenizer)

        self.config.vocab_size = len(word_to_id) + 1  # +1是因为有unknown的随机词向量
        self.config.num_classes = len(cat_to_id)

        if not self.config.embedding_as_input:
            self.config.pre_training = read_word_embedding(word_to_id, self.config.word2vec_file_path)
        # 初始化tensor图结构
        self.model = TextCNN(self.config)

        print("Configuring TensorBoard and Saver...")
        self.config.tensor_board_dir = self.config.tensor_board_dir + '/' + model_name
        self.config.model_save_dir = self.config.model_save_dir + '/' + model_name
        if not os.path.exists(self.config.tensor_board_dir):
            os.makedirs(self.config.tensor_board_dir)
        if not os.path.exists(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)
        save_path = os.path.join(self.config.model_save_dir, model_name)

        print("Loading training and validation data...")
        start_time = time.time()

        # 训练集验证集分割
        sublist_1, sublist_2 = split(train_data, shuffle=self.config.shuffle, ratio=self.config.split_ratio)

        # 输入数据转为多维数组，词转为id，超过max_length截断，小于max_length填充
        if self.config.embedding_as_input:
            x_train, y_train = process_with_embedding(
                sublist_2, cat_to_id, self.config.seq_length,
                self.tokenizer, embedding_loader=self.config.embedding_loader, is_train=True)
            x_val, y_val = process_with_embedding(
                sublist_1, cat_to_id, self.config.seq_length,
                self.tokenizer, embedding_loader=self.config.embedding_loader, is_train=True)
        else:
            x_train, y_train = process_for_train(
                sublist_2, word_to_id, cat_to_id, self.config.seq_length, self.tokenizer)
            x_val, y_val = process_for_train(
                sublist_1, word_to_id, cat_to_id, self.config.seq_length, self.tokenizer)
        print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

        tf.summary.scalar("loss", self.model.loss)
        tf.summary.scalar("accuracy", self.model.acc)
        merged_summary = tf.summary.merge_all()

        writer = tf.summary.FileWriter(self.config.tensor_board_dir)
        saver = tf.train.Saver()

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        print('Training and evaluating...')
        best_val_accuracy = 0
        last_improved = 0  # record global_step at best_val_accuracy
        flag = False
        result_dic = {
            'EndTime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'State': 1,
            'RoundNum': 0,
            'AccuracyRate': '0', 'LossRate': '0'
        }

        epoch = 0
        train_loss = 1
        for epoch in range(self.config.num_epochs):
            batch_train = batch_iter(x_train, y_train, self.config.batch_size)
            start = time.time()
            print('Epoch:', epoch + 1)
            for x_batch, y_batch in batch_train:
                feed_dict = self.feed_data(x_batch, y_batch, self.config.keep_prob)
                _, global_step, train_summaries, train_loss, train_accuracy = session.run(
                    [self.model.optim, self.model.global_step,
                     merged_summary, self.model.loss,
                     self.model.acc], feed_dict=feed_dict)

                if global_step % self.config.print_per_batch == 0:
                    end = time.time()
                    val_loss, val_accuracy = self.evaluate(session, x_val, y_val)
                    writer.add_summary(train_summaries, global_step)

                    # If improved, save the model
                    if val_accuracy > best_val_accuracy:
                        saver.save(session, save_path)
                        best_val_accuracy = val_accuracy
                        last_improved = global_step
                        improved_str = '*'
                    else:
                        improved_str = ''
                    print(
                        "step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, "
                        "val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                            global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                            (end - start) / self.config.print_per_batch, improved_str))
                    start = time.time()

                    result_dic['EndTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    result_dic['RoundNum'] = epoch
                    result_dic['AccuracyRate'] = str(best_val_accuracy)
                    result_dic['LossRate'] = str(train_loss)
                    yield result_dic

                if global_step - last_improved > self.config.require_improvement:
                    print("No optimization over " + str(self.config.require_improvement) + " steps, stop training")
                    flag = True
                    break
            if flag:
                break
            self.config.lr *= self.config.lr_decay

        result_dic['State'] = 2
        result_dic['EndTime'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        result_dic['RoundNum'] = epoch
        result_dic['AccuracyRate'] = str(best_val_accuracy)
        result_dic['LossRate'] = str(train_loss)

        # 训练结束要关闭sess，否则再次训练会出问题
        session.close()
        return result_dic

    def evaluate(self, sess, x_, y_):
        data_len = len(x_)
        batch_eval = batch_iter(x_, y_, self.config.batch_size)
        total_loss = 0.0
        total_acc = 0.0
        for x_batch, y_batch in batch_eval:
            batch_len = len(x_batch)
            feed_dict = self.feed_data(x_batch, y_batch, 1.0)
            loss, acc = sess.run([self.model.loss, self.model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len

        return total_loss / data_len, total_acc / data_len

    def feed_data(self, x_batch, y_batch, keep_prob):
        feed_dict = {
            self.model.input_x: x_batch,
            self.model.input_y: y_batch,
            self.model.keep_prob: keep_prob
        }
        return feed_dict


def train(model_name, ctg_list, data_balance=False):
    from train_word2vec import Tokenizer, Word2vecTrainer
    test_model_name = model_name if model_name else 'trouble-dealer-accessory'
    # test_model_name = 'trouble-dealer-accessory'

    if not os.path.exists('./data/' + test_model_name):
        os.mkdir('./data/' + test_model_name)

    # 1. 定义分词器
    # TODO 读取词典的操作需要自己定义
    user_dict_path = './train_data/user_dict.txt'
    stop_words_path = './train_data/stopwords.txt'

    t = Tokenizer(use_single_char=False, user_dict_path=user_dict_path, stop_words_path=stop_words_path)

    # 2. 获取原始句子语料
    # TODO 读取训练数据，此处需要自己定义
    # ctg_list = ['accessory', 'trouble', 'dealer', 'other']
    # ctg_list = ['accessory', 'trouble', 'dealer']
    train_data_dir = './train_data/'
    w2v_train_data_list = []
    cnn_train_data = {}
    for ctg in ctg_list:
        cnn_train_data[ctg] = []
        corpus_file_path = train_data_dir + ctg + '.txt'
        with open(corpus_file_path, 'r', encoding='utf-8') as train_data_f:
            for corpus in train_data_f.readlines():
                corpus = corpus.replace('\n', '')
                if corpus != '':
                    w2v_train_data_list.append(corpus.replace('\n', ''))
                    cnn_train_data[ctg].append(corpus.replace('\n', ''))

    cnn_train_data_list = []
    if data_balance:
        max_len = 0
        # 计算最大长度
        for corpus_list in cnn_train_data.values():
            if max_len < len(corpus_list):
                max_len = len(corpus_list)

        # 将不满足的随机复制
        for ctg, corpus_list in cnn_train_data.items():
            delta = max_len - len(corpus_list)
            copy_times = int(delta / len(corpus_list))

            new_corpus_list = []
            for index in range(0, copy_times - 1):
                new_corpus_list.extend(corpus_list)
            random_count = delta % len(corpus_list)
            if random_count > 0:
                random_corpus_list = random.sample(corpus_list, random_count)
                new_corpus_list.extend(random_corpus_list)

            for corpus in new_corpus_list:
                cnn_train_data_list.append(corpus.replace('\n', '') + '\t' + ctg)
    else:
        # 直接导入
        for ctg, corpus_list in cnn_train_data.items():
            for corpus in corpus_list:
                cnn_train_data_list.append(corpus.replace('\n', '')+'\t'+ctg)

    # 3. 训练词向量
    w2v_train_conf = Word2VecTrainConfig()
    w2v_train_conf.word2vec_save_path = './data/' + test_model_name + '/word2vec.txt'

    w2v_dal = W2VDAL()
    w2v_train_conf.use_tencent = True
    w2v_train_conf.tencent_word2vec_dal = w2v_dal

    w2v_trainer = Word2vecTrainer(config=w2v_train_conf, tokenizer=t)
    w2v_trainer.train(train_data=w2v_train_data_list, model_name=test_model_name)

    # 4. 训练cnn模型参数
    cnn_train_conf = TextConfig()

    # 将词典写入对应的文件目录以供恢复
    user_dict_save_path = './data/' + model_name + '/' + model_name + cnn_train_conf.user_dict_filename
    with open(user_dict_path, 'r', encoding='utf-8') as input_f, \
            open(user_dict_save_path, 'w', encoding='utf-8') as output_f:
        output_f.write(input_f.read())
    stop_words_save_path = './data/' + model_name + '/' + model_name + cnn_train_conf.stop_word_filename
    with open(stop_words_path, 'r', encoding='utf-8') as input_f, \
            open(stop_words_save_path, 'w', encoding='utf-8') as output_f:
        output_f.write(input_f.read())

    # todo 无字向量测试
    cnn_train_conf.word2vec_file_path = w2v_train_conf.word2vec_save_path

    # TODO tf内部替换词向量
    # cnn_trainer = TextCNNTrainer(config=cnn_train_conf, tokenizer=t)
    # train_result_iter = cnn_trainer.train(train_data=cnn_train_data_list, model_name=test_model_name)

    # TODO tf外部替换词向量
    from embedding_loader import EmbeddingLoader
    embedding_loader = EmbeddingLoader(w2v_train_conf.word2vec_save_path, w2v_dal=w2v_dal)
    cnn_train_conf.embedding_loader = embedding_loader
    cnn_trainer = TextCNNTrainer(config=cnn_train_conf, tokenizer=t)
    train_result_iter = cnn_trainer.train(train_data=cnn_train_data_list, model_name=test_model_name)

    while True:
        try:
            next(train_result_iter)
        except StopIteration:
            print("train done")
            print("model update success!")
            break
        except Exception as e:
            print("train init:", traceback.print_exc())
            print(e)
            break


if __name__ == '__main__':
    # from concurrent.futures import ThreadPoolExecutor
    #
    # executor = ThreadPoolExecutor(3)
    # ctg_list_list = [['accessory', 'not_accessory'], ['dealer', 'not_dealer'], ['trouble', 'not_trouble']]
    # model_name_list = ['accessory', 'dealer', 'trouble']
    # for i, ctg_list in enumerate(ctg_list_list):
    #     executor.submit(train, model_name_list[i], ctg_list)
    # train('accessory', ['accessory', 'not_accessory'])
    # train('trouble', ['trouble', 'not_trouble'])
    # train('dealer', ['dealer', 'not_dealer'])
	train('Salary', ['payment_ratio', 'payment_base', 'payment_method', 'transact_process', 'consequences', 'transfer_method', 'query_method', 'calculation_method', 'introduction'])