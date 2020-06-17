import tensorflow as tf
import heapq
import threading
import os
from text_config import TextConfig
from loader import *
from tokenizer import Tokenizer
from w2v_dal import W2VDAL


class PredictorFactory:
    _instance_lock = threading.Lock()

    def __init__(self):
        self.predictor_dict = {}

    def __new__(cls, *args, **kwargs):
        if not hasattr(PredictorFactory, "_instance"):
            with PredictorFactory._instance_lock:
                if not hasattr(PredictorFactory, "_instance"):
                    PredictorFactory._instance = object.__new__(cls)
        return PredictorFactory._instance

    def get_predictor(self, model_name):
        if model_name in self.predictor_dict.keys():
            return self.predictor_dict[model_name]
        else:
            self.predictor_dict[model_name] = Predictor(model_name)
            return self.predictor_dict[model_name]


class Predictor:
    def __init__(self, model_name):
        # 恢复中间文件
        self.config = TextConfig()
        prev = './data/' + model_name + '/' + model_name
        try:
            with open(prev + self.config.word2id_filename, "rb") as f:
                self.word2id = pickle.load(f)
            with open(prev + self.config.ctg2id_filename, "rb") as f:
                ctg2id = pickle.load(f)
                self.id2ctg = {v: k for k, v in ctg2id.items()}
        except Exception as e:
            print(e)

        # 加载各种词典
        user_dict_path = prev + self.config.user_dict_filename
        stop_word_path = prev + self.config.stop_word_filename
        self.tokenizer = Tokenizer(user_dict_path=user_dict_path,
                                   stop_words_path=stop_word_path,
                                   use_single_char=False)

        if self.config.embedding_as_input:
            from embedding_loader import EmbeddingLoader
            word2vec_save_path = './data/' + model_name + '/word2vec.txt'

            w2v_dal = W2VDAL()
            embedding_loader = EmbeddingLoader(word2vec_save_path, w2v_dal=w2v_dal)
            self.config.embedding_loader = embedding_loader

        # 恢复模型
        save_path = './checkpoints/' + model_name + '/' + model_name
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        tf_config.gpu_options.allow_growth = True
        self.session = tf.Session(config=tf_config)
        self.session.run(tf.global_variables_initializer())

        if os.path.exists(save_path + '.meta'):
            saver = tf.train.import_meta_graph(save_path + '.meta')
            saver.restore(self.session, save_path)
        else:
            raise RuntimeError("model file does not exist")
        self.model_name = model_name

    def predict(self, sentence, top=1):
        # 输入预处理
        if self.config.embedding_as_input:
            input_x = process_with_embedding(sentence,
                                             cat_to_id=None,
                                             max_length=self.config.seq_length,
                                             tokenizer=self.tokenizer,
                                             is_train=False,
                                             embedding_loader=self.config.embedding_loader)
        else:
            input_x = process_for_predict(sentence, self.word2id,
                                          max_length=self.config.seq_length, tokenizer=self.tokenizer)

        model_input = self.session.graph.get_tensor_by_name('input_x:0')
        dropout = self.session.graph.get_tensor_by_name('dropout:0')
        prob = self.session.graph.get_tensor_by_name('output/Softmax:0')

        feed_dict = {
            model_input: input_x,
            dropout: 1,
        }

        y_prob = self.session.run(prob, feed_dict=feed_dict)
        y_prob = y_prob.tolist()
        cat = []
        for prob in y_prob:
            # 列出排序最高的
            top_n_predict = list(map(prob.index, heapq.nlargest(top, prob)))
            top_n_acc = heapq.nlargest(top, prob)
            for i in range(0, top):
                cat.append(
                    {
                        'ctg_id': self.id2ctg[top_n_predict[i]],
                        'acc': top_n_acc[i]
                    })

        # TODO 应不应该在此reset
        tf.reset_default_graph()
        result = {
            'predict_list': cat,
        }
        return result


predictor_factory = PredictorFactory()


def predict_bi(sentence, top=1):
    model_list = ['accessory', 'dealer', 'trouble']
    for model in model_list:
        predictor = predictor_factory.get_predictor(model)
        predict_result = predictor.predict(sentence, top=top)
        print(sentence, ':', predict_result['predict_list'][0])
        if predict_result['predict_list'][0]['ctg_id'] == model:
            return model
    return 'unknown'


if __name__ == '__main__':
    # model_ = 'trouble-dealer-accessory'
    model_ = 'Salary'
    predictor_factory_ = PredictorFactory()
    predictor_ = predictor_factory_.get_predictor(model_)
    # print(predictor_.predict("2014社会保险一个月要交多少钱"))
    with open('test.txt','r',encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            line=line.replace('\r','').replace('\n','').split(' ')
            # if line:
                # print(line,predictor_.predict(line))
            # else:
                # print("*")
            r = predictor_.predict(line[1])
            if r['predict_list'][0]['ctg_id'] == line[0]:
                i+=1
        print(i/8983)