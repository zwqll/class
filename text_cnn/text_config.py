class TextConfig:
    # embedding_size = 256      # dimension of word embedding 词向量维度
    embedding_size = 200        # 腾讯词向量的维度是200维
    embedding_as_input = True   # 是否在进入tf图之前替换向量
    embedding_loader = None     # 词向量加载对象，需要手动指定

    vocab_size = 5000           # number of vocabulary 词典词个数
    pre_training = None         # use vector_char trained by word2vec 创建对象时赋值为word2vec预训练词向量

    split_ratio = 0.3           # 验证集占整体数据的比例，在0到1之间
    shuffle = True              # 是否打乱训练集

    seq_length = 20           # max length of sentence 句子长度
    # seq_length = 50             # 加入单字向量的话句子长度会变长
    num_classes = 6             # number of labels 分类个数

    num_filters = 64           # number of convolution kernel 过滤器数量
    filter_sizes = [2, 3, 4]    # size of convolution kernel 过滤器尺寸

    keep_prob = 0.5             # dropout 概率
    lr = 1e-3                   # learning rate 学习率
    lr_decay = 0.9              # learning rate decay 学习率衰减率
    clip = 6.0                  # gradient clipping threshold
    l2_reg_lambda = 0.01        # l2 regularization lambda

    num_epochs = 100            # epochs 训练轮数
    batch_size = 4             # batch_size 每批数据个数（不同分类训练时数值不同）
    print_per_batch = 100        # print result 每?批打印一次信息
    # batch_size = 5            # batch_size 每批数据个数（不同分类训练时数值不同）
    # print_per_batch = 5        # print result 每?批打印一次信息

    require_improvement = 50000  # 最少提升轮数（超过这么多轮没有提升就停止训练）

    base_data_path = './data/'                    # data相对路径
    train_filename = '-train.txt'                 # train data
    test_filename = '-test.txt'                   # test data
    val_filename = '-val.txt'                     # validation data

    user_dict_filename = '-user_dict.txt'         # 用户词典文件名
    stop_word_filename = '-stop_word.txt'         # 停用词文件名
    word2id_filename = '-word2id.pkl'             # save word2id to pickle file
    ctg2id_filename = '-ctg2id.pkl'               # save ctg2id to pickle file

    word2vec_file_path = ''                       # 词向量路径

    tensor_board_dir = './tensor_board'
    model_save_dir = './checkpoints'


class Word2VecTrainConfig:
    dim = 200                       # 词向量维度
    sg = 0                          # 训练使用skip-gram还是CBOW，0为CBOW
    use_tencent = False             # 是否使用腾讯词向量
    tencent_word2vec_dal = None     # 腾讯词向量获取操作类
    word2vec_save_path = ''         # 词向量文件存储路径
