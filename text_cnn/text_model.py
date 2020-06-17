import tensorflow as tf


class TextCNN(object):
    def __init__(self, config):
        # 一个服务训练多次时，由于网络结构没有初始化，会导致后续变量重名
        # 一种办法是每次都初始化图结构
        # 另一种方法是用特定id来命名图结构，如name='var' + uuid
        tf.reset_default_graph()

        self.config = config

        if config.embedding_as_input:
            self.input_x = tf.placeholder(
                tf.float32, shape=[None, self.config.seq_length, self.config.embedding_size], name='input_x')
        else:
            self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        # 当前迭代批数，每训练一个batch加1
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)

        # 指定运行设备 CPU不区分设备号，统一使用 /cpu:0
        with tf.device('/cpu:0'):
            if config.embedding_as_input:
                self.embedding_inputs = self.input_x
            else:
                self.embedding = tf.get_variable("embeddings",
                                                 shape=[self.config.vocab_size, self.config.embedding_size],
                                                 initializer=tf.constant_initializer(self.config.pre_training),
                                                 trainable=False)
                self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # 过滤器参数，长、宽、当前层深度、过滤器深度（下一层深度）
                    filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedding_inputs_expanded,
                        w,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.outputs = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 减轻过拟合  使输入tensor中某些元素变为0，其它没变0的元素变为原来的1/keep_prob大小
        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.outputs, self.keep_prob)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w', shape=[self.final_output.shape[1].value, self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.logits, 1, name='predictions')

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(cross_entropy) + self.config.l2_reg_lambda * self.l2_loss
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
