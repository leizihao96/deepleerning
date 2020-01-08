import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from test_data import  batch_yield_test,read_corpus_test
from eval import conlleval

learning_rate = 0.001
decay_rate = 0.8

class BiLSTM_CRF(object):
    def __init__(self, embeddings, tag2label, vocab,batch_size,epoch,hidden_dim,CRF,update_embedding,shuffle):
        self.batch_size = batch_size
        self.epoch_num = epoch
        self.hidden_dim = hidden_dim
        self.embeddings = embeddings
        self.CRF = CRF
        self.update_embedding = update_embedding
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = shuffle
    print('开始构建神经网络')
    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        #这里有一个可以借鉴的地方时，在定义input的placehoder的时候可以直接都是None，这样在输入的时候可以让每个batch_size的词长度不一样，但要保证
        #一个batch——size的样本的词长度一样
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
        self.lr_pl = tf.Variable(0.0, trainable=False)
    print('开始训练模型')
    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings =  word_embeddings

    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)

            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            print('output shape is :',output.get_shape())
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            print('pred shape',pred.get_shape())

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])
            print('self.logits',self.logits.get_shape())

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)

            self.loss = -tf.reduce_mean(log_likelihood)
            #1CRF

        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            print('self.labels', self.labels.get_shape())
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl).minimize(self.loss)
            self.train_op = optim

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):

        self.merged = tf.summary.merge_all()


    def train(self, train):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            # sess.run(self.init_op)
            saver = tf.train.import_meta_graph('./model_first/ner.model.meta')
            saver.restore(sess, './model_first/ner.model')
            writer = tf.summary.FileWriter("./train_gan", sess.graph)
            # self.add_summary(sess)

            for epoch in range(self.epoch_num):
                sess.run(tf.assign(self.lr_pl, learning_rate * (decay_rate ** epoch)))
                self.run_one_epoch(sess, train, epoch, saver,writer)
            saver.save(sess, 'model_con/ner.model')


    def run_one_epoch(self, sess, train, epoch, saver,writer):

        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels)
            _, loss_train, step_num_ = sess.run([self.train_op, self.loss, self.global_step],
                                                         feed_dict=feed_dict)
            # writer.add_summary(summary, global_step=step_num)
            if step + 1 == 1 or (step + 1) % 10 == 0 or step + 1 == num_batches:
                print('epoch {}, step {}, loss: {:.4}, global_step: {}'.format(epoch + 1, step + 1,loss_train, step_num))

            # if step + 1 == num_batches:
            #     saver.save(ses, global_step=step_num)

    def get_feed_dict(self, seqs, labels):

        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)

        feed_dict = {self.word_ids: word_ids,self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        return feed_dict, seq_len_list

    def test(self,test_report):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,'model_first/ner.model')
            tag,samples = self.demo_one(sess)
            self.display(tag,samples,test_report)

    def display(self,tag,samples,test_report):
        for j in range(len(tag)):
            temp = tag[j]
            temp = list(temp)
            site = []
            surgery = []
            drug = []
            Independent = []
            symptom = []
            h = False
            #这里涉及一个算法，根据一个序列list 抽出其中值为指定值得序列，并保证相邻的挨在一起，这里是一个纯算法问题，
            for i ,x in enumerate(temp):
                if 1 == int(x):
                    if h:
                        if len(site) > 0:
                            site[-1].append(i)
                    else:
                        site.append([i])
                        h = True
                else:
                    h = False
            h = False
            for i, x in enumerate(temp):
                if 2 == int(x):
                    if h:
                        if len(surgery) > 0:
                            surgery[-1].append(i)
                    else:
                        surgery.append([i])
                        h = True
                else:
                    h = False
            h = False
            for i, x in enumerate(temp):
                if 3 == int(x):
                    if h:
                        if len(drug) > 0:
                            drug[-1].append(i)
                    else:
                        drug.append([i])
                        h = True
                else:
                    h = False
            h = False
            for i, x in enumerate(temp):
                if 4 == int(x):
                    if h:
                        if len(Independent) > 0:
                            Independent[-1].append(i)
                    else:
                        Independent.append([i])
                        h = True
                else:
                    h = False
            h = False
            for i, x in enumerate(temp):
                if 5 == int(x):
                    if h:
                        if len(symptom) > 0:
                            symptom[-1].append(i)
                    else:
                        symptom.append([i])
                        h = True
                else:
                    h = False
            # for i, x in enumerate(temp):
                # if 1 == int(x):
                #     if h:
                #         if len(site) > 0:
                #             site[-1].append(i)
                #     else:
                #         site.append([i])
                #         h = True
                # elif 2 == int(x):
                #     if h:
                #         if len(surgery) > 0:
                #             surgery[-1].append(i)
                #     else:
                #         surgery.append([i])
                #         h = True
                # elif 3 == int(x):
                #     if h:
                #         if len(drug) > 0:
                #             drug[-1].append(i)
                #     else:
                #         drug.append([i])
                #         h = True
                # if 4 == int(x):
                #     if h:
                #         if len(Independent) > 0:
                #             Independent[-1].append(i)
                #     else:
                #         Independent.append([i])
                #         h = True
                # elif 5 == int(x):
                #     if h:
                #         if len(symptom) > 0:
                #             symptom[-1].append(i)
                #     else:
                #         symptom.append([i])
                #         h = True
                # elif h:
                #     h = False
            print(symptom)
            print('site',site)
            site_ = []
            symptom_ = []
            Independent_ = []
            drug_ = []
            surgery_ = []
            for l in site:
                one = []
                for index in l:
                    one.append(samples[j][index])
                one = ''.join(one)
                site_.append(one)
            for l in symptom:
                one = []
                for index in l:
                    one.append(samples[j][index])
                one = ''.join(one)
                symptom_.append(one)
            for l in Independent:
                one = []
                for index in l:
                    one.append(samples[j][index])
                one = ''.join(one)
                Independent_.append(one)
            for l in drug:
                one = []
                for index in l:
                    one.append(samples[j][index])
                one = ''.join(one)
                drug_.append(one)
            for l in surgery:
                one = []
                for index in l:
                    one.append(samples[j][index])
                one = ''.join(one)
                surgery_.append(one)

            oneline = str(j+1) + ','
            order = {}
            for q in range(len(site_)):
                siteindex = site[q]
                if len(siteindex) > 1:
                    start_num = siteindex[0]
                    end_num = siteindex[-1]+1
                else:
                    start_num = siteindex[0]
                    end_num = start_num + 1

                w_site = site_[q] + '\t' + str(start_num) + '\t' + str(end_num) +  '\t' + '解剖部位' + ';'
                order[start_num] = w_site

            for q1 in range(len(surgery_)):
                surg_index = surgery[q1]
                if len(surg_index) > 1:
                    start_num = surg_index[0]
                    end_num = surg_index[-1]+1
                else:
                    start_num = surg_index[0]
                    end_num = start_num + 1
                w_surgery = surgery_[q1] + '\t' + str(start_num)+'\t'+str(end_num)+'\t'+'手术'+';'
                order[start_num] = w_surgery
            for q2 in range(len(drug_)):
                drugindex = drug[q2]
                if len(drugindex) > 1:
                    start_num = drugindex[0]
                    end_num = drugindex[-1]+1
                else:
                    start_num = drugindex[0]
                    end_num = start_num + 1
                w_drug = drug_[q2] + '\t' + str(start_num)+'\t'+str(end_num)+'\t'+'药物'+';'
                order[start_num] = w_drug
            for q3 in range(len(Independent_)):
                Independentindex = Independent[q3]
                if len(Independentindex) > 1:
                    start_num = Independentindex[0]
                    end_num = Independentindex[-1]+1
                else:
                    start_num = Independentindex[0]
                    end_num = start_num + 1
                w_Independent = Independent_[q3] +'\t' + str(start_num)+'\t'+str(end_num)+'\t'+'独立症状'+';'
                order[start_num] = w_Independent
            for q4 in range(len(symptom_)):
                symptomindex = symptom[q4]
                if len(symptomindex) > 1:
                    start_num = symptomindex[0]
                    end_num = symptomindex[-1]+1
                else:
                    start_num = symptomindex[0]
                    end_num = start_num + 1
                w_symptom = symptom_[q4] + '\t' + str(start_num)+'\t'+str(end_num)+'\t'+'症状描述'+';'
                order[start_num] = w_symptom
            index_order = []
            for index in order.keys():
                index_order.append(index)
            index_order.sort()
            test_report.write(oneline)
            for h in range(len(index_order)):
                test_report.write(order[index_order[h]])
            test_report.write('\n')

            print('第', j, '个测试样本为：', ''.join(samples[j]))
            print('识别出来的解剖部位为：',site_)
            print('识别出来的手术为：',surgery_)
            print('识别出来的药物为：',drug_)
            print('识别出来的独立症状为：',Independent_)
            print('识别出来的症状描述为：',symptom_)


    def demo_one(self, sess):
        label_list = []
        test_data = read_corpus_test('./testing_file1.txt')
        #这里指定测试文本，获取测试的TXT文本我是将其在eclipse里面的一个脚本转换过得
        for seqs,samples in batch_yield_test(test_data, 1, self.vocab):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)

        return label_list,samples


    def predict_one_batch(self, sess, seqs):

        feed_dict, seq_len_list = self.get_feed_dict(seqs,None)

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list
print('训练完成')



