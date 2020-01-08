import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding


## Session configuration神经网络训练和测试的入口，通过定义主函数的执行函数，可以在这里控制神经网络的训练，保存模型；以及神经网络的测试，包括模型调用
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()



## hyperparameters
embedding_dim = 128

tag2label = {"N": 0,
             "解剖部位": 1, "手术": 2,
             "药物": 3, "独立症状": 4,
             "症状描述": 5}
## get char embeddings
word2id = read_dictionary('./vocab.pkl')
embeddings = random_embedding(word2id, embedding_dim)

train_data = read_corpus('./c.txt')


# embeddings, tag2label, vocab,batch_size,epoch,hidden_dim,CRF,update_embedding,shuffle
## training model
if __name__ == '__main__':
    model = BiLSTM_CRF(embeddings, tag2label, word2id, 4,80,128,False,True,True)
    model.build_graph()
    test_report = open('test_report.txt','w',encoding= 'utf-8')

    print("train data: {}".format(len(train_data)))
    model.test(test_report)
    # model.train(train=train_data)  # use test_data as the dev_data to see overfitting phenomena



