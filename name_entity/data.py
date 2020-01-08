import sys, pickle, os, random
import numpy as np

## 名体--lable, 转为TF输入，中文转化为INDEX，定义输入类，保证TF输入可以动态输入
tag2label = {"N": 0,
             "解剖部位": 1, "手术": 2,
             "药物": 3, "独立症状": 4,
             "症状描述": 5}

sopwords = [',','.','<','>',';','"','"','!','%','*','&','?','/','。','，','(',')','（','）','【','】','[',']','{','}','+','=','-']
print('命名体转化为Lable成功')
def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            # print(line)
            tmp = line.strip().split(' ')
            if (len(tmp)>1):
                char = tmp[0]
                label = tmp[1]
                sent_.append(char)
                tag_.append(label)
                # data.append((sent_, tag_))
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data




def vocab_build(vocab_path, corpus_path):
    """

    :param vocab_path:
    :param corpus_path:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    word2id['< UNK >'] = 0
    for sent_, tag_ in data:
        for word in sent_:
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)



def sentence2id(sent, word2id):
    """
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab)+1, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):

    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

if __name__ == "__main__":
    data = read_corpus('./c.txt')
    # print(data)
    vocab = read_dictionary('./vocab.pkl')
    batches = batch_yield(data, 4, vocab, tag2label, shuffle=True)
    for step, (seqs, labels) in enumerate(batches):
        if step < 3:
            print(seqs)
            print(labels)
            word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0)
            print('seq_len_list',seq_len_list)
            labels_, _ = pad_sequences(labels, pad_mark=0)
            print('labels_',labels_)


    #         seq_list, seq_len_list = pad_sequences(seqs, pad_mark=0)
    #         print(seq_list)
    #         print(len(seq_list[0]))
    #         print(len(seq_list[1]))
    #         print(len(seq_list[2]))
    #         print(len(seq_list[3]))
    #         print(seq_len_list)
    #         # print(step)
    #         # # print(seqs)
    #         # # print(labels)
    #         # print(len(seqs[0]))
    #         # print(len(labels[0]))
    #         # print(len(seqs[1]))
    #         # print(len(labels[1]))
    #         # print(len(seqs[2]))
    #         # print(len(labels[2]))
    #         # print(len(seqs[3]))
    #         # print(len(labels[3]))
    #         print('%%%%%%%%%%%%%%%%%%%%%%%%%')

