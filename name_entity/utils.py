import logging, sys, argparse

#预处理
def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    site = get_site_entity(tag_seq, char_seq)
    symptom = get_symptom_entity(tag_seq, char_seq)
    Independent = get_Independent_entity(tag_seq, char_seq)
    drug = get_drug_entity(tag_seq, char_seq)
    surgery = get_surgery_entity(tag_seq, char_seq)
    return site, symptom, Independent,drug,surgery


def get_site_entity(tag_seq, char_seq):
    length = len(char_seq)
    site = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == '解剖部位':
            per = char
            if i+1 == length:
                site.append(per)

    return site


def get_symptom_entity(tag_seq, char_seq):
    length = len(char_seq)
    symptom = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == '症状描述':
            loc = char
            if i+1 == length:
                symptom.append(loc)
    return symptom


def get_Independent_entity(tag_seq, char_seq):
    length = len(char_seq)
    Independent = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == '独立症状':
            org = char
            if i+1 == length:
                Independent.append(org)

    return Independent


def get_drug_entity(tag_seq, char_seq):
    length = len(char_seq)
    drug = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == '药物':
            org = char
            if i+1 == length:
                drug.append(org)
    return drug

def get_surgery_entity(tag_seq, char_seq):
    length = len(char_seq)
    surgery = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == '手术':
            org = char
            if i+1 == length:
                surgery.append(org)
    return surgery
