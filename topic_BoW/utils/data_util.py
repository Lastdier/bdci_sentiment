# coding: utf-8
# created by deng on 7/24/2018

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from pandas import read_csv
from time import time
from os.path import exists

from utils.path_util import from_project_root


def sentence_to_ngram(sentence, max_n=1):
    """ transform sentence to ngram list

    Args:
        sentence: sentence to be transformed
        max_n: max_n for ngram

    Returns:
        list: ngram list

    """
    words = sentence.split()
    ngrams = list()
    for n in range(1, max_n + 1):  # add ngram
        for i in range(0, len(words) - n + 1):
            ngrams.append(' '.join(words[i:i + n]))
    return ngrams


def generate_level_data(train_url, header=False, index=False):
    """ generate phrase level data and word level data

    Args:
        train_url: original train file url
        header: True to contain header in generated file
        index: True to contain index in generated file

    """
    train_df = load_to_df(train_url)
    phrase_level_url = from_project_root("processed_data/phrase_level_data.csv")
    train_df[['class', 'word_seg']].to_csv(phrase_level_url, header=header, index=index)
    word_level_url = from_project_root("processed_data/word_level_data.csv")
    train_df[['class', 'article']].to_csv(word_level_url, header=header, index=index)


def load_raw_data(data_url, ngram=1):
    """ load data to get labels list and sentences list, set ngram=None if you
        want every sentence to be a space separated string instead of ngram list

    Args:
        data_url: url to data file
        ngram: generate ngram in sentence

    Returns:
        (list, list): labels and sentences

    """
    if not exists(data_url):
        generate_level_data(from_project_root("data/train_set.csv"))

    with open(data_url, "r", encoding="utf-8") as data_file:
        labels = list()
        sentences = list()
        print("loading data from \n ", data_url)
        s_time = time()
        for line in data_file:
            line = line.split(',')
            labels.append(int(line[0]))
            if ngram is not None:
                sentences.append(sentence_to_ngram(line[1], ngram))
            else:
                sentences.append(line[1])
        e_time = time()
        print("finished loading in %.3f seconds\n" % (e_time - s_time))
    return labels, sentences


def load_to_df(data_url, save=False):
    """ load csv data with header into pandas DataFrame

    Args:
        data_url: str, url to csv data file
        save: boolean, save loaded df to pickle

    Returns:
        df, loaded DataFrame

    """
    pk_url = data_url.replace('.csv', '_df.pk')
    if exists(pk_url):
        return joblib.load(pk_url)
    data_pk = read_csv(data_url)
    if save:
        joblib.dump(data_pk, pk_url)
    return data_pk


def train_dev_split(data_url, dev_size=0.2, including_header=False):
    """ split .csv data into train data and dev data and save them in the same dir

    Args:
        data_url: url to original data
        dev_size: the ratio of dev_data in data
        including_header: if the data file including header

    """
    print("splitting data file %s" % data_url)
    train_url = data_url[:-4] + '_train.csv'
    dev_url = data_url[:-4] + '_dev.csv'
    with open(data_url, "r", encoding='utf-8') as data_file, \
            open(train_url, "w", encoding='utf-8', newline='\n') as train_file, \
            open(dev_url, "w", encoding='utf-8', newline='\n') as dev_file:
        lines = data_file.readlines()
        if including_header:
            lines = lines[1:]

        # use specific random_state to generate the same data all the time
        train_lines, dev_lines = train_test_split(lines, test_size=dev_size, random_state=233)
        train_file.writelines(train_lines)
        dev_file.writelines(dev_lines)
        print("finished splitting data(%d samples) into train_data(%d samples) and dev_data(%d samples)"
              % (len(lines), len(train_lines), len(dev_lines)))


def main():
    data_url = from_project_root("data/train_set.csv")
    # train_data_url = from_project_root("processed_data/phrase_level_data_train.csv")
    # labels, sentences = load_raw_data(train_data_url)
    pass


if __name__ == '__main__':
    main()
