# coding:utf-8 
'''
created on 2018/5/14

@author:sunyihuan
'''

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib.request
import tensorflow as tf

url = "http://mattmahoney.net/dc/"


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:
        print("Found and verfied", filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "Failed to verfify " + filename + ". Can you get to it with a browser?"
        )
    return filename


filename = maybe_download("text8.zip", 31344016)


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print("Data size", len(words))

vocabulary_size = 50000


def build_datase(words):
    counts = [["UNK", -1]]
    counts.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in counts:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    counts[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counts, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_datase(words)

del words
print("Most common words (+UNK)", count[:5])
print("Sample data", data[:10], [reverse_dictionary[i] for i in data[:10]])
