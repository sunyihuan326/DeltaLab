# coding:utf-8
'''
Created on 2017/12/6.

@author: chk01
'''
import scipy.io as scio
from collections import Counter
import csv

res = scio.loadmat('res.mat')['result']
error = scio.loadmat('error.mat')['result'][0]


def error_result():
    print('sample_num：=========', error[9])
    print('error_rate:==========', round(sum(error[:9]) / error[9], 2))
    for i in range(3):
        print('error_sample_{}'.format(i), '错误总占比===', round(100 * error[i] / error[9]), '%')
        print('error_sample_{}'.format(i), '错误占个比===', round(100 * error[i] / error[i - 9]), '%')
        print('---------------==================---------------------------------------')


def main():
    data = {
        '0': [], '3': [], '6': [],
        '1': [], '4': [], '7': [],
        '2': [], '5': [], '8': []
    }
    for i in range(len(res)):
        pre, cor = res[i]
        data[str(int(cor))].append(int(pre))
    for key in data:
        num = len(data[key])
        with open('{}-result.csv'.format(key), 'w') as csv_file:
            writer2 = csv.writer(csv_file)
            _counter = Counter(data[key])
            writer2.writerow(['X', '曲', '中', '直'])
            for i in range(3):
                writer2.writerow(
                    [['小', '中', '大'][i], round(100 * _counter[3 * i] / num, 2),
                     round(100 * _counter[3 * i + 1] / num, 2),
                     round(100 * _counter[3 * i + 2] / num, 2)])


if __name__ == '__main__':
    error_result()
