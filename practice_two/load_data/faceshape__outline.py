# coding:utf-8
'''
Created on 2017/12/12.

@author: chk01
'''
from practice_two.load_data.utils import *


def main():
    resList = fetch_source_data()
    data = {
        '0': {
            '0': 0,
            '1': 0,
            '2': 0
        },
        '1': {
            '0': 0,
            '1': 0,
            '2': 0
        },
        '2': {
            '0': 0,
            '1': 0,
            '2': 0
        },
        '3': {
            '0': 0,
            '1': 0,
            '2': 0
        },
        '4': {
            '0': 0,
            '1': 0,
            '2': 0
        }
    }
    for res in resList:
        outline = res['t_outline']
        faceshape = res['t_face']
        data[str(faceshape)][str(outline)] += 1
        # print(outline)
        # print(faceshape)
    print(data)


if __name__ == '__main__':
    main()
