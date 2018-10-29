# coding:utf-8
'''
Created on 2017/12/4.

@author: chk01
'''
from shuwei_fengge.practice_two.load_data.utils import *


def read_point():
    # 根据实际数量修改num,dir
    dir = 'F:/dataSets/LFPW/testset/'
    num = 240
    for i in range(1, num):
        file_name = 'image_{}.pts'.format(num_change(i))
        print(file_name, 'loding---->------>')
        try:
            openFileHandle = open(dir + file_name, 'r')
        except:
            continue
        j = 0
        tt = []

        while True:
            line = openFileHandle.readline()
            j += 1
            if j > 3:
                if line:
                    point = line.replace('\n', '').split(' ')
                    if len(point) == 2:
                        tt.append(np.array(point).astype('float'))
                else:
                    openFileHandle.close()
                    break
        try:
            assert np.array(tt).shape == (68, 2)
        except:
            print(file_name, 'loding---->------>', 'Failed')
            continue
        scio.savemat(dir + file_name.replace('pts', 'mat'), {'landmarks72': np.array(tt)})
        print(file_name, 'loding---->------>', 'OK')
