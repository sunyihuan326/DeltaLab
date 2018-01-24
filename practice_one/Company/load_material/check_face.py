# coding:utf-8
'''
Created on 2017/11/24.

@author: chk01
'''
from PIL import Image, ImageDraw
import numpy as np
from aip import AipFace

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

ss = [
    {
        "x": 317,
        "y": 445
    },
    {
        "x": 339,
        "y": 457
    },
    {
        "x": 363,
        "y": 462
    },
    {
        "x": 387,
        "y": 460
    },
    {
        "x": 410,
        "y": 453
    },
    {
        "x": 392,
        "y": 434
    },
    {
        "x": 366,
        "y": 427
    },
    {
        "x": 340,
        "y": 430
    }
]
ss2 = [
    {
        "x": 286,
        "y": 789
    },
    {
        "x": 310,
        "y": 816
    },
    {
        "x": 345,
        "y": 829
    },
    {
        "x": 382,
        "y": 825
    },
    {
        "x": 415,
        "y": 809
    },
    {
        "x": 392,
        "y": 779
    },
    {
        "x": 356,
        "y": 766
    },
    {
        "x": 318,
        "y": 768
    }
]

ss3 = [
    {
        "x": 186.23431396484,
        "y": 781.07836914062
    },
    {
        "x": 190.94956970215,
        "y": 887.44683837891
    },
    {
        "x": 208.82035827637,
        "y": 991.51239013672
    },
    {
        "x": 243.88159179688,
        "y": 1094.6579589844
    },
    {
        "x": 313.32250976562,
        "y": 1196.421875
    },
    {
        "x": 409.95184326172,
        "y": 1297.3879394531
    },
    {
        "x": 516.76049804688,
        "y": 1338.5772705078
    },
    {
        "x": 621.95092773438,
        "y": 1297.9616699219
    },
    {
        "x": 715.58197021484,
        "y": 1201.5905761719
    },
    {
        "x": 793.73162841797,
        "y": 1094.5134277344
    },
    {
        "x": 828.79833984375,
        "y": 985.90710449219
    },
    {
        "x": 842.79571533203,
        "y": 877.84558105469
    },
    {
        "x": 843.07263183594,
        "y": 770.02746582031
    },
    {
        "x": 286.52532958984,
        "y": 805.34545898438
    },
    {
        "x": 320.53109741211,
        "y": 783.14282226562
    },
    {
        "x": 358.26962280273,
        "y": 777.30969238281
    },
    {
        "x": 394.8955078125,
        "y": 788.79803466797
    },
    {
        "x": 422.55267333984,
        "y": 825.12365722656
    },
    {
        "x": 389.71365356445,
        "y": 827.28100585938
    },
    {
        "x": 351.80728149414,
        "y": 828.44097900391
    },
    {
        "x": 314.88143920898,
        "y": 818.96722412109
    },
    {
        "x": 364.38122558594,
        "y": 800.18713378906
    },
    {
        "x": 237.51943969727,
        "y": 716.79931640625
    },
    {
        "x": 282.92471313477,
        "y": 678.19982910156
    },
    {
        "x": 335.52688598633,
        "y": 671.83746337891
    },
    {
        "x": 386.32290649414,
        "y": 678.72845458984
    },
    {
        "x": 437.19183349609,
        "y": 712.87805175781
    },
    {
        "x": 384.02517700195,
        "y": 712.16192626953
    },
    {
        "x": 334.25109863281,
        "y": 709.26263427734
    },
    {
        "x": 284.30749511719,
        "y": 710.93035888672
    },
    {
        "x": 620.11322021484,
        "y": 821.71575927734
    },
    {
        "x": 648.14739990234,
        "y": 783.712890625
    },
    {
        "x": 686.39086914062,
        "y": 769.70770263672
    },
    {
        "x": 723.56109619141,
        "y": 773.60021972656
    },
    {
        "x": 755.83453369141,
        "y": 795.86096191406
    },
    {
        "x": 729.78723144531,
        "y": 812.77380371094
    },
    {
        "x": 694.08258056641,
        "y": 823.05749511719
    },
    {
        "x": 654.74255371094,
        "y": 823.45556640625
    },
    {
        "x": 681.13751220703,
        "y": 795.35485839844
    },
    {
        "x": 610.16479492188,
        "y": 706.83001708984
    },
    {
        "x": 655.36242675781,
        "y": 668.86791992188
    },
    {
        "x": 707.32287597656,
        "y": 657.13110351562
    },
    {
        "x": 758.94952392578,
        "y": 662.06146240234
    },
    {
        "x": 802.53216552734,
        "y": 702.78668212891
    },
    {
        "x": 756.97711181641,
        "y": 695.93054199219
    },
    {
        "x": 709.44671630859,
        "y": 696.11791992188
    },
    {
        "x": 661.86920166016,
        "y": 702.95770263672
    },
    {
        "x": 475.40063476562,
        "y": 826.98278808594
    },
    {
        "x": 463.40194702148,
        "y": 898.06896972656
    },
    {
        "x": 450.03332519531,
        "y": 970.06182861328
    },
    {
        "x": 427.25665283203,
        "y": 1038.3950195312
    },
    {
        "x": 473.82809448242,
        "y": 1046.5842285156
    },
    {
        "x": 574.96978759766,
        "y": 1047.0627441406
    },
    {
        "x": 617.94256591797,
        "y": 1038.7296142578
    },
    {
        "x": 595.13372802734,
        "y": 967.63861083984
    },
    {
        "x": 581.03271484375,
        "y": 897.25793457031
    },
    {
        "x": 566.90087890625,
        "y": 826.42340087891
    },
    {
        "x": 528.08703613281,
        "y": 1012.2965087891
    },
    {
        "x": 384.93768310547,
        "y": 1140.6510009766
    },
    {
        "x": 459.82809448242,
        "y": 1141.9265136719
    },
    {
        "x": 523.57092285156,
        "y": 1146.5535888672
    },
    {
        "x": 585.32489013672,
        "y": 1142.5402832031
    },
    {
        "x": 650.22100830078,
        "y": 1137.3419189453
    },
    {
        "x": 596.28155517578,
        "y": 1197.9930419922
    },
    {
        "x": 520.94976806641,
        "y": 1221.6053466797
    },
    {
        "x": 443.03750610352,
        "y": 1199.9367675781
    },
    {
        "x": 457.34335327148,
        "y": 1160.6879882812
    },
    {
        "x": 522.53918457031,
        "y": 1170.7669677734
    },
    {
        "x": 584.98773193359,
        "y": 1159.9285888672
    },
    {
        "x": 585.49279785156,
        "y": 1168.8901367188
    },
    {
        "x": 520.97106933594,
        "y": 1178.7885742188
    },
    {
        "x": 455.46667480469,
        "y": 1169.7445068359
    }
]


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def get_landmark72(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark"
    }
    result = client.detect(get_file_content(full_path), options=options)
    landmark72 = landmark72_trans(result['result'][0]['landmark72'])
    return landmark72, result['result'][0]['rotation_angle']


# chin 13# data1 = data[:13]
# eyes# data2 = data[13:22]# data2.extend(data[30:39])
# browns# data3 = data[22:30]# data3.extend(data[39:47])
# nose# data4 = data[47:58]
# mouse# data5 = data[58:]
def chin_check():
    import os
    test_dir = 'C:/Users/chk01/Desktop/chin_check'
    for file in os.listdir(test_dir):
        if file.endswith('jpg'):
            print(file, 'start')
            full_path = test_dir + '/' + file
            im = Image.open(full_path)
            landmark72, angle = get_landmark72(full_path)
            if -30 < angle < 30:
                pass
            else:
                im = im.rotate(angle, expand=1)
                im.show()
                im.save(full_path)
                im = Image.open(full_path)
                landmark72, angle = get_landmark72(full_path)

            drawSurface = ImageDraw.Draw(im)
            landmark72 = tuple(tuple(t) for t in landmark72)
            drawSurface.line(landmark72[:13], fill=255, width=3)
            im.save(full_path)
            print(file, 'end')
            print('----------------')


if __name__ == '__main__':
    # chin_check()
    file = 'check/8.png'
    # org = 'lip'
    # for i in range(20):
    #     # file = 'C:/Users/chk01/Desktop/Delta/image/check/src/{}/{}.jpg'.format(org, i + 1)
    im = Image.open(file)

    wid, hei = im.size
    landmark72, angle = get_landmark72(file)

    print(angle)
    if -10 < angle < 10:
        pass
    else:
        # angle = angle / 180 * math.pi
        im = im.rotate(angle, expand=1)
        im.show()
        im.save(file)
        im = Image.open(file)
        landmark72, angle = get_landmark72(file)
        print(angle)
        # tran_matrix = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        # landmark72 = np.matmul(landmark72, tran_matrix)
    # landmark72 = landmark72_trans(ss3)
    drawSurface = ImageDraw.Draw(im)
    landmark72 = tuple(tuple(t) for t in landmark72)
    drawSurface.line(landmark72[:13], fill=255, width=1)
    # drawSurface.line(landmark72, fill=255, width=3)
    # drawSurface.line(landmark72[13:21], fill=255, width=3)
    drawSurface.line([landmark72[13], landmark72[20]], fill=255, width=3)
    #     # drawSurface.line(landmark72[14], fill=100, width=10)
    #     # drawSurface.line(landmark72[30:39], fill=255, width=10)
    #     # drawSurface.line(landmark72[34], fill=100, width=10)
    #
    drawSurface.line(landmark72[22:30], fill=255, width=1)
    drawSurface.line([landmark72[22], landmark72[29]], fill=255, width=1)
    #     # drawSurface.line(landmark72[39:47], fill=255, width=10)
    point1 = [landmark72[58], landmark72[59], landmark72[60], landmark72[61], landmark72[62],
              landmark72[68], landmark72[67], landmark72[66], landmark72[58]]
    drawSurface.line(point1, fill=255, width=3)
    #     #
    #     # point2 = [landmark72[58], landmark72[65], landmark72[64], landmark72[63], landmark72[62],
    #     #           landmark72[69], landmark72[70], landmark72[71], landmark72[58]]
    #     # drawSurface.line(point2, fill=255, width=3)
    #
    drawSurface.line(landmark72[49:55], fill=255, width=1)
    #     # drawSurface.line(landmark72[58:66], fill=255, width=3)
    #     # drawSurface.line([landmark72[58], landmark72[65]], fill=255, width=3)
    drawSurface.line(landmark72[69:], fill=255, width=10)
    #     # drawSurface.line(landmark72[66:69], fill=255, width=1)
    #     # drawSurface.point(landmark72, fill=0)
    #     # im.save(file.replace('src', 'cartoon'))
    im.show()
    im.save('res.png')
