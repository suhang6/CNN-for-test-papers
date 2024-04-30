import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cnn.reco as reco

#颜色阈值分割
def getred(strFilePath):
    img = cv2.imread(strFilePath)
    # 在彩色图像的情况下，解码图像将以b g r顺序存储通道。
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # H、S、V范围一：
    lower1 = np.array([0, 43, 46])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    lower2 = np.array([156, 43, 46])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2
    return mask3
# 投影法分割 输入灰度图像,rc=0为行向分割,rc=1为列分割
def pre_pic(img_gray, rc):
    # img为灰度图,rc01
    # 二值化
    t, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    rows, cols = binary.shape
    # print(rows,cols)
    if rc == 0:
        hor_list = [0] * rows
        for i in range(rows):
            for j in range(cols):
                # 统计每一行的白色像素总数
                if binary.item(i, j) > 122:
                    hor_list[i] = hor_list[i] + 1

    elif rc == 1:
        hor_list = [0] * cols
        for j in range(cols):
            for i in range(rows):
                # 统计每一列的白色像素总数
                if binary.item(i, j) > 122:
                    hor_list[j] = hor_list[j] + 1

    '''
    对hor_list中的元素进行筛选，可以去除一些噪点
    '''
    hor_arr = np.array(hor_list)
    hor_arr[np.where(hor_arr < 2)] = 0
    hor_list = hor_arr.tolist()

    img_white = np.ones(shape=(rows, cols), dtype=np.uint8) * 255
    #绘制水平投影
    if rc == 0:
        for i in range(rows):
            pt1 = (cols - 1, i)
            pt2 = (cols - 1 - hor_list[i], i)
            cv2.line(img_white, pt1, pt2, (0,), 1)
    # # 绘制垂直投影
    elif rc == 1:
        for j in range(cols):
            pt1 = (j, rows - 1)
            pt2 = (j, rows - 1 - hor_list[j])
            cv2.line(img_white, pt1, pt2, (0,), 1)
    plt.imshow(img_white)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去 x 轴刻度
    plt.yticks([])  # 去 y 轴刻度
    plt.show()

    # 取出list中像素存在的区间
    vv_list = list()
    v_list = list()
    for index, i in enumerate(hor_list):
        if i > 0:
            v_list.append(index)
        else:
            if v_list:
                vv_list.append(v_list)
                # list的clear与[]有区别
                v_list = []
    # 取出各个文字区间,存入图片列表
    img_ready = []
    for i in vv_list:
        if rc == 0:
            img_hor = img_gray[i[0]:i[-1], :]
        elif rc == 1:
            img_hor = img_gray[:, i[0]:i[-1]]
        img_ready.append(img_hor)  # 加入结果list
        # plt.imshow(img_hor)
        # plt.show()
    return img_ready


# 该函数用于保持形状的改变图像尺寸
def resize_keep(img, target_size):
    # img = cv2.imread(img_name) # 读取图片
    old_size = img.shape[0:2]  # 原始图像大小
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img, (new_size[1], new_size[0]))  # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1]  # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0]  # 计算需要填充的像素数目（图像的高这一维度上）
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    return img_new


def all(img_path):
    # 提取红色部分保存
    a = getred(img_path)
    # 行分割
    plt.imshow(a)
    plt.show()
    b = pre_pic(a, 0)

    # 识别结果储存
    result = []
    for i in b:
        # 列分割
        c = pre_pic(i, 1)
        plt.imshow(i)
        plt.show()
        row = []
        for j in c:
            plt.imshow(j)
            plt.show()
            j = resize_keep(j, [28, 28])
            d=reco.preImg(j)
            if int(d)==5:d="T"
            row.append(d)
            # print(reco.preImg(j))  # 链接pre文件中的preImg函数,预测图片值
        result.append(row)
    # result=[['T'], ['T'], [3], [2], [2, 3], [3, 1]]
    #print(result)
    return result


if __name__ == "__main__":
    img_path = "paper2.png"
    print(all(img_path))
