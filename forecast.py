# 导入相关功能包
import numpy as np
import paddle.fluid as fluid
from paddle import paddle
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
#本文件用于预测
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model_save_dir = r"C:\Users\11387\Documents\cnn-paper\cnn\first.cnn.model"
paddle.enable_static()
place = fluid.CPUPlace()


def load_image(im):
    # im = Image.open(file).convert('L')  # 将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
    # im = im.resize((28, 28), )  # resize image with high-quality 图像大小为28*28

    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)  # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
    # print(im)
    im = im / 255.0 * 2.0 - 1.0  # 归一化到【-1~1】之间
    return im

def preImg(img):
    # img = Image.open(infer_path)输入更换
    plt.imshow(img)  # 根据数组绘制图像
    plt.show()  # 显示图像
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()
    # 加载数据并开始预测
    with fluid.scope_guard(inference_scope):
        # 获取训练好的模型
        # 从指定目录中加载 推理model(inference model)
        [inference_program,  # 推理Program
         feed_target_names,  # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
         fetch_targets] = fluid.io.load_inference_model(model_save_dir,
                                                        # fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                        infer_exe)  # infer_exe: 运行 inference model的 executor
        img_p = load_image(img)
        results = infer_exe.run(program=inference_program,  # 运行推测程序
                                feed={feed_target_names[0]: img_p},  # 喂入要预测的img
                                fetch_list=fetch_targets)  # 得到推测结果,
        # 获取概率最大的label
        lab = np.argsort(results)  # argsort函数返回的是result数组值从小到大的索引值
        #print("该图片的预测结果的label为: %d" % lab[0][0][-1])  # -1代表读取数组中倒数第一列
    return lab[0][0][-1]

if __name__ == '__main__':
    strFilePath = "infer_3.png"
    # img = Image.open(strFilePath)
    img = cv2.imread(strFilePath)
    print('1',type(img))
    print(preImg(img))
