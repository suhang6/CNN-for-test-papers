# 导入相关功能包
import os
import numpy as np
import paddle.fluid as fluid
from paddle import paddle
import matplotlib.pyplot as plt
from PIL import Image
import os



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

BUF_SIZE = 512
# 每批数据大小
BATCH_SIZE = 128
# 训练集
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(),
        buf_size=BUF_SIZE
    ),
    batch_size=BATCH_SIZE
)
# 测试集
test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.test(),
        buf_size=BUF_SIZE
    ),
    batch_size=BATCH_SIZE
)


def cnn(img):
    # 第一个卷积-池化层
    conv1 = fluid.layers.conv2d(input=img,num_filters=20,
                                filter_size=5,padding='SAME',act='relu')
    pooling1 = fluid.layers.pool2d(input=conv1,pool_size=2,
                                   pool_stride=2,pool_type='max')
    conv1_pool_1 = fluid.layers.batch_norm(input=pooling1)
    # 第二个卷积-池化层
    conv2 = fluid.layers.conv2d(input=conv1_pool_1, num_filters=50,
                                filter_size=5, padding='SAME',act='relu')
    pooling2 = fluid.layers.pool2d(input=conv2,pool_size=2,
                                   pool_stride=2,pool_type='max')
    prediction = fluid.layers.fc(input=pooling2,act='softmax',size=10)
    return prediction


# 数据格式
paddle.enable_static()
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')
prediction = cnn(image)
label = fluid.layers.data(name='lable', shape=[1], dtype='int64')
# 定义损失函数
cost = fluid.layers.cross_entropy(input=prediction, label=label)  # 交叉熵损失
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=prediction, label=label)
# 定义优化方法
optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)  # 为网络添加反向计算过程
print('ok')

# 创建执行器
paddle.enable_static()
place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

# 定义数据映射器
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
# print(type(feeder))

# 训练过程可视化
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []


def draw_train_process(title, iters, costs, accs, label_cost, lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()


EPCOH_NUM = 2  # 训练轮数
for epoch_num in range(EPCOH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),  # 运行主程序
                                        feed=feeder.feed(data),  # 给模型喂入数据
                                        fetch_list=[avg_cost, acc])  # fetch 误差、准确率

        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)

        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每200个batch打印一次信息  误差、准确率
        if batch_id % 200 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (epoch_num, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_program = fluid.default_main_program().clone(for_test=True)
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        test_cost, test_acc = exe.run(program=test_program,  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[avg_cost, acc])  # fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (epoch_num, test_cost, test_acc))

model_save_dir = "./first.cnn.model"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
fluid.io.save_inference_model(model_save_dir,  # 保存推理model的路径
                              ['image'],  # 推理（inference）需要 feed 的数据
                              [prediction],  # 保存推理（inference）结果的 Variables
                              exe)  # executor 保存 inference model
print('训练模型保存完成！')
draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
