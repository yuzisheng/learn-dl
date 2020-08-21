import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

# 激活函数
f = lambda x: x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    input_vecs = [[1], [2], [3], [4], [5]]
    labels = [7, 21, 33, 39, 52]
    return input_vecs, labels


def train_linear_unit(iteration, rate):
    lu = LinearUnit(1)
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, iteration, rate)
    return lu


if __name__ == '__main__':
    # 训练
    linear_unit = train_linear_unit(10, 0.01)
    print(linear_unit)

    # 测试
    test_vecs = [1.5, 3.4, 6.3, 11.7]
    for _ in test_vecs:
        print("work {} years, monthly salary = {}".format(_, linear_unit.predict([_])))

    # 可视化
    plt.scatter(test_vecs, [linear_unit.predict([_]) for _ in test_vecs], c='red')
    plt.plot([_ for _ in np.arange(0, 20, 0.1)],
             [linear_unit.weights[0] * _ + linear_unit.bias for _ in np.arange(0, 20, 0.1)], c='blue')
    plt.show()
    print("ok")
