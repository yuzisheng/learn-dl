from functools import reduce


class Perceptron:
    """ 感知器模型 """

    def __init__(self, input_num, activator):
        self.input_num = input_num
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return "weights\t:{}\nbias\t:{}\n".format(self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(reduce(lambda a, b: a + b,
                                     map(lambda x, w: x * w, input_vec, self.weights), 0.0) + self.bias)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = [_ for _ in map(lambda x, w: w + rate * delta * x, input_vec, self.weights)]
        self.bias += rate * delta

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)


def f(x):
    """跳跃函数作为激活函数"""
    return 1 if x > 0 else 0


def get_and_training_dataset():
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron(iteration, rate):
    p = Perceptron(2, f)
    input_vecs, labels = get_and_training_dataset()
    p.train(input_vecs, labels, iteration, rate)
    return p


if __name__ == '__main__':
    and_perception = train_and_perceptron(10, 0.1)
    print(and_perception)
    print("1 and 1 = {}".format(and_perception.predict([1, 1])))
    print("1 and 0 = {}".format(and_perception.predict([1, 0])))
    print("0 and 1 = {}".format(and_perception.predict([0, 1])))
    print("0 and 0 = {}".format(and_perception.predict([0, 0])))
    print("ok")
