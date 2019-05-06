from random import random
from math import floor
import math
import random
import copy
import netCDF4
from sklearn.datasets import load_boston




class calcNode():
    def __init__(self, input_dim, calc='~', permute_rate=0.5, input_rate=0.5):
        self.calc = calc
        self.left = 0
        self.right = 0
        self.mid = 0
        self.permute_rate = permute_rate
        self.permute_list = ['+', '-', '*', '/', '~', '@', 'sin', 'cos', 'tan']
        self.input_dim = input_dim
        self.input_rate = input_rate


    def permute(self):
        if random.random() < self.permute_rate:
            permute_idx = floor(random.random() * len(self.permute_list))
            self.calc = self.permute_list[permute_idx]
            self.left = {
                '+': calcNode(self.input_dim),
                '-': calcNode(self.input_dim),
                '*': calcNode(self.input_dim),
                '/': calcNode(self.input_dim),
                '~': 0,
                '@': 0,
                'sin': calcNode(self.input_dim),
                'cos': calcNode(self.input_dim),
                'tan': calcNode(self.input_dim),
            }.get(self.calc)
            self.right = {
                '+': calcNode(self.input_dim),
                '-': calcNode(self.input_dim),
                '*': calcNode(self.input_dim),
                '/': calcNode(self.input_dim),
                '~': 0,
                '@': 0,
                'sin': 0,
                'cos': 0,
                'tan': 0,
            }.get(self.calc)
            self.mid = {
                '+': 0,
                '-': 0,
                '*': 0,
                '/': 0,
                '~': (random.random() - 0.5) * 10,
                '@': random.randrange(0, self.input_dim),
                'sin': 0,
                'cos': 0,
                'tan': 0,
            }.get(self.calc)
            if isinstance(self.left, calcNode):
                self.left.permute()
            if isinstance(self.right, calcNode):
                self.right.permute()
        else:
            return



    def value(self, input_):  # input_ is a list []
        if self.calc == '+':
            return self.left.value(input_) + self.right.value(input_)
        if self.calc == '-':
            return self.left.value(input_) - self.right.value(input_)
        if self.calc == '*':
            return self.left.value(input_) * self.right.value(input_)
        if self.calc == '/':
            return self.left.value(input_) * self.right.value(input_)
        if self.calc == '~':
            return self.mid
        if self.calc == '@':
            return input_[self.mid]
        if self.calc == 'sin':
            return math.sin(self.left.value(input_))
        if self.calc == 'cos':
            return math.cos(self.left.value(input_))
        if self.calc == 'tan':
            return math.tan(self.left.value(input_))

        return 0

    def diff(self, input, label):
        return math.pow(self.value(input) - label, 2)

    def print_(self):
        print('(', end=' ')
        if self.calc not in ['~', '@'] :
            self.left.print_()
        print(self.calc, end=' ')
        if self.calc in ['~', '@'] :
            print(self.mid, end=' ')
        if self.calc in ['+', '-', '*', '/']:
            self.right.print_()
        print(')', end=' ')








if __name__ == '__main__':

    epochs = 1000000

    boston = load_boston()
    print(boston.data.shape)

    data = boston.data
    target = boston.target

    gene1 = calcNode(data.shape[1])
    gene2 = calcNode(data.shape[1])



    for epoch in range(epochs):
        print('-------------------------')
        gene1.permute()
        print('gene1_structure:')
        gene1.print_()
        print('gene2_structure:')
        gene2.print_()
        diff_sum_1 = 0
        diff_sum_2 = 0

        for i in range(data.shape[0]):
            data_ = data[i]
            target_ = target[i]
            dif1 = gene1.diff(data_, target_)
            dif2 = gene2.diff(data_, target_)
            diff_sum_1 += dif1
            diff_sum_2 += dif2

        print('\nepoch: {}/{} \n diff_sum_1: {}\n diff_sum_2: {}'.format(epoch, epochs, diff_sum_1/data.shape[0], diff_sum_2/data.shape[0]))
        if diff_sum_1 < diff_sum_2:
            gene2 = copy.deepcopy(gene1)
        else:
            gene1 = copy.deepcopy(gene2)

        print('-------------------------')




