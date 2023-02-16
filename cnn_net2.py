# -*- coding: utf-8 -*-
from getimgdata import GetImgData

import tensorflow as tf

import os
from sklearn.model_selection import train_test_split

import numpy as np

class CnnNet:
    def __init__(self, output_size, size=64, rate=0.5, filename='cnn_model.h5'):
        self.output_size = output_size  # 输出神经元个数
        self.size = size  # 图像大小
        self.rate = rate  # dropout丢失率
        self.filename = filename

    # 构建CNN网络
    def cnnLayer(self):
        # 模型
        # 实例化一个Sequential，接下来就可以使用add方法来叠加所需的网络层
        model = tf.keras.Sequential()

        # 添加一个二维卷积层，输出数据维度为32，卷积核维度为3*3。输入数据维度为[28, 28, 1]，这里的维度是WHC格式，
        # 意思是输入图像像素为28*28的尺寸，使用1通道的图像。
        # 参数1 filters 输出空间的维度
        # 参数2 kernel_size 卷积核大小
        # 参数3 strides=(1, 1) 一个整数，或2个整数的元组/列表，指定沿高度和宽度方向卷积的步长。可以是单个整数，给两个方向的步长指定相同的值
        # 参数4 padding='valid' 有两个可选的值
        #        SAME 外层自动进行补充 绝大部分情况下都是使用SAME
        #        VALID很少用
        # 参数7 activation=None 指定要使用的激活函数 这里使用ReLU作为激活函数
        # 参数9 kernel_initializer
        model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_normal',
                                         input_shape=[self.size, self.size, 1],
                                         name='conv1'))

        # 添加一个二维池化层，使用最大值池化，池化维度为2*2
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same', name='pool1'))
        # 添加Dropout层
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d1"))
        # 批量标准化
        model.add(tf.keras.layers.BatchNormalization())

        # 添加第二个卷积层
        model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_normal',
                                         name='conv2'))
        # 添加第二个池化层
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same', name='pool2'))
        # 添加Dropout层
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d2"))
        # 批量标准化
        model.add(tf.keras.layers.BatchNormalization())

        # 添加第三个卷积层
        model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         kernel_initializer='he_normal',
                                         name='conv3'))
        # 添加第三个池化层
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same', name='pool3'))
        # 添加Dropout层
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d3"))
        # 批量标准化
        model.add(tf.keras.layers.BatchNormalization())

        # 添加全连接的深度神经网络
        model.add(tf.keras.layers.Flatten(name='flatten'))  # 展开
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'))

        model.add(tf.keras.layers.Dropout(self.rate))

        model.add(tf.keras.layers.Dense(self.output_size, activation='softmax', kernel_initializer='he_normal'))

        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        return model

    # 依据训练样本的模型输出与样本实际值进行模型训练
    # 参数1 x_train 特征的训练集+验证集
    # 参数2 y_train 结果的训练集+验证集
    # 参数3 retrain=True 是否重新训练模型
    def cnnTrain(self, x_train, y_train, retrain=True):

        if retrain:  # 重新训练
            model = self.cnnLayer()
            batch_size = 100 # batch_size根据每个人有多少张图片来决定
            epochs = batch_size * self.output_size # 人数*batch_size
            model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, validation_split=0.2)  #模型训练
            model.save(self.filename)
        else:  # 加载已经训练好的模型
            if not os.path.exists(self.filename):
                print('文件%s不存在，请确认！' % self.filename)

    # 预测函数，导入已训练好的模型后再将新样本数据放入，进行模型预测
    # 参数 x_test 特征的测试集
    # 返回值1 pro 样本属于各类别的概率，形如：[[0.1, 0.1, 0.0, 0.0, 0.0, 0.8]]
    # 返回值2 pre 预测结果（数字标签：0,1,2,3,4,5,...）
    def predict(self, x_test):

        if not os.path.exists(self.filename):
            print('文件%s不存在，请确认！' % self.filename)
        else:
            pre_model = tf.keras.models.load_model(self.filename)
            pro = pre_model.predict(x_test)
            pre = np.argmax(pro, axis=1)
            return pro, pre

if __name__ == '__main__':
    path = './small_img_gray'  # 灰度图路径
    # 创建读取图片数据类对象 指定读取灰度图
    getdata = GetImgData(dir=path)
    # 读取self.dir中所有图片，将图片数据转为数组，并保存相应标签
    # 返回值1 x 图片像素数据
    # 返回值2 y 图片标签 原型为独热编码 如
    # [[1. 0. 0. ... 0. 0. 0.]
    #  [1. 0. 0. ... 0. 0. 0.]
    #  ...
    #  [0. 0. 0. ... 0. 0. 1.]]
    # 返回值3 number_name 数字标签和人名的对应关系，dict类型，如：{0:'hebo', 1:'hexianbin'}
    imgs, labels, number_name = getdata.readimg()
    #print(imgs.shape) # (700, 64, 64, 1)
    #print(labels.shape) # (700, 7)
    #print(number_name) # {0: 'hebo', ... , 6: 'zhangmin'}
    #print(type(number_name)) # <class 'dict'>

    # 划分训练集验证集测试集
    # 参数1 所要划分的样本的特征集
    # 参数2 所要划分的样本的结果集
    # 参数3 test_size 测试集占所有样本的比例
    # 参数4 random_state 随机数种子 通过固定random_state的值，每次可以分割得到同样的训练集和测试集
    # 返回值1 特征的训练集+验证集
    # 返回值2 特征的测试集
    # 返回值3 结果的训练集+验证集
    # 返回值4 结果的测试集
    x_train, x_test, y_train, y_test = train_test_split(
        imgs, labels, test_size=0.2, random_state=10
    )
    # 图片目录下人的数量
    output_size = len(number_name)
    # 创建CnnNet卷积神经网络类对象
    # 参数1 output_size 输出神经元个数 为small_img_gray目录下人的数量
    # 参数2 size=64 图像大小
    # 参数3 rate=0.5 # dropout丢失率
    # 参数4 filename='cnn_model.h5' 训练模型保存的文件
    cnnnet = CnnNet(output_size=output_size)
    cnnnet.cnnTrain(x_train, y_train)
    print('---finish training---')
    # 注意这步很关键，起到了重置计算图的作用，否则多次导入训练好的计算图会出现tensor重复的问题
    cnnnet = CnnNet(output_size=output_size)
    pro, pre = cnnnet.predict(x_test)
    acc_test = np.mean(np.argmax(y_test, axis=1) == pre)  #测试集精度
    print(acc_test)
