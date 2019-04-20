import pylab
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

#加载数据集
# 创建一个reader来读取TFRecord文件中的样例并创建输入队列
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML/traffic-signs-tensorflow/TFrecord/train.tfrecords"])  
# 从文件中读取并解析一个样例  
_, example = reader.read(filename_queue)  
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),  
        'label': tf.FixedLenFeature([], tf.int64),  
    })
# 将字符串解析成图像对应的像素数组,其他数据转换成需要的数据类型
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32) 

with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_image=[]
    train_label=[]
    temp_image=[]
    temp_label=[]
    for i in range(4550):
        image,label=sess.run([images,labels])
        temp_image.append(image)
        temp=np.zeros((1,62))
        temp[0][label]=1
        temp_label.append(temp[0])
    for j in range(91):
        train_image.append(np.array(temp_image[j*50:j*50+50])/255)
        train_label.append(np.array(temp_label[j*50:j*50+50]))
    print('训练集已加载完毕')
sess.close()

#加载训练集
# 创建一个reader来读取TFRecord文件中的样例并创建输入队列
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML/traffic-signs-tensorflow/TFrecord/test.tfrecords"])  
# 从文集读取并解析一个样例  
_, example = reader.read(filename_queue)  
features = tf.parse_single_example(
    example,features={
        'image_raw': tf.FixedLenFeature([], tf.string),  
        'label': tf.FixedLenFeature([], tf.int64),  
    })
# 将字符串解析成图像对应的像素数组,其他数据转换成需要的数据类型
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32) 

with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    test_image=[]
    test_label=[]
    temp_image=[]
    temp_label=[]
    for i in range(2520):
        image,label=sess.run([images,labels])
        temp_image.append(image/255)
        temp=np.zeros((1,62))
        temp[0][label]=1
        temp_label.append(temp[0])
        test_image.append(np.array(temp_image))
        test_label.append(np.array(temp_label))
    print('测试集已加载完毕')
sess.close()


#初始化权重
def weight (shape):
    temp = tf.truncated_normal(shape=shape, stddev = 0.1)
    return tf.Variable(temp)

#初始化偏置值
def bias (shape):
    temp = tf.constant(0.1, shape = shape)
    return tf.Variable(temp)

#卷积,步长为1,采用SAME边界处理
def convolution (data,weight):
    return tf.nn.conv2d(data,weight,strides=[1,1,1,1],padding='SAME')

#最大池化,步长为2,采用SAME边界处理,滑动窗为2*2
def pooling (data):
    return tf.nn.max_pool(data,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义输入数据,其中None,-1代表数量不定,
x=tf.placeholder(tf.float32,[None,2304])
y=tf.placeholder(tf.float32,[None,62])
data_image = tf.reshape(x,[-1,48,48,1])
#第一层:一次卷积一次池化
w_1=weight([5,5,1,32])
b_1=bias([32])
#使用relu激活函数处理数据
d_conv1=tf.nn.relu(convolution(data_image,w_1)+b_1)
d_pool1=pooling(d_conv1)

#第二层:一次卷积一次池化
w_2=weight([5,5,32,64])
b_2=bias([64])
d_conv2=tf.nn.relu(convolution(d_pool1,w_2)+b_2)
d_pool2=pooling(d_conv2)

#第三层:全连接
w_3=weight([12*12*64,1024])
b_3=bias([1024])
d_3=tf.reshape(d_pool2,[-1,12*12*64])
d_fc3=tf.nn.relu(tf.matmul(d_3,w_3)+b_3)

#dropout操作,防止过拟合
keep_prob=tf.placeholder(tf.float32)
d_fc3_drop=tf.nn.dropout(d_fc3,keep_prob)

#第四层:softmax输出
w_4=weight([1024,62])
b_4=bias([62])
d_4=tf.nn.softmax(tf.matmul(d_fc3_drop,w_4)+b_4)

loss_function = - tf.reduce_sum(y * tf.log(d_4))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_function)

#判断预测标签和实际标签是否匹配
correct = tf.equal(tf.argmax(d_4,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,"float32"))
sess = tf.Session()     
sess.run(tf.global_variables_initializer())

for j in range(1):
    for i in range(1):
        optimizer.run(session = sess, feed_dict = {x:train_image[i], y:train_label[i],keep_prob:0.5})
        train_accuracy = accuracy.eval(session = sess,
                                            feed_dict = {x:train_image[i], y:train_label[i],keep_prob:1.0})
        print("step %d, train_accuracy %g %%" %(i+j*50, train_accuracy*100))
print("test accuracy %g %%" % accuracy.eval(session = sess,feed_dict = {x:test_image, y:test_label,keep_prob:1.0})*100)
sess.close()