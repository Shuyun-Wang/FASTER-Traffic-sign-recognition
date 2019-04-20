import os
import numpy as np
import tensorflow as tf
from PIL import Image


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 加载数据并返回两个Numpy数组
def load_data(data_dir):
    # 获取data_dir的所有子目录
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # 循环遍历标签目录并收集两个列表中的数据，标签和图像
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        #图像和标签添加到列表中
        for f in file_names:
            im=Image.open(f)
            im = im.resize((48,48))
            im = im.convert('L')
            im=np.array(im)
            images.append(im)
            labels.append(int(d))
    return images, labels


# 加载数据集和测试集的路径
ROOT_PATH = "/home/srhyme/ML/traffic-signs-tensorflow/datasets"
train_data_dir = os.path.join(ROOT_PATH, "Training")
test_data_dir = os.path.join(ROOT_PATH, "Testing")

# 转化训练集
images,labels = load_data(train_data_dir)
#把列表转换成数组
images=np.array(images)
labels=np.array(labels)
writer = tf.python_io.TFRecordWriter('/home/srhyme/ML/traffic-signs-tensorflow/TFrecord/train.tfrecords')
for index in range(images.shape[0]):
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'label': int64_feature(labels[index]),  
        'image_raw': bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('训练集已写入完毕,写入数量为',images.shape[0])

# 转换测试集
images,labels = load_data(test_data_dir)
#把列表转换成数组
images=np.array(images)
labels=np.array(labels)
writer = tf.python_io.TFRecordWriter('/home/srhyme/ML/traffic-signs-tensorflow/TFrecord/test.tfrecords')
for index in range(images.shape[0]): 
    #把图像矩阵转化为字符串  
    image_raw = images[index].tostring()  
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构  
    example = tf.train.Example(features=tf.train.Features(feature={  
        'label': int64_feature(labels[index]),  
        'image_raw': bytes_feature(image_raw)}))  
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())   
writer.close()
print('测试集已写入完毕,写入数量为',images.shape[0])