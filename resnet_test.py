import tensorflow as tf
import resnet
import numpy as np
 

# 创建一个reader来读取TFRecord文件中的样例并创建输入队列
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/srhyme/ML/traffic-signs-tensorflow/TFrecord/test.tfrecords"])  
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

#数据集加载
with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    test_image=[]
    test_label=[]
    temp_image=[]
    temp_label=[]
    for i in range(2520):
        image,label=sess.run([images,labels])
        temp_image.append(image)
        temp=np.zeros((1,62))
        temp[0][label]=1
        temp_label.append(temp[0])
    test_image.append(np.array(temp_image[:])/255)
    test_label.append(np.array(temp_label[:]))
print('训练集已加载完毕')



def test():
    x = tf.placeholder(tf.float32, [None, 2304])
    y = tf.placeholder(tf.float32, [None, 62])
    y_res = resnet.model_specification(x)
    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_res, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        x_b, y_b = test_image[0], test_label[0]
        ckpt = tf.train.latest_checkpoint("./model_save/")
        if ckpt:
            saver.restore(sess, ckpt)
            result = sess.run(accuracy, feed_dict={x:x_b, y:y_b})
            print("accuracy ",result)
 
 
def main():
    test()
 
if __name__ == '__main__':
    main()