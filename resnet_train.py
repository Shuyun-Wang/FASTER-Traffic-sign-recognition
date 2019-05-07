import tensorflow as tf 
import sys
import numpy as np
import resnet
import random
 

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

#数据集加载
with tf.Session() as sess:
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    train_image=[]
    train_label=[]
    temp_image=[]
    temp_label=[]
    for i in range(4625):
        image,label=sess.run([images,labels])
        temp_image.append(image)
        temp=np.zeros((1,62))
        temp[0][label]=1
        temp_label.append(temp[0])


#数据集打乱
index=[i for i in range(4625)]
random.shuffle(index)
temp__image=[]
temp__label=[]
for i in range(2):
    for h in range(4625):
        temp__image.append(temp_image[index[h]])
        temp__label.append(temp_label[index[h]])
for j in range(185):
    train_image.append(np.array(temp__image[j*50:j*50+50])/255)
    train_label.append(np.array(temp__label[j*50:j*50+50]))
print('训练集已加载完毕')


def train():
    x = tf.placeholder(tf.float32, [None,2304 ])
    y = tf.placeholder(tf.float32, [None, 62])
    y_outputs = resnet.model_specification(x)
    global_step = tf.Variable(0, trainable=False)
    
    keep_prob=tf.placeholder(tf.float32)
    y_out=tf.nn.dropout(y_outputs,keep_prob)

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_out, labels=tf.argmax(y, 1))
    loss = tf.reduce_mean(entropy)
    # train_op = tf.train.AdamOptimizer(1e-4).minimize(loss,global_step=global_step)
    # loss_function = - tf.reduce_sum(y * tf.log(y_outputs+1e-8))
    # optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss_function,global_step=global_step)

    rate = tf.train.exponential_decay(0.003, global_step, 100, 0.97,staircase=True)
    train_op = tf.train.AdamOptimizer(rate).minimize(loss, global_step=global_step)
    correct = tf.equal(tf.argmax(y_outputs,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,"float32"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.summary.scalar('loss',loss)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./graph', sess.graph)
    saver=tf.train.Saver(max_to_keep=1)
    max_acc=0

    for j in range(3):
        for i in range(185):
            summary_str, step, _ = sess.run([summary_op,global_step, train_op], feed_dict = {x:train_image[i], y:train_label[i] , keep_prob : 0.8})
            summary_writer.add_summary(summary_str,step)
            train_accuracy = accuracy.eval(session = sess,feed_dict = {x:train_image[i], y:train_label[i],keep_prob : 1.0})
            print("step %d, train_accuracy %g" %(step, train_accuracy))
            if train_accuracy >= max_acc:
                max_acc = train_accuracy
                saver.save(sess,"./model_save/",global_step= step)

def main(_):
    train()
    print("训练完毕，模型已保存")
 
if __name__ == '__main__':
    tf.app.run()