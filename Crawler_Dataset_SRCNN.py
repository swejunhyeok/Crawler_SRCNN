import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# output resolution 사이즈
N = 512
# input resolution 사이즈
M = N // 2

Epoch = 5001

learning_rate = 0.0001
Traindata_set_Num = 400
batch_size = 8

def Xavier_initializer(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable= tf.Variable(initializer(shape=shape), name=name)
    return variable

def conv2d(x, W, name):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def __avg_pool__(x, name):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def __conv2D__(input_x, name, w_shape, b_shape):
    w_conv = Xavier_initializer("w" + name, w_shape)
    b_conv = tf.get_variable("b" + name, b_shape, initializer=tf.contrib.layers.xavier_initializer())
    h_conv = tf.nn.relu(conv2d(input_x, w_conv, "c" + name) + b_conv, name= "r" + name)
    return h_conv

def __model__(input_x, filter_size):
    # input_x (256, 256, 3)

    # Layer_1 (128, 128, 64)
    input_x = __conv2D__(input_x, name="1_1", w_shape=[filter_size, filter_size, 3, 64],  b_shape=[64])
    input_x = __conv2D__(input_x, name="1_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    conv_1 = input_x
    input_x = __avg_pool__(input_x, name="p1")

    # Layer_2 (64, 64, 128)
    input_x = __conv2D__(input_x, name="2_1", w_shape=[filter_size, filter_size, 64, 128], b_shape=[128])
    input_x = __conv2D__(input_x, name="2_2", w_shape=[filter_size, filter_size, 128, 128], b_shape=[128])
    conv_2 = input_x
    input_x = __avg_pool__(input_x, name="p2")

    # Layer_3 (32, 32, 256)
    input_x = __conv2D__(input_x, name="3_1", w_shape=[filter_size, filter_size, 128, 256], b_shape=[256])
    input_x = __conv2D__(input_x, name="3_2", w_shape=[filter_size, filter_size, 256, 256], b_shape=[256])
    input_x = __conv2D__(input_x, name="3_3", w_shape=[filter_size, filter_size, 256, 256], b_shape=[256])
    input_x = __conv2D__(input_x, name="3_4", w_shape=[filter_size, filter_size, 256, 256], b_shape=[256])
    conv_3 = input_x
    input_x = __avg_pool__(input_x, name="p3")

    # Layer_4 (16, 16, 512)
    input_x = __conv2D__(input_x, name="4_1", w_shape=[filter_size, filter_size, 256, 512], b_shape=[512])
    input_x = __conv2D__(input_x, name="4_2", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    input_x = __conv2D__(input_x, name="4_3", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    input_x = __conv2D__(input_x, name="4_4", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    conv_4 = input_x
    input_x = __avg_pool__(input_x, name="p4")

    # Layer_4 (8, 8, 512)
    input_x = __conv2D__(input_x, name="5_1", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    input_x = __conv2D__(input_x, name="5_2", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    input_x = __conv2D__(input_x, name="5_3", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    input_x = __conv2D__(input_x, name="5_4", w_shape=[filter_size, filter_size, 512, 512], b_shape=[512])
    conv_5 = input_x
    input_x = __avg_pool__(input_x, name="p5")

    # Reconstruction
    # R_Layer_1 (16, 16, 64)
    input_x = tf.layers.conv2d_transpose(input_x, 64, filter_size, (2, 2), padding='same')
    input_x = tf.concat([input_x, conv_5], 3)
    input_x = __conv2D__(input_x, name="r1_1", w_shape=[filter_size, filter_size, 64 + 512, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r1_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r1_3", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r1_4", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])

    # R_Layer2 (32, 32, 64)
    input_x = tf.layers.conv2d_transpose(input_x, 64, filter_size, (2, 2), padding='same')
    input_x = tf.concat([input_x, conv_4], 3)
    input_x = __conv2D__(input_x, name="r2_1", w_shape=[filter_size, filter_size, 64 + 512, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r2_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r2_3", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r2_4", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])

    # R_Layer3 (64, 64, 64)
    input_x = tf.layers.conv2d_transpose(input_x, 64, filter_size, (2, 2), padding='same')
    input_x = tf.concat([input_x, conv_3], 3)
    input_x = __conv2D__(input_x, name="r3_1", w_shape=[filter_size, filter_size, 64 + 256, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r3_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r3_3", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r3_4", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])

    # R_Layer4 (128, 128, 64)
    input_x = tf.layers.conv2d_transpose(input_x, 64, filter_size, (2, 2), padding='same')
    input_x = tf.concat([input_x, conv_2], 3)
    input_x = __conv2D__(input_x, name="r4_1", w_shape=[filter_size, filter_size, 64 + 128, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r4_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])

    # R_Layer5 (256, 256, 64)
    input_x = tf.layers.conv2d_transpose(input_x, 64, filter_size, (2, 2), padding='same')
    input_x = tf.concat([input_x, conv_1], 3)
    input_x = __conv2D__(input_x, name="r5_1", w_shape=[filter_size, filter_size, 64 + 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r5_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])

    # R_Layer6 (512, 512, 64)
    input_x = tf.layers.conv2d_transpose(input_x, 64, filter_size, (2, 2), padding='same')
    input_x = __conv2D__(input_x, name="r6_1", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])
    input_x = __conv2D__(input_x, name="r6_2", w_shape=[filter_size, filter_size, 64, 64], b_shape=[64])

    # R_Layer7 (512, 512, 3)
    input_x = __conv2D__(input_x, name="r7", w_shape=[filter_size, filter_size, 64, 3], b_shape=[3])

    # 결과값
    y = tf.identity(input_x, name="OUTPUT")
    return y

# 원하는 경로에 폴더 생성
def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def search(dirname):
    images = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        ext = os.path.splitext(full_filename)[-1]
        if ext == '.jpg':
            images.append(filename)
    return images

def Image_Processing(start_num, end_num, train):
    content = search("/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/TEST/Image512X512")
    if end_num == -1:
        end_num = content.__len__()
    contents = []
    for id in range(start_num, end_num):
        content_img = Image.open("Image512X512/" + content[id])
        data = np.array(content_img, dtype="uint8")
        if data.shape[2] == 3:
            if train:
                data = cv2.resize(data, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            contents.append(data)
    return np.asarray(contents)

def next_batch(num, data):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    resizing_data = [cv2.resize(data_shuffle[i], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for i in
                     range(num)]

    return np.asarray(resizing_data), np.asarray(data_shuffle)

with tf.Session() as sess:
    train_data = Image_Processing(0, Traindata_set_Num, False)
    print(train_data.shape)
    test_Input_data = Image_Processing(Traindata_set_Num, -1, True)
    print(test_Input_data.shape)
    test_Target_data = Image_Processing(Traindata_set_Num, -1, False)
    print(test_Target_data.shape)

    inputs = tf.placeholder(dtype=tf.float32, shape=(None, M, M, 3), name="INPUTS")
    targets = tf.placeholder(dtype=tf.float32, shape=(None, N, N, 3), name="DESIRE")

    outputs = __model__(inputs, 5)

    # MSE
    loss = tf.losses.mean_squared_error(outputs, targets)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # weight 값들 초기화
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = "crawler_train"
    make_folder(save_path)
    make_folder("model")
    file = open('/Users/jhkimMultiGpus/Desktop/SRCNN.txt', 'w')
    for i in range(Epoch):
        batch_data = next_batch(batch_size, train_data)
        total_loss = 0
        feed_dict = {inputs: batch_data[0], targets: batch_data[1]}
        optimizer.run(feed_dict=feed_dict)
        total_loss += loss.eval(feed_dict=feed_dict)

        if i % 100 == 0:
            ix = random.randrange(0, batch_size)
            img = cv2.resize(batch_data[0][ix, :, :, :], (N, N), cv2.INTER_NEAREST)
            out = outputs.eval(feed_dict)[ix, :, :, :]
            out = np.reshape(out, [N, N, 3])
            tar = np.reshape(batch_data[1][ix, :, :, :], [N, N, 3])

            result = np.concatenate([img, out, tar], axis=1)

            cv2.imwrite(save_path + "/result" + str(i) + ".jpg", result)
            make_folder("model/crawler_model_" + str(i))
            saver.save(sess, "model/crawler_model_"+ str(i) +"/CSRCNN_MODEL.ckpt")
            #plt.imsave(save_path + "/result" + str(i) + ".jpg", result)
        total_loss /= (batch_size + 1)
        print(i, total_loss)

    save_path = "crawler_test"
    make_folder(save_path)

    for i in range(test_Input_data.__len__()):
        feed_dict = {inputs: np.reshape(test_Input_data[i], [1, M, M, 3]), targets: np.reshape(test_Target_data[i], [1, N, N, 3])}
        total_loss = loss.eval(feed_dict=feed_dict)
        img = cv2.resize(test_Input_data[i, :, :, :], (N, N), cv2.INTER_NEAREST)
        out = outputs.eval(feed_dict)[0, :, :, :]
        out = np.uint8(np.clip(out, 0, 255))

        out = np.reshape(out, [N, N, 3])
        tar = np.reshape(test_Target_data[i, :, :, :], [N, N, 3])

        result = np.concatenate([img, out, tar], axis=1)

        #cv2.imwrite(save_path + "/result" + str(i) + ".jpg", result)
        plt.imsave(save_path + "/result" + str(i) + ".jpg", result)
        print(i, total_loss)
        file.write('%s\n' % total_loss)
    file.close()
