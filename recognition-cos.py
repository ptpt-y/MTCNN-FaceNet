# coding: utf-8

import tensorflow as tf
import sys, os
import argparse
import time
import split

import cv2
import numpy as np

from scipy import misc
from src.mtcnn import PNet, RNet, ONet
import src.facenet as facenet
from tools import detect_face, get_model_filenames

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,7"

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('img_path', type=str,
                        help='The image path of the testing image', 
                        default='./datasets/mtcnn_160_face/img/')
    parser.add_argument('--split_img', type=bool,
                        help='If split_img is true, split images in img_path.',
                        default=False)
    parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='20180408-102900/20180408-102900.pb')
    parser.add_argument('--dist', type=float,
                        help='The dist threshold', default=0.425)
    parser.add_argument('--save_path', type=str,
                        help='The directory of result.',
                        default='./video_face_result/cos/')

    return parser.parse_args(argv)


def reco_face(args):
    save_path = "{}{}{}{}{}".format(args.save_path, args.model_dir[2:4], '_', args.dist, '/')
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                file_paths = args.img_path
                imgs_list = os.listdir(file_paths)
                all_img = len(os.listdir(file_paths))
                people_sum = 0
                img_list = []
                emb_list = []
                
                facenet.load_model(args.model_dir)
#                 facenet.load_model('20190218-164145/20190218-164145.pb')
                image_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                        
                start_time = time.time()
                for img in imgs_list:
                    x = len(os.listdir(save_path))
                    file = "{}{}".format(file_paths, img)
                    image = cv2.imread(file)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    image = facenet.prewhiten(image)
                    image_reshaped = image.reshape(-1,160,160,3)
                    emb_temp = np.zeros((1, embedding_size))
                    emb_temp[0, :] = sess.run(embeddings, feed_dict={image_placeholder: image_reshaped, phase_train_placeholder: False })[0]
                    
                    if x == 0:
                        people_sum += 1
                        output_peoplename = "{}{}".format(save_path, img)
                        misc.imsave(output_peoplename, image)
                        print("save new face")
                        img_list.append(image_reshaped)
                        emb_list.append(emb_temp[0, :])
                    else:
                        is_exist = False
                        for k in range(x):
                            dist = np.dot(emb_temp[0, :], emb_list[k]) / (np.linalg.norm(emb_temp[0, :]) * np.linalg.norm(emb_list[k]))
                            print(' %1.4f  ' % dist, end='')
                            if (dist > args.dist):
                                print("\n already existed as",k+1)
                                is_exist = True
                                break
                                    
                        if not is_exist:
                            people_sum += 1
                            output_peoplename = "{}{}".format(save_path, img)
                            misc.imsave(output_peoplename, image)
                            print("save new face")
                            emb_list.append(emb_temp[0, :])
                            img_list.append(image_reshaped)
                                
                duration = time.time() - start_time
                print("detect time:",duration)

    return people_sum


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if args.split_img:
        split.split(sys.argv[1:2])
    reco_face(parse_arguments(sys.argv[1:]))
    print('识别完成')

