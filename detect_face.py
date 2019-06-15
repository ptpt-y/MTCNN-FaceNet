# coding: utf-8

import tensorflow as tf
import sys, os
import argparse
import time

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

    parser.add_argument('video_path', type=str,
                        help='The image path of the testing image', default='./video.mp4')
    parser.add_argument('--model_dir', type=str,
                        help='The directory of trained model',
                        default='./save_model/all_in_one/')
    parser.add_argument(
        '--threshold',
        type=float,
        nargs=3,
        help='Three thresholds for pnet, rnet, onet, respectively.',
        default=[0.8, 0.8, 0.8])
    parser.add_argument('--minsize', type=int,
                        help='The minimum size of face to detect.', default=20)
    parser.add_argument('--factor', type=float,
                        help='The scale stride of orginal image', default=0.7)
    parser.add_argument('--save_path', type=str,
                        help='If save_image is true, specify the output path.',
                        default='./datasets/mtcnn_160_face/video_img/')
    parser.add_argument('--capture_interval', type=int,
                        help='capture interval',
                        default=24)

    return parser.parse_args(argv)


def detect_frame(capture_count, img, file_paths, minsize, threshold, factor, save_path):
    output_dir_img = './datasets/mtcnn_160_face/img/'
    if not os.path.exists(output_dir_img):
        os.makedirs(output_dir_img) 
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                if len(file_paths) == 3:
                    image_pnet = tf.placeholder(
                        tf.float32, [None, None, None, 3])
                    pnet = PNet({'data': image_pnet}, mode='test')
                    out_tensor_pnet = pnet.get_all_output()

                    image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                    rnet = RNet({'data': image_rnet}, mode='test')
                    out_tensor_rnet = rnet.get_all_output()

                    image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                    onet = ONet({'data': image_onet}, mode='test')
                    out_tensor_onet = onet.get_all_output()

                    saver_pnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                     if v.name[0:5] == "pnet/"])
                    saver_rnet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                     if v.name[0:5] == "rnet/"])
                    saver_onet = tf.train.Saver(
                                    [v for v in tf.global_variables()
                                     if v.name[0:5] == "onet/"])

                    saver_pnet.restore(sess, file_paths[0])

                    def pnet_fun(img): return sess.run(
                        out_tensor_pnet, feed_dict={image_pnet: img})

                    saver_rnet.restore(sess, file_paths[1])

                    def rnet_fun(img): return sess.run(
                        out_tensor_rnet, feed_dict={image_rnet: img})

                    saver_onet.restore(sess, file_paths[2])

                    def onet_fun(img): return sess.run(
                        out_tensor_onet, feed_dict={image_onet: img})

                else:
                    saver = tf.train.import_meta_graph(file_paths[0])
                    saver.restore(sess, file_paths[1])

                    def pnet_fun(img): return sess.run(
                        ('softmax/Reshape_1:0',
                         'pnet/conv4-2/BiasAdd:0'),
                        feed_dict={
                            'Placeholder:0': img})

                    def rnet_fun(img): return sess.run(
                        ('softmax_1/softmax:0',
                         'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={
                            'Placeholder_1:0': img})

                    def onet_fun(img): return sess.run(
                        ('softmax_2/softmax:0',
                         'onet/conv6-2/onet/conv6-2:0',
                         'onet/conv6-3/onet/conv6-3:0'),
                        feed_dict={
                            'Placeholder_2:0': img})

                random_key = np.random.randint(0, high=99999)
                output_dir_bbox = './datasets/mtcnn_160_face/bbox/'
                if not os.path.exists(output_dir_bbox):
                    os.makedirs(output_dir_bbox) 
                bounding_boxes_filename = os.path.join(output_dir_bbox, 'bounding_boxes_%05d.txt' % random_key)
            
                with open(bounding_boxes_filename, "w") as text_file:
                    start_time = time.time()
                    rectangles, points = detect_face(img, minsize,
                                                  pnet_fun, rnet_fun, onet_fun,
                                                  threshold, factor)
                    duration = time.time() - start_time

                    print("detect time:",duration)

                    nrof_faces = rectangles.shape[0]
                    if nrof_faces>0:
                        det = rectangles[:,0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            output_filename = "{}{}{}{}{}".format(output_dir_img, capture_count,'_',i, '.jpg')
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-32/2, 0)
                            bb[1] = np.maximum(det[1]-32/2, 0)
                            bb[2] = np.minimum(det[2]+32/2, img_size[1])
                            bb[3] = np.minimum(det[3]+32/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                            scaled = cv2.cvtColor(scaled,cv2.COLOR_BGR2RGB)
                            misc.imsave(output_filename, scaled)
                            text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('NO FACE in capture %d' % (capture_count))
                        text_file.write('%s\n' % (output_dir_img))

                points = np.transpose(points)
                for rectangle in rectangles:
                    cv2.putText(img, str(rectangle[4]),
                                (int(rectangle[0]), int(rectangle[1])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0))
                    cv2.rectangle(img, (int(rectangle[0]), int(rectangle[1])),
                                  (int(rectangle[2]), int(rectangle[3])),
                                  (255, 0, 0), 2)
                for point in points:
                    for i in range(0, 10, 2):
                        cv2.circle(img, (int(point[i]), int(
                            point[i + 1])), 4, (255, 0, 255), thickness=2)
                cv2.imwrite(save_path + str(capture_count) + '.jpg', img)

    return rectangles

def main1(args):
    capture_interval = args.capture_interval
    capture_num = 100
    capture_count = 0
    frame_count = 0
    
    file_paths = get_model_filenames(args.model_dir)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path) 

    video = cv2.VideoCapture(args.video_path)
    while(True):
        ret,frame = video.read()
        if not ret:
            break
        if(capture_count % capture_interval == 0):
            l = detect_frame(capture_count, frame, file_paths, args.minsize, args.threshold, args.factor, args.save_path)
            ll=np.delete(l,4,axis=1)
            print(capture_count,":",ll,'\n')
            frame_count += 1
        capture_count += 1

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main1(parse_arguments(sys.argv[1:]))
    print('视频阅读完成')

