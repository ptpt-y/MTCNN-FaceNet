# from imutils import paths
import argparse
import cv2
import sys
import os
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,4,7"


def split(img_path):
    img_path = img_path[0]
    image_dir = img_path[:-1]
#     print(image)
    threshold = 30
    clear_dir = "{}{}".format(image_dir, '_clear')
    if not os.path.exists(clear_dir):
            os.makedirs(clear_dir) 
    blurry_dir = "{}{}".format(image_dir, '_blurry')
    if not os.path.exists(blurry_dir):
            os.makedirs(blurry_dir) 
            
    img_list = os.listdir(img_path)
    for image_name in img_list:
        image_path = "{}{}".format(img_path, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm > threshold:
            shutil.copy(image_path, os.path.join(clear_dir, os.path.split(image_path)[1]))
        else:
            shutil.copy(image_path, os.path.join(blurry_dir, os.path.split(image_path)[1]))
    
    print("blurry images splited.")
    return clear_dir, blurry_dir