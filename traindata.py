#!/usr/bin/python3
import os
import sys
from PIL import Image
from os.path import isdir,isfile,join
import numpy as np
from config import pos_img_dir, neg_img_dir,flatten_img_len,train_set_pos,train_set_neg




def list_img_files(base_path):
    if not isdir(base_path):
        return []

    return [join(base_path, f) for f in os.listdir(base_path) if isfile(join(base_path, f))]

def imgfile2image(imgfile):
    image = Image.open(imgfile, "r")
    image = image.convert("L")  # makes it grayscale

    data = np.asarray(image.getdata(), dtype=np.float64).reshape((image.size[1], image.size[0])).flatten()

    (title,_) = os.path.splitext( os.path.basename (imgfile))

    return title, image, data

def createTestSet(imgListFolder,imgsAmount,skippedImgAmount, flattenImgLen):
    """

    :param imgListFolder:
    :param imgsAmount:
    :param ignoredImgAmount:
    :param flattenImgLen:
    :return:
    """

    img_list = list_img_files(imgListFolder)

    test_data = np.zeros((imgsAmount,flattenImgLen), dtype=np.float64)
    test_output=np.zeros((imgsAmount), dtype=np.float64)
    title_list=[]

    ind =0
    skip_ind=0

    for img in img_list:
        if skip_ind<skippedImgAmount:
            skip_ind+=1
            continue


        title,image,data = imgfile2image(img)
        test_data[ind]=np.copy(data)
        title_list.append(title)
        ind+=1
        if ind == imgsAmount:
            break

    return test_data,test_output,title_list

def createDataSet(posImgListFolder, negImgListFolder,posImgsAmount,negImgsAmount,flattenImgLen):
    """

    :param posImgListFolder:
    :param negImgListFolder:
    :param posImgsAmount:
    :param neImgsAmount:
    :param flattenImgLen:
    :return:
    """

    pos_img_list = list_img_files(posImgListFolder)
    neg_img_list = list_img_files(negImgListFolder)


    #input_data = np.zeros((256, 361), dtype=np.float64)
    #input_label = np.zeros((256), dtype=np.float64)


    input_data = np.zeros((posImgsAmount + negImgsAmount, flattenImgLen), dtype=np.float64)
    input_label = np.zeros((posImgsAmount+negImgsAmount), dtype=np.float64)
    pos_title_list=[]
    neg_title_list=[]

    ind = 0
    for img in pos_img_list:
        title, image, data = imgfile2image(img)
        input_data[ind] = np.copy(data)
        input_label[ind] = 1
        pos_title_list.append(title)
        ind += 1
        if ind == posImgsAmount:
            break

    for img in neg_img_list:
        title, image, data = imgfile2image(img)
        input_data[ind] = np.copy(data)
        input_label[ind] = -1
        neg_title_list.append(title)
        ind += 1
        if ind == (posImgsAmount+negImgsAmount):
            break

    return input_data,input_label, pos_title_list, neg_title_list


if __name__ == "__main__":
    pass

basedir="/home/osboxes/PycharmProjects/hog_data/faces/train/"

lbl_yes_dir=join(basedir,"face")
lbl_no_dir=join(basedir,"non-face")

from config import pos_img_dir, neg_img_dir,flatten_img_len,train_set_pos,train_set_neg

input_data, input_label, _, _ = createDataSet(pos_img_dir, neg_img_dir, train_set_pos,train_set_neg, flatten_img_len)




pass

