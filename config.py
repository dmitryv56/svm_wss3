import os
from os.path import join

basedir="/home/osboxes/PycharmProjects/hog_data/faces/train/"

pos_img_dir=join(basedir,"face")
neg_img_dir=join(basedir,"non-face")

img_resolution=(19,19)
(imgx,imgy) = img_resolution
flatten_img_len = imgx * imgy
train_set_pos =256
train_set_neg =256
