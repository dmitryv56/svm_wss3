import os
from os.path import join

basedir="/home/osboxes/PycharmProjects/hog_data/faces/train/"

#Given a training set of instance label pairs ( x(i), y(i) ), i=1,2,..., L where x(i) is a vector belongs n-dimensional
# euclidean space, y(i) belongs to { +1, -1 }

# training set of the instance-label pairs (x(i), +1) path
pos_img_dir=join(basedir,"face")
# training set of the instance-label pairs (x(i), -1) path
neg_img_dir=join(basedir,"non-face")

# image resolution
img_resolution=(19,19)
(img_x,img_y) = img_resolution

# the image flattened to one dimension. This vector is an instance of training vector 'x'.
# The length of the training vector is
flatten_img_len = img_x * img_y

# amount of '+1' labeled instances in the training set
train_set_pos =256
# amount of '-1' labeled instances in the training set
train_set_neg =256

# svm serialization file

model_svm_file=join(basedir,'../faces.pkl')
