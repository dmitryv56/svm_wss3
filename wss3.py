#!/usr/bin/python3

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  ...following to R.E.Fan, P.-H. Chen, C.-J. Lin 'Working set selection Using Second Order Information for
#  Training Support Vector Machines'-Journal of Machine Learning Research 6, 2005, 1889-1918


import os
from os.path import isdir, join, isfile
import sys

import numpy as np
import math
import datetime
import uuid
from PIL import Image
from traindata import createDataSet, createTestSet
from config import pos_img_dir, neg_img_dir,flatten_img_len,train_set_pos,train_set_neg, model_svm_file
import pickle
from pathlib import Path



LOG_FILE = ''.join(["svm", datetime.datetime.now().strftime("_%H%M%S_"), uuid.uuid1().hex, ".log"])
f = None  # output file

class svm(object):
    """

    """
    def __init__(self, m, model_svm_file=None, log_file=None, dbg_log=None):
        """

        :param model_file:
        :param log_file:
        """
        self.m_features = m
        self.w = np.zeros((self.m_features), dtype=np.float64)
        self.b = 0.0

        self.mdict={"w":None,"b":None,"description":None}
        self.log_file=log_file
        self.dbg_log=dbg_log


        self.setSVMmodel( model_svm_file )

        return
    #------------------------------------------------------------------------------------------------------------------>
    #
    #
    def setSVMmodel(self,model_svm_file):
        self.svm_model_file=model_svm_file


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def loadSVMmodel(self):
        """

        :return:
        """
        my_file=Path(self.svm_model_file)
        if my_file.is_file():
            self.mdict.clear()

            inputf = open(self.svm_model_file, 'rb')

            self.mdict = pickle.load(inputf)

            inputf.close()

            self.print_dict_data()

        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def saveSVMmodel(self,b,w,description):
        """

        :param b:
        :param w:
        :param description:
        :return:
        """

        self.mdict.clear()
        self.mdict['b']           = b
        self.mdict['w']           = np.copy(w)
        self.mdict['description'] = description
        self.print_dict_data()

        output = open(self.svm_model_file, 'wb')
        pickle.dump(self.mdict, output)
        output.close()

        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def print_dict_data(self):
        """

        :param x_data:
        :return:
        """

        if self.log_file and self.dbg_log:
            print("\n\nSVM model description : {}\n Model: bias = {} w = \n".format(self.mdict["description"],
                                                                                    self.mdict["b"]), file=self.log_file)
            for t in range(self.mdict['w'].size):
                print((self.mdict['w'])[t], end=" ", file=self.log_file)
                if t > 0 and t % 16 == 0:
                    print("\n", file=self.log_file)


        return
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def print_feature_vector(self, x_data):
        """

        :param x_data:
        :return:
        """

        print("\n\nData vector (x) :\n ")
        for t in range(x_data.size):
            print(x_data[t], end=" ", file=self.log_file)
            if t > 0 and t % 16 == 0:
                print("\n", file=self.log_file)

        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def examinate(self, x ):
        """

        :param x:
        :return:
        """


        y=np.dot(self.mdict["w"], x ) + self.mdict["b"]
        if y>=0 :
            res =1
        else:
            res = -1

        ss = np.array2string(self.mdict['w'], precision=4, separator=',', suppress_small=True)

        #print ("\nLabel is {} (y={}) for vector  = \n          {}".format(res, y, ss ))
        if self.log_file:
            print("\nLabel is {} (y={}) for vector  = \n          {}".format(res, y, ss), file = self.log_file)

        #self.print_dict_data()

        return res






#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
#
#
class wss(svm):
    """
    Working set selection (WSS) for support vector machines (SVM)  (V.Vapnik and others)

    """

    eps=1e-03  # stopping tolerance
    tau=1e-12

    n_instances = 0
    log_file=None
    dbg_log=None
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def __init__(self, n, m, x,y,model_svm_file=None, log_file=None,dbg_log=None ):
        """

        :param n:
        :param m:
        :param x:
        :param y:
        :param log_file:
        :param dbg_log:
        """
        super().__init__(m, model_svm_file,log_file,dbg_log)
        self.n_instances = n
        self.m_features=m
        self.x_shape=( self.n_instances, self.m_features)
        self.y_shape=(self.n_instances,1)
        self.x=np.copy(x)
        self.y=np.copy(y)
        if log_file:
            self.log_file=log_file
        if dbg_log:
            self.dbg_log=dbg_log

        self.Ker = np.zeros((self.n_instances,self.n_instances))
        self.kernel=None
        self.kernel_arg=[]
        pass
        self.initVectors()

        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def initVectors(self):
        self.A = np.zeros((self.n_instances), dtype=np.float64)
        self.G = np.ones((self.n_instances), dtype=np.float64)
        self.G = self.G * (-1)
        self.C = 2.0
        #self.w = np.zeros((self.m_features), dtype=np.float64)
        #self.b = 0.0

        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def createKernel(self,kernel_type, d=1, k=1.0, c=-1.0):
        """

        :param kernel_type:
        :return:
        """

        kernel_type=kernel_type.lower()
        if kernel_type == "gauss":
            self.kernel = "gauss"
            func="rbfgauss"
        if kernel_type == "linear":
            self.kernel="limear"
            func="linear"
        if kernel_type == "rbf":
            self.kernel = "rbf"
            func="rbfgauss"

        if kernel_type == "laplace":
            self.kernel = "laplace"
            func="laplace"

        elif kernel_type == "polynomial":
            self.kernel = "polynomial"
            func="polyn"
            self.kernel_arg.append(d)

        elif kernel_type== "sigmoid":
            self.kernel="sigmoid"
            func="sigmoid"
            self.kernel_arg.append(k)
            self.kernel_arg.append(c)



        elif kernel_type == "bessel":
            self.kernel = "bessel"

        elif kernel_type == "anova":
            self.kernel = "anova"
            self.kernel_arg.append(d)
            func="anova"

        else:
            self.kernel="rbf"

        for i in range(self.n_instances):
            for j in range(self.n_instances):

                self.Ker[i][j] = getattr(self,func)(i,j)


        if self.log_file and self.dbg_log:
            print ("\n\nKernel \n", file=self.log_file)
            [print (*line, file =self.log_file) for line in self.Ker ]


        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def kernel2Q(self):

        for i in range(self.n_instances):
            for j in range(self.n_instances):
                self.Ker[i][j]=self.Ker[i][j]*self.y[i]*self.y[j]

        return

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def rbfgauss(self,ind_i, ind_j):

        if ind_i == ind_j:
            return 1.0

        _,std,norm,_ = self.froNorm_x_y(ind_i,ind_j)

        return math.exp(-(norm*norm)/(2.0*std*std))

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def linear(self, ind_i, ind_j):
        return self.dot_x_y(ind_i,ind_j)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def polyn(self, ind_i,ind_j):

        return math.pow( (self.dot_x_y(ind_i,ind_j) + 1.0), self.kernel_arg[0])

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def laplace(self, ind_i, ind_j):
        pass

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def anova(selfself, ind_i,ind_j):
        pass

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def sigmoid(self,ind_i,ind_j):
        pass

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    # Frobenius' norm calculation
    #
    def froNorm_x_y(self,ind_i,ind_j):
        """

        :param ind_i:
        :param ind_j:
        :return: mean of <row i> - <row j>
                 std -standart derivation
                 frobenius norm
                 vector <row i>-<row j>
        """

        a= self.x[ind_i] -self.x[ind_j]
        mean=a.m_featuresean()
        std=a.std()
        norm = np.linalg.norm(a)
        return mean, std,norm, a

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>

    def dot_x_y(self,ind_i,ind_j):
        """

        :param ind_i:
        :param ind_j:
        :return:
        """
        return np.dot(self.x[ind_i],self.x[ind_j])



    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #

    def trainingFlow(self):
        """

        :return:
        """

        self.kernel2Q()

        while 1:
            i,j = self.selectB()
            if (j == -1):
                break

            # working set is (i,j)
            a = self.Ker[i][i]+self.Ker[j][j] -2 * self.y[i]*self.y[j] * self.Ker[i][j]

            if a<=0.0:
                a = self.tau

            b = -self.y[i]*self.G[i] + self.y[j]*self.G[j]

            #update alfa

            oldAi=self.A[i]
            oldAj=self.A[j]
            self.A[i]+=self.y[i]*b/a
            self.A[j]-=self.y[j]*b/a

            # project alfa back to the teasible region
            sum = self.y[i]*oldAi +self.y[j]*oldAj
            if self.A[i] > self.C:
                self.A[i]=self.C

            if self.A[i] <0.0:
                self.A[i]=0.0

            self.A[j] = self.y[j]*(sum-self.y[i]*self.A[i])

            if self.A[j]>self.C:
                self.A[j]=self.C
            if self.A[j]<0.0:
                self.A[j]=0.0

            self.A[i]=self.y[i]*(sum-self.y[j]*self.A[j])

            #update gradien

            deltaAi=self.A[i]-oldAi
            deltaAj=self.A[j]-oldAj
            for t in range(self.n_instances):
                self.G[t] +=self.Ker[t][i]*deltaAi +self.Ker[t][j]*deltaAj

            if self.log_file:
                print("\n\nGradient for ({} {}) working set. The coefficients are {},{}\n".format(i,j, self.A[i],self.A[j]), file = self.log_file )
                for t in range(self.n_instances ):
                    print(self.G[t],end=" ", file=self.log_file)
                    if t>0 and t%16==0:
                        print("\n", file =self.log_file)

        # Gradient and coefficients

        if self.log_file:
            print("\n\nA[] vector\n", file = self.log_file)
            for t in range(self.n_instances):
                print(self.A[t], end=" ",file=self.log_file)
                if t>0 and t%16==0:
                    print("\n", file = self.log_file )

        return


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #

    def selectB(self):
        """

        :return:
        """


        #select i

        i=-1
        G_max=-1.0/self.tau
        G_min=1./self.tau
        for t in range(self.n_instances):
            if (self.y[t] == 1 and self.A[t]<self.C ) or  ( self.y[t] == -1 and self.A[t]>0) :
                if (-self.y[t] * self.G[t] >= G_max):
                    i = t
                    G_max = -self.y[t]*self.G[t]

        # select j

        j=-1
        obj_min=1.0/self.tau
        for t in range(self.n_instances):
            if (self.y[t] == 1 and self.A[t] >0 ) or (self.y[t] == -1 and self.A[t]<self.C):
                b=G_max + self.y[t]*self.G[t]
                if (-self.y[t]*self.G[t]<=G_min):
                    G_min=-self.y[t]*self.G[t]
                if b>0:
                    a=self.Ker[i][i] + self.Ker[t][t] - 2*self.y[i]*self.y[t]*self.Ker[i][t]
                    if a<=0.0:
                        a=self.tau
                    if (-(b*b)/a<=obj_min):
                        j=t
                        obj_min=-(b*b)/a

        if (G_max -G_min)<self.eps:
            return -1,-1

        return i,j

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #
    def set_model_params(self):

        self.A_max = 0.0

        for i in range(self.n_instances):
            if abs(self.A[i]) > self.A_max:
                self.A_max=abs(self.A[i])

        for j in range(self.m_features):
            self.w[j]=0.0
            for i in range(self.n_instances):
                self.w[j]+=self.A[i]*self.y[i]*self.x[i][j]

        self.b=0.0
        max_product=-1.0/self.tau
        min_product=1.0/self.tau

        for i in range(self.n_instances):
            vect_product=np.dot(self.x[i], self.w)
            if self.y[i]==1 and vect_product<min_product:
                min_product = vect_product
            elif self.y[i] == -1 and vect_product>max_product:
                max_product=vect_product

        self.b=-(max_product+min_product)/2.0

        if self.log_file:
            print("\n\n model \n\nb = {}\n w[] = \n".format(self.b), file=self.log_file )
            for t in range(self.m_features):

                print(self.w[t],end=" ", file=self.log_file)
                if t>0 and t%16==0:
                    print("\n", file=self.log_file)



        self.saveSVMmodel(self.b,self.w,"face model. 256 faces, 256 nonfaces - training set")

        return
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
    #
    #

    def testingFlow(self, x ):
        """
        x-test instance vector of  self.m_features length
        :return:
        """

        res=self.b + np.dot(self.w, x)

        if res >0.0:
            ret_label =1
        else:
            ret_label = -1

        return ret_label

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>






def main(args, f):
    """

    :param args:
    :param f:
    :return:
    """




    input_data, output_label, _, _ = createDataSet(pos_img_dir, neg_img_dir, train_set_pos, train_set_neg,
                                                  flatten_img_len)

    train = wss(train_set_pos + train_set_neg, flatten_img_len, input_data,  output_label, model_svm_file, f, True )
    print(train.eps)
    print(train.tau)

    train.createKernel("linear")


    train.trainingFlow()
    train.set_model_params()


    x_test,_,title_list=createTestSet(pos_img_dir,10,256,flatten_img_len)
    #x_test=np.array([[1,1],[-2,10]])

    #print(train.testingFlow(x_test[0]))
    #(train.testingFlow( x_test[1]) )

    for i in range(10):
        print("{} : {}".format( title_list[i], train.testingFlow(x_test[i])))

    pass


    x_test1,_,title_list1=createTestSet(neg_img_dir,10,256,flatten_img_len)
    #x_test=np.array([[1,1],[-2,10]])

    #print(train.testingFlow(x_test[0]))
    #(train.testingFlow( x_test[1]) )

    for i in range(10):
        print("{} : {}".format( title_list1[i], train.testingFlow(x_test1[i])))




    print("Examine with model")
    if f:
        print("\n\n\n\n\n\n\n Examine with mode \n", file = f )

    exam=svm(flatten_img_len,model_svm_file,f,True)
    exam.loadSVMmodel()

    for i in range(10):
        res = exam.examinate(x_test[i])
        print("{} : {}".format(title_list[i], res ))

    for i in range(10):
        res=exam.examinate(x_test1[i])
        print("{} : {}".format(title_list1[i], res ))


    pass

def main1(args, f):

    x_test, _, title_list = createTestSet(pos_img_dir, 10, 256, flatten_img_len)


    x_test1, _, title_list1 = createTestSet(neg_img_dir, 10, 256, flatten_img_len)


    print("Examine with model")
    if f:
        print("\n\n\n\n\n\n\n Examine with mode \n", file=f)

    exam = svm(flatten_img_len, model_svm_file, f, True)
    exam.loadSVMmodel()
    list_res=[]
    for i in range(10):
        res = exam.examinate(x_test[i])
        list_res.append((title_list[i],res))
        #print("{} : {}".format(title_list[i], res))

    for i in range(10):
        res = exam.examinate(x_test1[i])
        list_res.append((title_list1[i],res))
        #print("{} : {}".format(title_list1[i], res))


    # total log
    print("\n\n\n\n\n      Total \n\n\n")
    print("\n\n\n\n\n      Total \n\n\n", file=f)

    for (title,res) in list_res:
        print("title {} : {}".format(title,res))
        print("title {} : {}".format(title, res), file = f )

if __name__ == "__main__":
    f = open(LOG_FILE, 'w')
    #main(sys.argv, f) # training and examine
    main1(sys.argv, f)# examine on the saved svm model base
    f.close()




