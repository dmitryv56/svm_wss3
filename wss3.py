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
from config import pos_img_dir, neg_img_dir,flatten_img_len,train_set_pos,train_set_neg



LOG_FILE = ''.join(["svm", datetime.datetime.now().strftime("_%H%M%S_"), uuid.uuid1().hex, ".log"])
f = None  # output file

class wss(object):
    """

    """

    eps=1e-03  # stopping tolerance
    tau=1e-12
    insignificant_level = 0.5
    n_instances = 0
    log_file=None
    dbg_log=None

    def __init__(self, n, m, x,y,log_file=None,dbg_log=None ):
        """

        :param n:
        :param m:
        :param x:
        :param y:
        :param log_file:
        :param dbg_log:
        """

        self.n_instances = n
        self.m=m
        self.x_shape=( self.n_instances, self.m)
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

    def initVectors(self):
        self.A = np.zeros((self.n_instances), dtype=np.float64)
        self.G = np.ones((self.n_instances), dtype=np.float64)
        self.G = self.G * (-1)
        self.C = 2.0
        self.w = np.zeros((self.m), dtype=np.float64)
        self.b = 0.0


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


        if self.log_file:
            print ("\n\nKernel \n", file=self.log_file)
            [print (*line, file =self.log_file) for line in self.Ker ]
            #for i in range(self.n_instances):
            #    for j in range(0, self.n_instances,16):
            #        for j1 in range (j, j+16 ):
            #            print ("{} ".format(self.Ker[i][j1]), file=self.log_file)
            #        print("\n",file=self.log_file )


    def kernel2Q(self):

        for i in range(self.n_instances):
            for j in range(self.n_instances):
                self.Ker[i][j]=self.Ker[i][j]*self.y[i]*self.y[j]

    def rbfgauss(self,ind_i, ind_j):

        if ind_i == ind_j:
            return 1.0

        _,std,norm,_ = self.froNorm_x_y(ind_i,ind_j)

        return math.exp(-(norm*norm)/(2.0*std*std))

    def linear(self, ind_i, ind_j):
        return self.dot_x_y(ind_i,ind_j)

    def polyn(self, ind_i,ind_j):

        return math.pow( (self.dot_x_y(ind_i,ind_j) + 1.0), self.kernel_arg[0])

    def laplace(self, ind_i, ind_j):
        pass

    def anova(selfself, ind_i,ind_j):
        pass

    def sigmoid(self,ind_i,ind_j):
        pass

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
        mean=a.mean()
        std=a.std()
        norm = np.linalg.norm(a)
        return mean, std,norm, a

    def dot_x_y(self,ind_i,ind_j):
        """

        :param ind_i:
        :param ind_j:
        :return:
        """
        return np.dot(self.x[ind_i],self.x[ind_j])





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


    def set_model_params(self):

        self.A_max = 0.0

        for i in range(self.n_instances):
            if abs(self.A[i]) > self.A_max:
                self.A_max=abs(self.A[i])

        for j in range(self.m):
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
            for t in range(self.m):

                print(self.w[t],end=" ", file=self.log_file)
                if t>0 and t%16==0:
                    print("\n", file=self.log_file)



    def testingFlow(self, x ):
        """

        :return:
        """
        #ignore small  abs(A[i]) coefficients
        # for i in range(self.n_instances):
        #     if abs(self.A[i])<self.insignificant_level*self.A_max:
        #         continue

        #res=self.b
        #res=res + np.dot(self.x[i],x) * self.A[i]*self.y[i]+self.b
        res=self.b + np.dot(self.w, x)

        if res >0.0:
            ret_label =1
        else:
            ret_label = -1

        return ret_label








def main(args, f):
    """

    :param args:
    :param f:
    :return:
    """

    input_matrix = np.array([[2.0, 4.0],
                             [2.0, 2.0],
                             [4.0, 2.0],
                             [4.0, 5.0],
                             [5.0, 9.0],
                             [5.0, 1.0],
                             [6.0, 4.0],
                             [8.0, 5.0],
                             [9.0, 1.0],
                             [10.0,6.0],
                             [4.0, 4.1]])

    output_vector = np.array([-1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1])

    input_data, output_label, _, _ = createDataSet(pos_img_dir, neg_img_dir, train_set_pos, train_set_neg,
                                                  flatten_img_len)

    a = wss(train_set_pos + train_set_neg, flatten_img_len, input_data,  output_label, f, True )
    print(a.eps)
    print(a.tau)

    a.createKernel("linear")
    a.Ker

    a.trainingFlow()
    a.set_model_params()


    x_test,_,title_list=createTestSet(pos_img_dir,10,256,flatten_img_len)
    #x_test=np.array([[1,1],[-2,10]])

    #print(a.testingFlow(x_test[0]))
    #(a.testingFlow( x_test[1]) )

    for i in range(10):
        print("{} : {}".format( title_list[i], a.testingFlow(x_test[i])))

    pass


    x_test1,_,title_list1=createTestSet(neg_img_dir,10,256,flatten_img_len)
    #x_test=np.array([[1,1],[-2,10]])

    #print(a.testingFlow(x_test[0]))
    #(a.testingFlow( x_test[1]) )

    for i in range(10):
        print("{} : {}".format( title_list1[i], a.testingFlow(x_test1[i])))


    pass

if __name__ == "__main__":
    f = open(LOG_FILE, 'w')
    main(sys.argv, f)
    f.close()




