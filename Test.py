#from dlframe import DataSet, Splitter, Model, Judger, WebManager
from dlframe.dataset import DataSet
from dlframe.judger import Judger
from dlframe.model import Model
from dlframe.splitter import Splitter
from dlframe.webmanager import WebManager
from typing import Any, Tuple
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import pandas as pd

# 从本地传入数据集 并分好数据和标签
def loaddata(filename):
    dataset = []
    fp = open(filename)
    for i in fp.readlines()[0:]:
        a = i.strip().split(',')
        dataset.append(a[1:])
    return dataset

def loadtarget(filename):
    targrtset = []
    fp = open(filename)
    for i in fp.readlines()[0:]:
        a = i.strip().split(',')
        targrtset.append([float(a[0])])

    return targrtset

# 数据集
class TestDataset(DataSet):
    def __init__(self, filename) -> None:
        super().__init__()
        self.data = loaddata(filename)
        self.target = loadtarget(filename)
        #self.data_line=self.data.shape[1] #获取属性数

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

class TrainTestDataset(DataSet):
    def __init__(self, filename) -> None:
        super().__init__()
        self.data = loaddata(filename)
        self.target = loadtarget(filename)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]  

# 数据集切分器
class TestSplitter(Splitter):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset: TestDataset) -> Tuple[DataSet, DataSet]:
        self.logger.print("I'm ratio:{}".format(self.ratio))
        x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=1-self.ratio)
        x_train = np.mat(x_train)
        x_test = np.mat(x_test)
        y_train = np.mat(y_train)
        y_test = np.mat(y_test)
        trainset = np.column_stack((y_train, x_train))
        np.savetxt('trainset.csv', trainset, delimiter=',', fmt='%s')
        trainset = TrainTestDataset('trainset.csv')
        testset = np.column_stack((y_test, x_test))
        np.savetxt('testset.csv', testset, delimiter=',', fmt='%s')
        testset = TrainTestDataset('testset.csv')
        return trainset, testset

# 模型
from sklearn import *
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn import svm

class TestModel(Model):
    model_name:str

    def __init__(self,model_name, learning_rate) -> None:
        super().__init__()
        self.model_name=model_name
        self.learning_rate = learning_rate
        #self.model = linear_model.LogisticRegression(random_state=0)
    
    """
    def chooseModel(self):
        if(model_name=='DT'):
            self.model=DT(random_state=0)
        elif(model_name=='KNN'):
            self.model=KNN(n_neighbors=self.data.shape[1]-1) 
    """

    def train(self, trainDataset: TestDataset ) -> None:
        self.logger.print("training, lr = {}".format(self.learning_rate))
        if(self.model_name=='DT'):
            self.model=DecisionTreeClassifier()
        elif(self.model_name=='KNN'):
            self.model=KNeighborsClassifier(n_neighbors=np.array(trainDataset.data).shape[1]-1)
        elif(self.model_name=='RF'):
            self.model=RandomForestClassifier(n_estimators=10,random_state=123)
        elif(self.model_name=='XGDT'):
            self.model=GradientBoostingClassifier(random_state=123)
        elif(self.model_name=='XGBoost'):
            self.model=XGBClassifier()
        elif(self.model_name=='AdaBoost'):  #二分类器，数据集重设
            self.model=AdaBoostClassifier(random_state=123)
        elif(self.model_name=='KMeans'):  
            self.model=KMeans(n_clusters=3) #引入更多数据集需要重新设置聚类目标
        elif(self.model_name=='SVM'):
            self.model=svm.SVC(kernel='linear')
        elif(self.model_name=='LVQ'):
            self.model=LVQ(self.model_name,self.learning_rate) 
        self.model = self.model.fit(trainDataset.data, trainDataset.target)

        return super().train(trainDataset)

    def test(self, testDataset: TestDataset) -> Any:
        self.logger.print("testing")
        iris_y_pred = self.model.predict(testDataset.data)
        return iris_y_pred

class LVQ(TestModel):
    def __init__(self,model_name, learning_rate, max_iter=10000, eta=0.1, e=0.01):
        super().__init__(model_name, learning_rate)
        self.max_iter = max_iter
        self.eta = eta
        self.e = e

    def dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_mu(self, X, Y):
        k = len(Y)
        X=np.array(X)
        Y=np.array(Y)
        index = np.random.choice(X.shape[0], 1, replace=False)
        mus = []
        mus.append(X[index])
        mus_label = []
        mus_label.append(Y[index])
        for _ in range(k - 1):
            max_dist_index = 0
            max_distance = 0
            for j in range(X.shape[0]):
                min_dist_with_mu = 999999

                for mu in mus:
                    dist_with_mu = self.dist(mu, X[j])
                    if min_dist_with_mu > dist_with_mu:
                        min_dist_with_mu = dist_with_mu

                if max_distance < min_dist_with_mu:
                    max_distance = min_dist_with_mu
                    max_dist_index = j
            mus.append(X[max_dist_index])
            mus_label.append(Y[max_dist_index])

        mus_array = np.array([])
        for i in range(k):
            if i == 0:
                mus_array = mus[i]
            else:
                mus[i] = mus[i].reshape(mus[0].shape)
                mus_array = np.append(mus_array, mus[i], axis=0)
        mus_label_array = np.array(mus_label)
        return mus_array, mus_label_array

    def get_mu_index(self, x):
        min_dist_with_mu = 999999
        index = -1

        for i in range(self.mus_array.shape[0]):
            dist_with_mu = self.dist(self.mus_array[i], x)
            if min_dist_with_mu > dist_with_mu:
                min_dist_with_mu = dist_with_mu
                index = i

        return index

    def fit(self, X, Y):
        self.mus_array, self.mus_label_array = self.get_mu(X, Y)
        iter = 0

        while(iter < self.max_iter):
            old_mus_array = copy.deepcopy(self.mus_array)
            index = np.random.choice(Y.shape[0], 1, replace=False)

            mu_index = self.get_mu_index(X[index])
            if self.mus_label_array[mu_index] == Y[index]:
                self.mus_array[mu_index] = self.mus_array[mu_index] + \
                    self.eta * (X[index] - self.mus_array[mu_index])
            else:
                self.mus_array[mu_index] = self.mus_array[mu_index] - \
                    self.eta * (X[index] - self.mus_array[mu_index])

            diff = 0
            for i in range(self.mus_array.shape[0]):
                diff += np.linalg.norm(self.mus_array[i] - old_mus_array[i])
            if diff < self.e:
                print('迭代{}次退出'.format(iter))
                return
            iter += 1
        print("迭代超过{}次，退出迭代".format(self.max_iter))


""" class DT(TestModel): #决策树模型
    def __init__(self,name,learning_rate):
        super().__init__(name,learning_rate)
                
    def train(self, trainDataset: TestDataset ) -> None:
        self.logger.print("training, lr = {}".format(self.learning_rate))
        self.model=DecisionTreeClassifier()
        self.model.fit(trainDataset.data, trainDataset.target)
        return super().train(trainDataset) """




# 结果判别器
class TestJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, y_hat, test_dataset: TestDataset) -> None:
        #self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        #self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        #self.logger.print(len(y_hat))
        #self.logger.print(len(test_dataset))
        self.logger.print("准确率:%.3f" % accuracy_score(test_dataset.target, y_hat))
        return super().judge(y_hat, test_dataset)

if __name__ == '__main__':
    # 注册与运行
    WebManager().register_dataset(
        TestDataset("iris.csv"), 'iris'
    ).register_dataset(
        TestDataset("wine.csv"), 'wine'
    ).register_splitter(
        TestSplitter(0.8), 'ratio:0.8'
    ).register_splitter(
        TestSplitter(0.5), 'ratio:0.5'
    ).register_model(
        TestModel('DT',1e-3),'DT'
    ).register_model(
        TestModel('KNN', 1e-3),'KNN'
    ).register_model(
        TestModel('RF',1e-3),'RF' #随机森林
    ).register_model(
        TestModel('XGDT',1e-3),'XGDT' #梯度提升树
    ).register_model(
        TestModel('XGBoost',1e-3),'XGBoost' 
    ).register_model(
        TestModel('AdaBoost',1e-3),'AdaBoost' #二分类器，数据集重设
    ).register_model(
        TestModel('KMeans', 1e-3),'KMeans'
    ).register_model(
        TestModel('SVM', 1e-3),'SVM'
    ).register_model(
        TestModel('LVQ', 1e-3),'LVQ'
    ).register_judger(
        TestJudger()
    ).start()