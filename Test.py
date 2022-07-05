from dlframe import DataSet, Splitter, Model, Judger, WebManager
from typing import Any, Tuple
import math
from sklearn.datasets import load_wine, load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

# 数据集
class TestDataset(DataSet):
    def __init__(self, num) -> None:
        super().__init__()
        if (num == 1):
            iris = load_iris()
            self.data = iris['data']
            self.target = iris['target']
        if (num == 2):
            wine = load_wine()
            self.data = wine['data']
            self.target = wine['target']
        if(num==3):
            boston = load_boston()
            self.data = boston['data']
            self.target = boston['target']

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

class TrainTestDataset(DataSet):
    def __init__(self, item) -> None:
        super().__init__()
        self.item = item

    def __len__(self) -> int:
        return len(self.item)

    def __getitem__(self, idx: int) -> Any:
        return self.item[idx]

# 数据集切分器
class TestSplitter(Splitter):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset: TestDataset) -> Tuple[DataSet, DataSet]:
        X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, train_size=self.ratio)
        self.logger.print("split!")
        X_train = TrainTestDataset(X_train)
        Y_train = TrainTestDataset(Y_train)
        X_test = TrainTestDataset(X_test)
        Y_test = TrainTestDataset(Y_test)
        return X_train, X_test

# 模型
class TestModel(Model):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate

    def train(self, trainDataset: TestDataset) -> None:
        self.logger.print("trainging, lr = {}".format(self.learning_rate))
        regressor = LinearRegression()
        regressor = regressor.fit(trainDataset)
        return super().train(trainDataset)

    def test(self, testDataset: DataSet) -> Any:
        self.logger.print("testing")
        return testDataset

# 结果判别器
class TestJudger(Judger):
    def __init__(self) -> None:
        super().__init__()

    def judge(self, y_hat, test_dataset: DataSet) -> None:
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))
        return super().judge(y_hat, test_dataset)

if __name__ == '__main__':
    # 注册与运行
    WebManager().register_dataset(
        TestDataset(1), 'iris'
    ).register_dataset(
        TestDataset(2), 'wine'
    ).register_splitter(
        TestSplitter(0.8), 'ratio:0.8'
    ).register_splitter(
        TestSplitter(0.5), 'ratio:0.5'
    ).register_model(
        TestModel(1e-3)
    ).register_judger(
        TestJudger()
    ).start()