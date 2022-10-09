import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class NeuralNetwork(nn.Module):
    def __init__(self, variables_count, xA, xB):
        super(NeuralNetwork, self).__init__()
        self.linearA = nn.Linear(variables_count, xA)
        self.linearB = nn.Linear(xA, xB)
        self.linearC = nn.Linear(xB, 1)

    def forward(self, x):
        yA = F.relu(self.linearA(x))
        yB = F.relu(self.linearB(yA))
        return self.linearC(yB)


def train(model, lr, epochs, x, y):
    loss_function = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()
        y_value = model(x)
        loss_value = loss_function(y_value, y)
        loss_value.backward()
        optimizer.step()


def get_accuracy(y_test, y_test_pred):
    R2 = r2_score(y_test, y_test_pred)
    print(R2)


def predict_new(SubquestionTypeAveragePoints, CorrectAnswersShare, SubjectAveragePoints, ContainsImage, NegativePoints, MinimumPointsShare, df, model):
    SubquestionTypeAveragePoints_mean = df["SubquestionTypeAveragePoints"].mean()
    SubquestionTypeAveragePoints_std = df["SubquestionTypeAveragePoints"].std()
    CorrectAnswersShare_mean = df["CorrectAnswersShare"].mean()
    CorrectAnswersShare_std = df["CorrectAnswersShare"].std()
    SubjectAveragePoints_mean = df["SubjectAveragePoints"].mean()
    SubjectAveragePoints_std = df["SubjectAveragePoints"].std()
    ContainsImage_mean = df["ContainsImage"].mean()
    ContainsImage_std = df["ContainsImage"].std()
    NegativePoints_mean = df["NegativePoints"].mean()
    NegativePoints_std = df["NegativePoints"].std()
    MinimumPointsShare_mean = df["MinimumPointsShare"].mean()
    MinimumPointsShare_std = df["MinimumPointsShare"].std()

    SubquestionTypeAveragePoints = (SubquestionTypeAveragePoints - SubquestionTypeAveragePoints_mean) / SubquestionTypeAveragePoints_std
    CorrectAnswersShare = (CorrectAnswersShare - CorrectAnswersShare_mean) / CorrectAnswersShare_std
    SubjectAveragePoints = (SubjectAveragePoints - SubjectAveragePoints_mean) / SubjectAveragePoints_std
    ContainsImage = (ContainsImage - ContainsImage_mean) / ContainsImage_std
    NegativePoints = (NegativePoints - NegativePoints_mean) / NegativePoints_std
    MinimumPointsShare = (MinimumPointsShare - MinimumPointsShare_mean) / MinimumPointsShare_std

    x_unseen = torch.Tensor([SubquestionTypeAveragePoints, CorrectAnswersShare, SubjectAveragePoints, ContainsImage, NegativePoints, MinimumPointsShare])
    y_unseen = model(torch.atleast_2d(x_unseen))
    print(y_unseen.item())


def main(arguments):
    conn_str = (
        r"Driver={ODBC Driver 17 for SQL Server};"
        r"Server=(localdb)\mssqllocaldb;"
        r"Database=TaoEnhancerDB;"
        r"Trusted_Connection=yes;"
    )
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})

    engine = create_engine(connection_url)

    sql = "SELECT * FROM SubquestionTemplateRecord"
    df = pd.read_sql(sql, engine)
    df = df.drop('Id', axis=1)  # id is irrelevant in this context

    # necessary preprocessing
    data = df[df.columns[:-1]]
    data = data.apply(
        lambda x: (x - x.mean()) / x.std()
    )

    data['SubquestionPoints'] = df.SubquestionPoints

    X = data.drop('SubquestionPoints', axis=1).to_numpy()
    y = data['SubquestionPoints'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = NeuralNetwork(X.shape[1], 32, 16)

    x = torch.tensor(np.array(X_train), dtype=torch.float)
    y = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float)

    model.train()
    train(model, 0.05, 500, x, y)

    y_train_pred = model(torch.tensor(X_train, dtype=torch.float))
    y_test_pred = model(torch.tensor(X_test, dtype=torch.float))

    y_train_pred = y_train_pred.detach().numpy()
    y_test_pred = y_test_pred.detach().numpy()

    if arguments[1] is not None:
        if arguments[1] == 'get_accuracy':
            get_accuracy(y_test, y_test_pred)

        if arguments[1] == 'predict_new':
            SubquestionTypeAveragePoints = float(sys.argv[2])
            CorrectAnswersShare = float(sys.argv[3])
            SubjectAveragePoints = float(sys.argv[4])
            ContainsImage = float(sys.argv[5])
            NegativePoints = float(sys.argv[6])
            MinimumPointsShare = float(sys.argv[7])
            predict_new(SubquestionTypeAveragePoints, CorrectAnswersShare, SubjectAveragePoints, ContainsImage,
                        NegativePoints, MinimumPointsShare, df, model)


if __name__ == '__main__':
    main(sys.argv)