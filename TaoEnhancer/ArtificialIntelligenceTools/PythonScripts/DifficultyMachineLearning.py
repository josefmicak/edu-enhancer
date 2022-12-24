﻿import pickle
import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path


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


def predict_new(SubquestionTypeAveragePoints, AnswerCorrectness, SubjectAveragePoints, ContainsImage, NegativePoints, MinimumPointsShare, subquestionPoints, df, model):
    SubquestionTypeAveragePoints_mean = df["SubquestionTypeAveragePoints"].mean()
    SubquestionTypeAveragePoints_std = df["SubquestionTypeAveragePoints"].std()
    AnswerCorrectness_mean = df["AnswerCorrectness"].mean()
    AnswerCorrectness_std = df["AnswerCorrectness"].std()
    SubjectAveragePoints_mean = df["SubjectAveragePoints"].mean()
    SubjectAveragePoints_std = df["SubjectAveragePoints"].std()
    ContainsImage_mean = df["ContainsImage"].mean()
    ContainsImage_std = df["ContainsImage"].std()
    NegativePoints_mean = df["NegativePoints"].mean()
    NegativePoints_std = df["NegativePoints"].std()
    MinimumPointsShare_mean = df["MinimumPointsShare"].mean()
    MinimumPointsShare_std = df["MinimumPointsShare"].std()

    SubquestionTypeAveragePoints = (SubquestionTypeAveragePoints - SubquestionTypeAveragePoints_mean) / SubquestionTypeAveragePoints_std
    AnswerCorrectness = (AnswerCorrectness - AnswerCorrectness_mean) / AnswerCorrectness_std
    SubjectAveragePoints = (SubjectAveragePoints - SubjectAveragePoints_mean) / SubjectAveragePoints_std
    ContainsImage = (ContainsImage - ContainsImage_mean) / ContainsImage_std
    NegativePoints = (NegativePoints - NegativePoints_mean) / NegativePoints_std
    MinimumPointsShare = (MinimumPointsShare - MinimumPointsShare_mean) / MinimumPointsShare_std

    x_unseen = torch.Tensor([SubquestionTypeAveragePoints, AnswerCorrectness, SubjectAveragePoints, ContainsImage, NegativePoints, MinimumPointsShare])
    y_unseen = model.predict(torch.atleast_2d(x_unseen))
    if round(y_unseen[0], 2) > subquestionPoints:
        return subquestionPoints
    else:
        return round(y_unseen[0], 2)


def load_model(model, login, X_train, y_train, retrainModel):
    base_path = Path(__file__)
    file_path_string = "../model/results/" + login + "_LR.sav"
    file_path = (base_path / file_path_string).resolve()

    if retrainModel is True:
        model.fit(X_train, y_train)
        save_model(model, login)
    else:
        try:  # a file with trained model already exists for this user
            model = pickle.load(open(file_path, 'rb'))
        except:  # a file with trained model does not exist for this user - it's necessary to create a new file
            model.fit(X_train, y_train)
            save_model(model, login)
    return model


def save_model(model, login):
    base_path = Path(__file__)
    file_path_string = "../model/results/" + login + "_LR.sav"
    file_path = (base_path / file_path_string).resolve()
    pickle.dump(model, open(file_path, 'wb'))


def main(arguments):
    if len(arguments) > 1:
        platform = arguments[1]
        login = arguments[2]
        retrainModel = eval(arguments[3])
        testNumberIdentifier = arguments[4]
    else:
        login = "login"

    conn_str = ""
    if platform == 'Windows':
        conn_str = (
            r"Driver={ODBC Driver 17 for SQL Server};"
            r"Server=(localdb)\mssqllocaldb;"
            r"Database=TaoEnhancerDB;"
            r"Trusted_Connection=yes;"
        )
    elif platform == 'Linux':
        conn_str = (
            r"Driver={ODBC Driver 17 for SQL Server};"
            r"Server=127.0.0.1;"
            r"Database=TaoEnhancerDB;"
            r"Uid=MyUser;"
            r"Pwd=Userpassword1;"
        )

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(connection_url)

    sql = "SELECT * FROM SubquestionResultRecord WHERE OwnerLogin = '" + login + "'"
    df = pd.read_sql(sql, engine)
    df = df.drop('OwnerLogin', axis=1)  # owner login is irrelevant in this context
    df = df.drop('QuestionNumberIdentifier', axis=1)  # question number identifier is irrelevant in this context
    df = df.drop('SubquestionIdentifier', axis=1)  # subqustion identifier is irrelevant in this context
    df = df.drop('TestResultIdentifier', axis=1)  # test result identifier is irrelevant in this context

    # necessary preprocessing
    data = df[df.columns[:-1]]
    data = data.apply(
        lambda x: (x - x.mean()) / x.std()
    )

    data['StudentsPoints'] = df.StudentsPoints

    X = data.drop('StudentsPoints', axis=1).to_numpy()
    y = data['StudentsPoints'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model = load_model(model, login, X_train, y_train, retrainModel)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #selected testTemplate
    sql = "SELECT * FROM TestTemplate WHERE OwnerLogin = '" + login + "' AND TestNumberIdentifier = '" + testNumberIdentifier + "'"
    testTemplateDf = pd.read_sql(sql, engine)
    NegativePoints = testTemplateDf.iloc[0]['NegativePoints']
    MinimumPoints = testTemplateDf.iloc[0]['MinimumPoints']
    SubjectsArray = ["Chemie", "Zeměpis", "Matematika", "Dějepis", "Informatika"]  # TODO - skutecne predmety
    TestSubjectIndex = SubjectsArray.index(testTemplateDf.iloc[0]['Subject'])

    #question number identifiers of every question included in the test
    sql = "SELECT DISTINCT QuestionNumberIdentifier FROM QuestionTemplate WHERE OwnerLogin = '" + login +\
          "' AND TestTemplateTestNumberIdentifier = '" + testNumberIdentifier + "'"
    questionNumberIdentifierList = pd.read_sql(sql, engine).values.tolist()
    questionNumberIdentifiers = ""
    some_list_len = len(questionNumberIdentifierList)
    for i in range(some_list_len):
        #the question number identifier must be properly formatted in order to be used in a SQL query
        questionNumberIdentifier = str(questionNumberIdentifierList[i])
        questionNumberIdentifier = questionNumberIdentifier[1:]
        questionNumberIdentifier = questionNumberIdentifier[:len(questionNumberIdentifier) - 1]
        questionNumberIdentifiers += questionNumberIdentifier + ", "
    questionNumberIdentifiers = questionNumberIdentifiers[:len(questionNumberIdentifiers) - 2]

    #test difficulty statistics (all relevant data needed to measure the test difficulty)
    sql = "SELECT * FROM TestDifficultyStatistics WHERE UserLogin = '" + login + "'"
    testDifficultyStatisticsDf = pd.read_sql(sql, engine)
    SubjectAveragePointsArray = testDifficultyStatisticsDf.iloc[0]['SubjectAveragePoints'].split("~")
    SubquestionTypeAveragePointsArray = testDifficultyStatisticsDf.iloc[0]['SubquestionTypeAveragePoints'].split("~")
    SubquestionTypeAverageAnswerCorrectnessArray = testDifficultyStatisticsDf.iloc[0]['SubquestionTypeAverageAnswerCorrectness'].split("~")

    #all subquestion templates included in the test
    sql = "SELECT * FROM SubquestionTemplate WHERE OwnerLogin = '" + login + "' AND QuestionNumberIdentifier IN (" + questionNumberIdentifiers + ")"
    subquestionTemplatesDf = pd.read_sql(sql, engine)
    TotalTestPoints = subquestionTemplatesDf['SubquestionPoints'].sum()
    MinimumPointsShare = MinimumPoints / TotalTestPoints
    PredictedTestPoints = 0

    for i in range(subquestionTemplatesDf.shape[0]):
        subquestionTemplate = subquestionTemplatesDf.iloc[i]
        subquestionPoints = subquestionTemplate["SubquestionPoints"]
        ContainsImage = 0
        if (len(str(subquestionTemplate["ImageSource"])) > 0):
            ContainsImage = 1

        SubquestionTypeAveragePoints = float(
            SubquestionTypeAveragePointsArray[subquestionTemplate["SubquestionType"] - 1].replace(",", "."))
        AnswerCorrectness = float(
            SubquestionTypeAverageAnswerCorrectnessArray[subquestionTemplate["SubquestionType"] - 1].replace(",", "."))
        SubjectAveragePoints = float(SubjectAveragePointsArray[TestSubjectIndex].replace(",", "."))
        PredictedTestPoints += predict_new(SubquestionTypeAveragePoints, AnswerCorrectness, SubjectAveragePoints, ContainsImage,
                    NegativePoints, MinimumPointsShare, subquestionPoints, df, model)
    print(round(PredictedTestPoints, 2))


if __name__ == '__main__':
        main(sys.argv)