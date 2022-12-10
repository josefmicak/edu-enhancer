import pickle
import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path


def get_accuracy(y_test, y_test_pred):
    R2 = r2_score(y_test, y_test_pred)
    print(R2)


def predict_new(SubquestionTypeAveragePoints, AnswerCorrectness, SubjectAveragePoints, ContainsImage, NegativePoints, MinimumPointsShare, df, model):
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
    print(round(y_unseen[0], 2))


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
        function_name = arguments[4]
    else:
        login = "login"

    conn_str = (
        r"Driver={ODBC Driver 17 for SQL Server};"
        r"Server=(localdb)\mssqllocaldb;"
        r"Database=TaoEnhancerDB;"
        r"Trusted_Connection=yes;"
    )

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(connection_url)

    sql = "SELECT * FROM SubquestionResultRecord WHERE OwnerLogin = '" + "login" + "'"
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

    if len(arguments) > 1:
        if function_name == 'get_accuracy':
            get_accuracy(y_test, y_test_pred)
        elif function_name == 'predict_new':
            SubquestionTypeAveragePoints = float(arguments[5])
            AnswerCorrectness = float(arguments[6])
            SubjectAveragePoints = float(arguments[7])
            ContainsImage = float(arguments[8])
            NegativePoints = float(arguments[9])
            MinimumPointsShare = float(arguments[10])
            predict_new(SubquestionTypeAveragePoints, AnswerCorrectness, SubjectAveragePoints, ContainsImage,
                        NegativePoints, MinimumPointsShare, df, model)


if __name__ == '__main__':
        main(sys.argv)