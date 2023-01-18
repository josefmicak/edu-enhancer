import pickle
import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import torch
import locale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path


def get_accuracy(y_test, y_test_pred):
    R2 = r2_score(y_test, y_test_pred)
    print(R2)


def predict_new(SubquestionTypeAveragePoints, CorrectAnswersShare, SubjectAveragePoints, ContainsImage, NegativePoints, MinimumPointsShare, df, model, duplicateColumns):
    finalTensorValues = list() #values from non-duplicate columns

    if duplicateColumns[0] != 1:
        SubquestionTypeAveragePoints_mean = df["SubquestionTypeAveragePoints"].mean()
        SubquestionTypeAveragePoints_std = df["SubquestionTypeAveragePoints"].std()
        if SubquestionTypeAveragePoints_std == 0:
            SubquestionTypeAveragePoints_std = 0.01
        SubquestionTypeAveragePoints = (SubquestionTypeAveragePoints - SubquestionTypeAveragePoints_mean) / SubquestionTypeAveragePoints_std
        finalTensorValues.append(SubquestionTypeAveragePoints)

    if duplicateColumns[1] != 1:
        CorrectAnswersShare_mean = df["CorrectAnswersShare"].mean()
        CorrectAnswersShare_std = df["CorrectAnswersShare"].std()
        if CorrectAnswersShare_std == 0:
            CorrectAnswersShare = 0.01
        CorrectAnswersShare = (CorrectAnswersShare - CorrectAnswersShare_mean) / CorrectAnswersShare_std
        finalTensorValues.append(CorrectAnswersShare)

    if duplicateColumns[2] != 1:
        SubjectAveragePoints_mean = df["SubjectAveragePoints"].mean()
        SubjectAveragePoints_std = df["SubjectAveragePoints"].std()
        if SubjectAveragePoints_std == 0:
            SubjectAveragePoints_std = 0.01
        SubjectAveragePoints = (SubjectAveragePoints - SubjectAveragePoints_mean) / SubjectAveragePoints_std
        finalTensorValues.append(SubjectAveragePoints)

    if duplicateColumns[3] != 1:
        ContainsImage_mean = df["ContainsImage"].mean()
        ContainsImage_std = df["ContainsImage"].std()
        if ContainsImage_std == 0:
            ContainsImage_std = 0.01
        ContainsImage = (ContainsImage - ContainsImage_mean) / ContainsImage_std
        finalTensorValues.append(ContainsImage)

    if duplicateColumns[4] != 1:
        NegativePoints_mean = df["NegativePoints"].mean()
        NegativePoints_std = df["NegativePoints"].std()
        if NegativePoints_std == 0:
            NegativePoints_std = 0.01
        NegativePoints = (NegativePoints - NegativePoints_mean) / NegativePoints_std
        finalTensorValues.append(NegativePoints)

    if duplicateColumns[5] != 1:
        MinimumPointsShare_mean = df["MinimumPointsShare"].mean()
        MinimumPointsShare_std = df["MinimumPointsShare"].std()
        if MinimumPointsShare_std == 0:
            MinimumPointsShare_std = 0.01
        MinimumPointsShare = (MinimumPointsShare - MinimumPointsShare_mean) / MinimumPointsShare_std
        finalTensorValues.append(MinimumPointsShare)

    x_unseen = torch.Tensor(finalTensorValues)
    y_unseen = model.predict(torch.atleast_2d(x_unseen))
    print(round(y_unseen.item(), 2))


def load_model(model, login, X_train, y_train, retrainModel):
    base_path = Path(__file__)
    file_path_string = "../model/templates/" + login + "_LR.sav"
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
    file_path_string = "../model/templates/" + login + "_LR.sav"
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

    locale.setlocale(locale.LC_NUMERIC, 'cs_CZ')

    conn_str = ""
    if platform == 'Windows':
        conn_str = (
            r"Driver={ODBC Driver 17 for SQL Server};"
            r"Server=(localdb)\mssqllocaldb;"
            r"Database=EduEnhancerDB;"
            r"Trusted_Connection=yes;"
        )
    elif platform == 'Linux':
        conn_str = (
            r"Driver={ODBC Driver 17 for SQL Server};"
            r"Server=127.0.0.1;"
            r"Database=EduEnhancerDB;"
            r"Uid=MyUser;"
            r"Pwd=Userpassword1;"
        )

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(connection_url)

    sql = "SELECT * FROM SubquestionTemplateRecord WHERE OwnerLogin = '" + login + "'"
    df = pd.read_sql(sql, engine)
    df = df.drop('OwnerLogin', axis=1)  # owner login is irrelevant in this context
    df = df.drop('SubquestionTemplateId', axis=1)  # subquestion identifier is irrelevant in this context
    df = df.drop('SubquestionTemplateRecordId', axis=1)  # subquestion identifier is irrelevant in this context

    # check for duplicate columns and remove them
    duplicateColumns = list()
    for column in df:
        if (df[column].nunique()) == 1 and column != "SubquestionPoints":
            df.drop(column, axis=1, inplace=True)
            duplicateColumns.append(1)
        else:
            duplicateColumns.append(0)
    
    # necessary preprocessing
    data = df[df.columns[:-1]]
    data = data.apply(
        lambda x: (x - x.mean()) / x.std()
    )

    data['SubquestionPoints'] = df.SubquestionPoints

    X = data.drop('SubquestionPoints', axis=1).to_numpy()
    y = data['SubquestionPoints'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model = load_model(model, login, X_train, y_train, retrainModel)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if len(arguments) > 1:
        if function_name == 'get_accuracy':
            get_accuracy(y_test, y_test_pred)
        elif function_name == 'predict_new':
            SubquestionTypeAveragePoints = locale.atof(arguments[5])
            CorrectAnswersShare = locale.atof(arguments[6])
            SubjectAveragePoints = locale.atof(arguments[7])
            ContainsImage = locale.atof(arguments[8])
            NegativePoints = locale.atof(arguments[9])
            MinimumPointsShare = locale.atof(arguments[10])
            predict_new(SubquestionTypeAveragePoints, CorrectAnswersShare, SubjectAveragePoints, ContainsImage,
                        NegativePoints, MinimumPointsShare, df, model, duplicateColumns)


if __name__ == '__main__':
        main(sys.argv)