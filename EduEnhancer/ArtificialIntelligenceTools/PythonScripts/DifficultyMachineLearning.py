import pickle
import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import locale
import os
import json
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


def predict_new(subquestion_type_average_points, answer_correctness, subject_average_points, wrong_choice_points_share,
                negative_points, minimum_points_share, subquestion_points, df, model, duplicate_columns):
    final_tensor_values = list() #values from non-duplicate columns

    if duplicate_columns[0] != 1:
        subquestion_type_average_points_mean = df["SubquestionTypeAveragePoints"].mean()
        subquestion_type_average_points_std = df["SubquestionTypeAveragePoints"].std()
        if subquestion_type_average_points_std == 0:
            subquestion_type_average_points_std = 0.01
        subquestion_type_average_points = (subquestion_type_average_points
                                           - subquestion_type_average_points_mean) / subquestion_type_average_points_std
        final_tensor_values.append(subquestion_type_average_points)

    if duplicate_columns[1] != 1:
        answer_correctness_mean = df["AnswerCorrectness"].mean()
        answer_correctness_std = df["AnswerCorrectness"].std()
        if answer_correctness_std == 0:
            answer_correctness_std = 0.01
        answer_correctness = (answer_correctness - answer_correctness_mean) / answer_correctness_std
        final_tensor_values.append(answer_correctness)

    if duplicate_columns[2] != 1:
        subject_average_points_mean = df["SubjectAveragePoints"].mean()
        subject_average_points_std = df["SubjectAveragePoints"].std()
        if subject_average_points_std == 0:
            subject_average_points_std = 0.01
        subject_average_points = (subject_average_points - subject_average_points_mean) / subject_average_points_std
        final_tensor_values.append(subject_average_points)

    if duplicate_columns[3] != 1:
        wrong_choice_points_share_mean = df["WrongChoicePointsShare"].mean()
        wrong_choice_points_share_std = df["WrongChoicePointsShare"].std()
        if wrong_choice_points_share_std == 0:
            wrong_choice_points_share_std = 0.01
        wrong_choice_points_share = (wrong_choice_points_share - wrong_choice_points_share_mean) / wrong_choice_points_share_std
        final_tensor_values.append(wrong_choice_points_share)

    if duplicate_columns[4] != 1:
        negative_points_mean = df["NegativePoints"].mean()
        negative_points_std = df["NegativePoints"].std()
        if negative_points_std == 0:
            negative_points_std = 0.01
        negative_points = (negative_points - negative_points_mean) / negative_points_std
        final_tensor_values.append(negative_points)

    if duplicate_columns[5] != 1:
        minimum_points_share_mean = df["MinimumPointsShare"].mean()
        minimum_points_share_std = df["MinimumPointsShare"].std()
        if minimum_points_share_std == 0:
            minimum_points_share_std = 0.01
        minimum_points_share = (minimum_points_share - minimum_points_share_mean) / minimum_points_share_std
        final_tensor_values.append(minimum_points_share)

    x_unseen = torch.Tensor(final_tensor_values)
    y_unseen = model.predict(torch.atleast_2d(x_unseen))
    if round(y_unseen.item(), 2) > subquestion_points:
        return subquestion_points
    else:
        return round(y_unseen.item(), 2)


def load_model(model, login, X_train, y_train, retrain_model):
    base_path = Path(__file__)
    file_path_string = "../model/results/" + login + "_LR.sav"
    file_path = (base_path / file_path_string).resolve()

    if retrain_model is True:
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


def read_secrets() -> dict:
    base_path = Path(__file__)
    file_path_string = "../secrets/secrets.json"
    file_path = (base_path / file_path_string).resolve()
    try:
        with open(file_path, mode='r') as f:
            return json.loads(f.read())
    except FileNotFoundError:
        print("notfound")
        return {}


def main(arguments):
    if len(arguments) > 1:
        platform = arguments[1]
        login = arguments[2]
        retrain_model = eval(arguments[3])
        testTemplateId = arguments[4]
    else:
        platform = 'Windows'
        login = "login"
        retrain_model = False

    locale.setlocale(locale.LC_NUMERIC, 'cs_CZ.utf8')
    secrets = read_secrets()

    conn_str = ""
    if platform == 'Windows':
        conn_str = secrets["WINDOWS_CONNECTION_STRING"]
    elif platform == 'Linux':
        conn_str = secrets["LINUX_CONNECTION_STRING"]
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(connection_url)

    sql = "SELECT * FROM SubquestionResultRecord WHERE OwnerLogin = '" + login + "'"
    df = pd.read_sql(sql, engine)
    df = df.drop('OwnerLogin', axis=1)  # owner login is irrelevant in this context
    df = df.drop('SubquestionResultRecordId', axis=1)  # subquestion identifier is irrelevant in this context
    df = df.drop('SubquestionResultId', axis=1)  # subquestion identifier is irrelevant in this context

    # check for duplicate columns and remove them
    duplicate_columns = list()
    for column in df:
        if (df[column].nunique()) == 1 and column != "SubquestionPoints":
            df.drop(column, axis=1, inplace=True)
            duplicate_columns.append(1)
        else:
            duplicate_columns.append(0)

    # necessary preprocessing
    standardized_df = df[df.columns[:-1]]
    standardized_df = standardized_df.apply(
        lambda x: (x - x.mean()) / x.std()
    )

    standardized_df['StudentsPoints'] = df.StudentsPoints

    X = standardized_df.drop('StudentsPoints', axis=1).to_numpy()
    y = standardized_df['StudentsPoints'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model = load_model(model, login, X_train, y_train, retrain_model)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #selected testTemplate
    sql = "SELECT * FROM TestTemplate WHERE OwnerLogin = '" + login + "' AND TestTemplateId = '" + testTemplateId + "'"
    test_template_df = pd.read_sql(sql, engine)
    negative_points = test_template_df.iloc[0]['NegativePoints']
    minimum_points = test_template_df.iloc[0]['MinimumPoints']

    #question number identifiers of every question included in the test
    sql = "SELECT DISTINCT QuestionTemplateId FROM QuestionTemplate WHERE OwnerLogin = '" + login +\
          "' AND TestTemplateId = '" + testTemplateId + "'"
    question_template_id_list = pd.read_sql(sql, engine).values.tolist()
    question_template_ids = ""
    some_list_len = len(question_template_id_list)
    for i in range(some_list_len):
        #the question number identifier must be properly formatted in order to be used in a SQL query
        question_template_id = str(question_template_id_list[i])
        question_template_id = question_template_id[1:]
        question_template_id = question_template_id[:len(question_template_id) - 1]
        question_template_ids += question_template_id + ", "
    question_template_ids = question_template_ids[:len(question_template_ids) - 2]

    #test difficulty statistics (all relevant data needed to measure the test difficulty)
    sql = "SELECT * FROM TestDifficultyStatistics WHERE UserLogin = '" + login + "'"
    test_difficulty_statistics_df = pd.read_sql(sql, engine)
    subjects_ids_array = test_difficulty_statistics_df.iloc[0]['SubjectIds'].split("|")
    subject_average_points_array = test_difficulty_statistics_df.iloc[0]['SubjectAveragePoints'].split("|")
    subquestion_type_average_points_array =\
        test_difficulty_statistics_df.iloc[0]['SubquestionTypeAveragePoints'].split("|")
    subquestion_type_average_answer_correctness_array =\
        test_difficulty_statistics_df.iloc[0]['SubquestionTypeAverageAnswerCorrectness'].split("|")

    TestSubjectIndex = subjects_ids_array.index(str(test_template_df.iloc[0]['SubjectId']))

    #all subquestion templates included in the test
    sql = "SELECT * FROM SubquestionTemplate WHERE OwnerLogin = '" + login + "' AND QuestionTemplateId IN (" + \
          question_template_ids + ")"
    subquestion_templates_df = pd.read_sql(sql, engine)
    total_test_points = subquestion_templates_df['SubquestionPoints'].sum()
    minimum_points_share = minimum_points / total_test_points
    predicted_test_points = 0

    for i in range(subquestion_templates_df.shape[0]):
        subquestion_template = subquestion_templates_df.iloc[i]
        subquestion_points = subquestion_template["SubquestionPoints"]
        wrong_choice_points_share = subquestion_template["WrongChoicePoints"] / \
                            subquestion_template["DefaultWrongChoicePoints"]

        subquestion_type_average_points = locale.atof(
            subquestion_type_average_points_array[subquestion_template["SubquestionType"] - 1])
        answer_correctness = locale.atof(
            subquestion_type_average_answer_correctness_array[subquestion_template["SubquestionType"] - 1])
        subject_average_points = locale.atof(subject_average_points_array[TestSubjectIndex])
        predicted_test_points += predict_new(subquestion_type_average_points, answer_correctness,
                                             subject_average_points, wrong_choice_points_share, negative_points,
                                             minimum_points_share, subquestion_points, df, model, duplicate_columns)
    print(round(predicted_test_points, 2))


if __name__ == '__main__':
        main(sys.argv)