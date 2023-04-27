import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import locale
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pathlib import Path


class NeuralNetwork(nn.Module):
    def __init__(self, variables_count, xA, xB):
        """
        Neural network initialization, appropriate layers are created

        Parameters:
        variables_count (int): Amount of input variables - input size of first layer
        xA (int): Output size of first layer
        xB (int): Output size of second layer

        """
        super(NeuralNetwork, self).__init__()
        self.linearA = nn.Linear(variables_count, xA)
        self.linearB = nn.Linear(xA, xB)
        self.linearC = nn.Linear(xB, 1)

    def forward(self, x):
        """
        Neural network definition

        Parameters:
        x (int): Input data

        """
        yA = F.relu(self.linearA(x))
        yB = F.relu(self.linearB(yA))
        return self.linearC(yB)


def train(model, lr, epochs, x, y):
    """
    Trains neural network model

    Parameters:
    model (NeuralNetwork): Neural network model used to perform computations
    lr (float): Learning rate
    epochs (int): Amount of epochs to be performed
    x (DataFrame): Input dataset
    y (DataFrame): Output dataset

    """
    loss_function = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()
        y_value = model(x)
        loss_value = loss_function(y_value, y)
        loss_value.backward()
        optimizer.step()


def get_accuracy(y_test, y_test_pred):
    """
    Returns accuracy (R-squared score) of the model

    Parameters:
    y_test (DataFrame): Testing output data
    y_test_pred (DataFrame): Predicted results that are compared to the actual data

    """
    R2 = r2_score(y_test, y_test_pred)
    print(R2)


def predict_new(subquestion_type_average_points, answer_correctness, subject_average_points, wrong_choice_points_share,
                negative_points, minimum_points_share, df, model, duplicate_columns):
    """
    Returns the predicted amount of points that the student should get for the subquestion

    Parameters:
    subquestion_type_average_points (float): Average subquestion points awarded for this subquestion type
    answer_correctness (float): Answer correctness (-1 to 1)
    subject_average_points (float): Answer correctness (-1 to 1)
    wrong_choice_points_share (float): Share of wrong choice points to maximum possible wrong choice points
    negative_points (int): Negative points settings (1/2/3)
    minimum_points_share (float): Share of minimum test points to total test points
    df (DataFrame): Dataset from SubquestionTemplateRecord table
    model (NeuralNetwork): Neural network model used to perform computations
    duplicate_columns (list): List of integers (0/1) indicate which column's values are identical within the column

    """
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
    y_unseen = model(torch.atleast_2d(x_unseen))
    print(round(y_unseen.item(), 2))


def load_model(model, login, x, y, retrain_model):
    """
    Loads model (loads from appropriate path or trains and saves a new model)

    Parameters:
    model (NeuralNetwork): Neural network model used to perform computations
    login (str): User login
    x (DataFrame): Input data
    y (DataFrame): Output data
    retrain_model (bool): Indicates whether model should be retrained or not

    """
    base_path = Path(__file__)
    file_path_string = "../model/results/" + login + "_NN.pt"
    file_path = (base_path / file_path_string).resolve()

    if retrain_model is True:
        train(model, 0.05, 500, x, y)
        save_model(model, login)
    else:
        try:  # a file with trained model already exists for this user
            model.load_state_dict(torch.load(file_path))
        except:  # a file with trained model does not exist for this user - it's necessary to create a new file
            train(model, 0.05, 500, x, y)
            save_model(model, login)
    return model


def save_model(model, login):
    """
    Saves trained model to appropriate path

    Parameters:
    model (NeuralNetwork): Neural network model used to perform computations
    login (str): User login

    """
    base_path = Path(__file__)
    file_path_string = "../model/results/" + login + "_NN.pt"
    file_path = (base_path / file_path_string).resolve()
    torch.save(model.state_dict(), file_path)


def read_secrets() -> dict:
    """
    Loads Windows and Ubuntu connection strings

    """
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
        function_name = arguments[4]
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
    df = df.drop('SubquestionResultId', axis=1)  #subquestion result identifier is irrelevant in this context
    df = df.drop('SubquestionResultRecordId', axis=1)  # subqustion identifier is irrelevant in this context

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

    x = torch.tensor(np.array(X_train), dtype=torch.float)
    y = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float)

    model = NeuralNetwork(X.shape[1], 32, 16)
    model = load_model(model, login, x, y, retrain_model)

    y_train_pred = model(torch.tensor(X_train, dtype=torch.float))
    y_test_pred = model(torch.tensor(X_test, dtype=torch.float))

    y_train_pred = y_train_pred.detach().numpy()
    y_test_pred = y_test_pred.detach().numpy()

    if len(arguments) > 1:
        if function_name == 'get_accuracy':
            get_accuracy(y_test, y_test_pred)
        elif function_name == 'predict_new':
            subquestion_type_average_points = locale.atof(arguments[5])
            answer_correctness = locale.atof(arguments[6])
            subject_average_points = locale.atof(arguments[7])
            wrong_choice_points_share = locale.atof(arguments[8])
            negative_points = locale.atof(arguments[9])
            minimum_points_share = locale.atof(arguments[10])
            predict_new(subquestion_type_average_points, answer_correctness, subject_average_points, wrong_choice_points_share,
                        negative_points, minimum_points_share, df, model, duplicate_columns)


if __name__ == '__main__':
        main(sys.argv)