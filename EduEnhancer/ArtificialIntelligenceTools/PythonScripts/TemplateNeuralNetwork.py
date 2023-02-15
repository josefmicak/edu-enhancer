import sys

from sklearn.ensemble import GradientBoostingRegressor
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


def predict_new(subquestion_type_average_points, correct_answers_share, subject_average_points, wrong_choice_points_share,
                negative_points, minimum_points_share, df, model, duplicate_columns):
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
        correct_answers_share_mean = df["CorrectAnswersShare"].mean()
        correct_answers_share_std = df["CorrectAnswersShare"].std()
        if correct_answers_share_std == 0:
            correct_answers_share = 0.01
        correct_answers_share = (correct_answers_share - correct_answers_share_mean) / correct_answers_share_std
        final_tensor_values.append(correct_answers_share)

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
    base_path = Path(__file__)
    file_path_string = "../model/templates/" + login + "_NN.pt"
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
    base_path = Path(__file__)
    file_path_string = "../model/templates/" + login + "_NN.pt"
    file_path = (base_path / file_path_string).resolve()
    torch.save(model.state_dict(), file_path)


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
        function_name = arguments[4]
        show_graphs = False
    else:
        platform = 'Windows'
        login = "login"
        retrain_model = False
        show_graphs = True

    locale.setlocale(locale.LC_NUMERIC, 'cs_CZ.utf8')
    secrets = read_secrets()

    conn_str = ""
    if platform == 'Windows':
        conn_str = secrets["WINDOWS_CONNECTION_STRING"]
    elif platform == 'Linux':
        conn_str = secrets["LINUX_CONNECTION_STRING"]

    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
    engine = create_engine(connection_url)

    sql = "SELECT * FROM SubquestionTemplateRecord WHERE OwnerLogin = '" + login + "'"
    df = pd.read_sql(sql, engine)
    df = df.drop('OwnerLogin', axis=1)  # owner login is irrelevant in this context
    df = df.drop('SubquestionTemplateId', axis=1)  # subquestion identifier is irrelevant in this context
    df = df.drop('SubquestionTemplateRecordId', axis=1)  # subquestion identifier is irrelevant in this context

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

    standardized_df['SubquestionPoints'] = df.SubquestionPoints

    X = standardized_df.drop('SubquestionPoints', axis=1).to_numpy()
    y = standardized_df['SubquestionPoints'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    x = torch.tensor(np.array(X_train), dtype=torch.float)
    y = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float)

    model = NeuralNetwork(X.shape[1], 32, 16)
    model = load_model(model, login, x, y, retrain_model)

    y_train_pred = model(torch.tensor(X_train, dtype=torch.float))
    y_test_pred = model(torch.tensor(X_test, dtype=torch.float))

    y_train_pred = y_train_pred.detach().numpy()
    y_test_pred = y_test_pred.detach().numpy()

    if show_graphs:
        import matplotlib.pylab as plt
        import seaborn as sb

        pd.set_option('display.max_columns', None)
        # table of unstandardized data (Table 8.1)
        print(df.head())

        # table of standardized data (Table 8.2)
        print(standardized_df.head())

        standardized_df_correlation = standardized_df.corr(method='pearson')

        # heatmap of correlation (Figure 8.6)
        ax = sb.heatmap(standardized_df_correlation,
                   xticklabels=standardized_df_correlation.columns,
                   yticklabels=standardized_df_correlation.columns,
                   cmap='coolwarm',
                   annot=True,
                   linewidth=0.5)
        ax.figure.tight_layout()
        plt.show()

        # feature importance (Figure 8.7)
        rc = {'axes.facecolor': 'white',
              'axes.grid': False,
              'font.family': 'Times New Roman',
              'font.size': 12}
        plt.rcParams.update(rc)
        y = ['SubquestionTypeAveragePoints', 'CorrectAnswersShare',
                                             'SubjectAveragePoints', 'WrongChoicePointsShare',
                                             'NegativePoints', 'MinimumPointsShare']
        gb = GradientBoostingRegressor(n_estimators=100)
        gb.fit(X_train, y_train.ravel())
        x = gb.feature_importances_
        plt.barh(y, x)
        plt.ylabel("vstupní proměnné")
        plt.xlabel("relevance")
        plt.tight_layout()
        plt.show()

        # training loss
        model = NeuralNetwork(X.shape[1], 32, 16)
        epochs = 500
        lr = 0.05
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        x = torch.tensor(np.array(X_train), dtype=torch.float)
        y = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float)
        train_loss_vals = []
        for i in range(epochs):
            optimizer.zero_grad()
            y_value = model(x)
            loss_value = loss_function(y_value, y)
            loss_value.backward()
            train_loss_vals.append(loss_value.item())
            optimizer.step()

        # testing loss
        model = NeuralNetwork(X.shape[1], 32, 16)
        epochs = 500
        lr = 0.05
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        x = torch.tensor(np.array(X_test), dtype=torch.float)
        y = torch.tensor(np.array(y_test).reshape(-1, 1), dtype=torch.float)
        test_loss_vals = []
        for i in range(epochs):
            optimizer.zero_grad()
            y_value = model(x)
            loss_value = loss_function(y_value, y)
            loss_value.backward()
            test_loss_vals.append(loss_value.item())
            optimizer.step()

        # loss graph (Figure 8.8)
        plt.plot(
            np.array(train_loss_vals).reshape((epochs, -1)).mean(axis=1),
            color="r",
            label='trénovací chyba'
        )
        plt.plot(
            np.array(test_loss_vals).reshape((epochs, -1)).mean(axis=1),
            color="b",
            label='testovací chyba'
        )
        plt.xlabel('epochy')
        plt.ylabel('chyba')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # R2 graph (Figure 8.9)
        model = NeuralNetwork(X.shape[1], 32, 16)
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        x = torch.tensor(np.array(X_test), dtype=torch.float)
        y = torch.tensor(np.array(y_test).reshape(-1, 1), dtype=torch.float)
        r2_train_vals = []
        r2_test_vals = []
        epochs = 500

        for i in range(epochs):
            optimizer.zero_grad()
            y_value = model(x)
            loss_value = loss_function(y_value, y)
            loss_value.backward()
            optimizer.step()
            y_train_pred = model(torch.tensor(X_train, dtype=torch.float))
            y_test_pred = model(torch.tensor(X_test, dtype=torch.float))

            y_train_pred = y_train_pred.detach().numpy()
            y_test_pred = y_test_pred.detach().numpy()
            R2_train = r2_score(y_train, y_train_pred)
            r2_train_vals.append(R2_train)
            R2_test = r2_score(y_test, y_test_pred)
            r2_test_vals.append(R2_test)

        plt.plot(
            np.array(r2_train_vals).reshape((epochs, -1)).mean(axis=1),
            color="r",
            label='trénovací skóre',
            linewidth=0.5
        )
        plt.plot(
            np.array(r2_test_vals).reshape((epochs, -1)).mean(axis=1),
            color="b",
            label='testovací skóre',
            linewidth=0.5
        )
        ax = plt.gca()
        ax.set_ylim([0, 1])
        plt.xlabel('epochy')
        plt.ylabel('skóre')
        plt.tight_layout()
        plt.legend()
        plt.show()

        # graph of actual subquestionPoints and predicted subquestionPoints (Figure 8.10)
        rc = {'axes.facecolor': 'white',
              'axes.grid': True,
              'grid.color': 'silver',
              'font.family': 'Times New Roman',
              'font.size': 12}
        plt.rcParams.update(rc)

        plt.figure(figsize=(5, 5), dpi=100)
        plt.xlim(0, 25)
        plt.ylim(0, 25)
        plt.scatter(y_train, y_train_pred, lw=1, color="r", label="trénovací data")
        plt.scatter(y_test, y_test_pred, lw=1, color="b", label="testovací data")
        plt.xlabel("SubquestionPoints")
        plt.ylabel("predikované SubquestionPoints")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if len(arguments) > 1:
        if function_name == 'get_accuracy':
            get_accuracy(y_test, y_test_pred)
        elif function_name == 'predict_new':
            subquestion_type_average_points = locale.atof(arguments[5])
            correct_answers_share = locale.atof(arguments[6])
            subject_average_points = locale.atof(arguments[7])
            wrong_choice_points_share = locale.atof(arguments[8])
            negative_points = locale.atof(arguments[9])
            minimum_points_share = locale.atof(arguments[10])
            predict_new(subquestion_type_average_points, correct_answers_share, subject_average_points, wrong_choice_points_share,
                        negative_points, minimum_points_share, df, model, duplicate_columns)


if __name__ == '__main__':
        main(sys.argv)