import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from db_conn import *

table_name = 'loan'


class class_loan_classification():
    def __init__(self):
        pass

    def import_loan_data(self):
        conn, cur = open_db()

        create_sql = f""" 
            drop table if exists {table_name};
        
            create table {table_name} (
                id int auto_increment primary key,
                loan_id varchar(20),
                gender varchar(10),
                married boolean,
                dependents int,
                education varchar(20),
                self_employed boolean,
                applicant_income int,
                coapplicant_income int,
                loan_amount int,
                loan_amount_term int,
                credit_history int,
                property_area varchar(20),
                loan_status boolean
                ); 
        """

        cur.execute(create_sql)
        conn.commit()

        file_name = 'loan prediction.csv'
        loan_data = pd.read_csv(file_name)

        rows = []

        insert_sql = f"""insert into {table_name} (
                loan_id, 
                gender,
                married,
                dependents,
                education,
                self_employed,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_amount_term,
                credit_history,
                property_area,
                loan_status)
                values(%s,%s,%s = 'Yes',left (%s, 1),%s,%s = 'Yes',%s,%s,%s,%s,%s,%s,%s = 'Y');
        """

        loan_data.replace({np.nan: None}, inplace=True)
        rows = [r for r in loan_data.itertuples(index=False, name=None)]

        cur.executemany(insert_sql, rows)
        conn.commit()

        close_db(conn, cur)

    def load_data_for_binary_classification(self):
        conn, cur = open_db()

        sql = f"select * from {table_name};"
        cur.execute(sql)

        data = cur.fetchall()

        close_db(conn, cur)

        self.X = [(t['married'], t['dependents'], t['self_employed'],
                   t['credit_history'])
                  for t in data]
        self.X = [[0 if t is None else t for t in data] for data in self.X]

        self.X = np.array(self.X)

        self.y = [t['loan_status'] for t in data]
        self.y = np.array(self.y)

    def data_split_train_test(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

        '''
        print("X_train=", self.X_train)
        print("X_test=", self.X_test)
        print("y_train=", self.y_train)
        print("y_test=", self.y_test)
        '''

    def classification_performance_eval_binary(self, y_test, y_predict):
        tp, tn, fp, fn = 0, 0, 0, 0

        for y, yp in zip(y_test, y_predict):
            if y == 1 and yp == 1:
                tp += 1
            elif y == 1 and yp == 0:
                fn += 1
            elif y == 0 and yp == 1:
                fp += 1
            else:
                tn += 1

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)

        print("accuracy=%f" % accuracy)
        print("precision=%f" % precision)
        print("recall=%f" % recall)
        print("f1 score=%f" % f1_score)

    def train_and_test_svm_model(self):
        dsvm = svm.SVC()
        dsvm.fit(self.X_train, self.y_train)
        self.y_predict = dsvm.predict(self.X_test)

    def svm_KFold_performance(self):
        dsvm = svm.SVC()
        cv_results = cross_validate(dsvm, self.X, self.y, cv=5,
                                    scoring=['accuracy', 'precision', 'recall',
                                             'f1'])

        print(cv_results)

        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                print(f'\n{metric[5:]}: {scores.mean():.2f}')

    def train_and_test_logistic_regression_model(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)

    def logistic_regression_KFold_performance(self):
        model = LogisticRegression()
        cv_results = cross_validate(model, self.X, self.y, cv=5,
                                    scoring=['accuracy', 'precision', 'recall',
                                             'f1'])

        print(cv_results)

        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                print(f'\n{metric[5:]}: {scores.mean():.2f}')

    def train_and_test_random_forest_model(self):
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)

    def random_forest_KFold_performance(self):
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        cv_results = cross_validate(model, self.X, self.y, cv=5,
                                    scoring=['accuracy', 'precision', 'recall',
                                             'f1'])

        print(cv_results)

        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                print(f'\n{metric[5:]}: {scores.mean():.2f}')


def binary_svm_train_test_performance():
    clf = class_loan_classification()
    clf.load_data_for_binary_classification()
    clf.data_split_train_test()
    clf.train_and_test_svm_model()
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)


def binary_svm_KFold_performance():
    clf = class_loan_classification()
    clf.load_data_for_binary_classification()
    clf.svm_KFold_performance()


def binary_logistic_regression_train_test_performance():
    clf = class_loan_classification()
    clf.load_data_for_binary_classification()
    clf.data_split_train_test()
    clf.train_and_test_logistic_regression_model()
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)


def binary_logistic_regression_KFold_performance():
    clf = class_loan_classification()
    clf.load_data_for_binary_classification()
    clf.logistic_regression_KFold_performance()


def binary_random_forest_train_test_performance():
    clf = class_loan_classification()
    clf.load_data_for_binary_classification()
    clf.data_split_train_test()
    clf.train_and_test_logistic_regression_model()
    clf.classification_performance_eval_binary(clf.y_test, clf.y_predict)


def binary_random_forest_KFold_performance():
    clf = class_loan_classification()
    clf.load_data_for_binary_classification()
    clf.logistic_regression_KFold_performance()


if __name__ == "__main__":
    clf = class_loan_classification()
    clf.import_loan_data()

    print("Binary SVM Train Test Performance")
    binary_svm_train_test_performance()
    binary_svm_KFold_performance()

    print("Binary Logistic Regression Train Test Performance")
    binary_logistic_regression_train_test_performance()
    binary_logistic_regression_KFold_performance()

    print("Binary Random Forest Train Test Performance")
    binary_random_forest_train_test_performance()
    binary_random_forest_KFold_performance()
