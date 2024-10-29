import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from db_conn import *


class class_iris_classification():
    def __init__(self):
        pass

    def import_iris_data(self):
        conn, cur = open_db()

        create_sql = """ 
            drop table if exists iris;
        
            create table iris (
                id int auto_increment primary key,
                sepal_length float,
                sepal_width float,
                petal_length float,
                petal_width float,
                species varchar(10), 
                enter_date datetime default now() 
                ); 
        """

        cur.execute(create_sql)
        conn.commit()

        file_name = 'iris.csv'
        iris_data = pd.read_csv(file_name)

        rows = []

        insert_sql = """insert into iris(sepal_length, sepal_width, petal_length, petal_width, species)
                        values(%s,%s,%s,%s,%s);"""

        rows = [r for r in iris_data.itertuples(index=False, name=None)]

        cur.executemany(insert_sql, rows)
        conn.commit()

        close_db(conn, cur)

    def load_data_for_multiclass_classification(self):

        conn, cur = open_db()

        sql = "select * from iris;"
        cur.execute(sql)

        data = cur.fetchall()

        self.X = [(t['sepal_length'], t['sepal_width'], t['petal_length'],
                   t['petal_width']) for t in data]
        # self.X = [ (t['sepal_length'], t['sepal_width'] ) for t in data ]
        # self.X = [ (t['sepal_length'], t['petal_length'] ) for t in data ]

        self.X = np.array(self.X)

        self.y = [0 if t['species'] == 'setosa' else 1 if t[
                                                              'species'] == 'versicolor' else 2
                  for t in data]
        self.y = np.array(self.y)

    def data_split_train_test(self):

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.4, random_state=None)

    def classification_performance_eval_multiclass(self, y_test, y_predict,
        verbose=True):
        target_names = ['setosa', 'versicolor', 'virginica']
        labels = [0, 1, 2]

        self.confusion_matrix = confusion_matrix(y_test, y_predict,
                                                 labels=labels)
        self.classification_report = classification_report(y_test, y_predict,
                                                           target_names=target_names,
                                                           labels=labels,
                                                           output_dict=True)

        # 출력 옵션 처리
        if verbose:
            print(f"[confusion_matrix]\n{self.confusion_matrix}")
            classification_report_for_print = classification_report(y_test,
                                                                    y_predict,
                                                                    target_names=target_names,
                                                                    labels=labels,
                                                                    output_dict=False)
            print(
                f"\n[classification_report]\n{classification_report_for_print}")

    def train_and_test_dtree_model(self):
        dtree = tree.DecisionTreeClassifier()
        dtree_model = dtree.fit(self.X_train, self.y_train)
        self.y_predict = dtree_model.predict(self.X_test)

    def multiclass_dtree_KFold_performance(self):
        kfold_reports = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        i = 0
        for train_index, test_index in kf.split(self.X):
            i += 1
            print(f"\n\nFold {i} :")

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            dtree = tree.DecisionTreeClassifier()
            dtree_model = dtree.fit(X_train, y_train)
            y_predict = dtree_model.predict(X_test)

            # 성능 평가 후 kfold_reports에 저장
            self.classification_performance_eval_multiclass(y_test, y_predict,
                                                            verbose=True)
            kfold_reports.append(
                pd.DataFrame(self.classification_report).transpose())

        # 모든 폴드의 평균 성능을 계산하고 출력
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print(f'\n\n\n[Mean report]\n{mean_report}')


def multiclass_dtree_train_test_performance():
    clf = class_iris_classification()
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_dtree_model()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)


def multiclass_dtree_KFold_performance():
    clf = class_iris_classification()
    clf.load_data_for_multiclass_classification()
    clf.multiclass_dtree_KFold_performance()


if __name__ == "__main__":
    multiclass_dtree_train_test_performance()
    multiclass_dtree_KFold_performance()
