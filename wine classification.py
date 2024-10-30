import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from db_conn import *

table_name = 'wine'


class class_wine_classification():
    def __init__(self):
        pass

    def import_wine_data(self):
        conn, cur = open_db()

        create_sql = f""" 
            drop table if exists {table_name};
        
            create table {table_name} (
                id int auto_increment primary key,
                class int,
                alcohol float,
                malic_acid float,
                ash float,
                alcalinity_of_ash float,
                magnesium int,
                total_phenols float,
                flavanoids float,
                nonflavanoid_phenols float,
                proanthocyanins float,
                color_intensity float,
                hue float,
                od280_od315 float,
                proline int
                ); 
        """

        cur.execute(create_sql)
        conn.commit()

        file_name = 'wine.data.csv'
        wine_data = pd.read_csv(file_name)

        rows = []

        insert_sql = f"""insert into {table_name}(class, alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, od280_od315, proline)
                        values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"""

        rows = [r for r in wine_data.itertuples(index=False, name=None)]

        cur.executemany(insert_sql, rows)
        conn.commit()

        close_db(conn, cur)

    def load_data_for_multiclass_classification(self):

        conn, cur = open_db()

        sql = f"select * from {table_name};"
        cur.execute(sql)

        data = cur.fetchall()

        self.X = [(t['alcohol'], t['malic_acid'], t['ash'],
                   t['alcalinity_of_ash'], t['magnesium'], t['total_phenols'],
                   t['flavanoids'], t['nonflavanoid_phenols'],
                   t['proanthocyanins'], t['color_intensity'], t['hue'],
                   t['od280_od315'], t['proline']) for t in data]

        self.X = np.array(self.X)

        self.y = [t['class'] for t in data]
        self.y = np.array(self.y)

    def data_split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def classification_performance_eval_multiclass(self, y_test, y_predict,
        verbose=True):
        target_names = ['1', '2', '3']
        labels = [1, 2, 3]

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

    def train_and_test_svm_model(self):
        model = svm.SVC()
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)

    def multiclass_svm_KFold_performance(self):
        kfold_reports = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        i = 0
        for train_index, test_index in kf.split(self.X):
            i += 1
            print(f"\n\nFold {i} :")

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model = svm.SVC()
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)

            # 성능 평가 후 kfold_reports에 저장
            self.classification_performance_eval_multiclass(y_test, y_predict,
                                                            verbose=True)
            kfold_reports.append(
                pd.DataFrame(self.classification_report).transpose())

        # 모든 폴드의 평균 성능을 계산하고 출력
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print(f'\n\n\n[Mean report]\n{mean_report}')

    def train_and_test_logistic_regression_model(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)

    def multiclass_logistic_regression_KFold_performance(self):
        kfold_reports = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        i = 0
        for train_index, test_index in kf.split(self.X):
            i += 1
            print(f"\n\nFold {i} :")

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)

            # 성능 평가 후 kfold_reports에 저장
            self.classification_performance_eval_multiclass(y_test, y_predict,
                                                            verbose=True)
            kfold_reports.append(
                pd.DataFrame(self.classification_report).transpose())

        # 모든 폴드의 평균 성능을 계산하고 출력
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print(f'\n\n\n[Mean report]\n{mean_report}')

    def train_and_test_random_forest_model(self):
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        model.fit(self.X_train, self.y_train)
        self.y_predict = model.predict(self.X_test)

    def multiclass_random_forest_KFold_performance(self):
        kfold_reports = []

        kf = KFold(n_splits=5, random_state=42, shuffle=True)

        i = 0
        for train_index, test_index in kf.split(self.X):
            i += 1
            print(f"\n\nFold {i} :")

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model = RandomForestClassifier(n_estimators=5, random_state=0)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)

            # 성능 평가 후 kfold_reports에 저장
            self.classification_performance_eval_multiclass(y_test, y_predict,
                                                            verbose=True)
            kfold_reports.append(
                pd.DataFrame(self.classification_report).transpose())

        # 모든 폴드의 평균 성능을 계산하고 출력
        mean_report = pd.concat(kfold_reports).groupby(level=0).mean()
        print(f'\n\n\n[Mean report]\n{mean_report}')


def multiclass_svm_train_test_performance():
    clf = class_wine_classification()
    clf.import_wine_data()
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_svm_model()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)


def multiclass_svm_KFold_performance():
    clf = class_wine_classification()
    clf.import_wine_data()
    clf.load_data_for_multiclass_classification()
    clf.multiclass_svm_KFold_performance()


def multiclass_logistic_regression_train_test_performance():
    clf = class_wine_classification()
    clf.import_wine_data()
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_logistic_regression_model()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)


def multiclass_logistic_regression_KFold_performance():
    clf = class_wine_classification()
    clf.import_wine_data()
    clf.load_data_for_multiclass_classification()
    clf.multiclass_logistic_regression_KFold_performance()


def multiclass_random_forest_train_test_performance():
    clf = class_wine_classification()
    clf.import_wine_data()
    clf.load_data_for_multiclass_classification()
    clf.data_split_train_test()
    clf.train_and_test_random_forest_model()
    clf.classification_performance_eval_multiclass(clf.y_test, clf.y_predict)


def multiclass_random_forest_KFold_performance():
    clf = class_wine_classification()
    clf.import_wine_data()
    clf.load_data_for_multiclass_classification()
    clf.multiclass_random_forest_KFold_performance()


if __name__ == "__main__":
    print(
        "========================================multiclass_svm_train_test_performance========================================")
    multiclass_svm_train_test_performance()
    multiclass_svm_KFold_performance()

    print(
        "========================================multiclass_logistic_regression_train_test_performance========================================")
    multiclass_logistic_regression_train_test_performance()
    multiclass_logistic_regression_KFold_performance()

    print(
        "========================================multiclass_random_forest_train_test_performance========================================")
    multiclass_random_forest_train_test_performance()
    multiclass_random_forest_KFold_performance()
