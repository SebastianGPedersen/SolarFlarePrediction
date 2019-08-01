from typing import Iterable, Dict, Any
from abc import abstractmethod
from sklearn.metrics import f1_score
from General.Conventions import extract_labels, Datasets
import csv
from General.Paths import Gitlab_Path


class Evaluation_Model:
    """
    All models should be derived from this. Standardizes scoring methods.
    """

    def __init__(self, name: str, reproducible_arguments: Dict[str, Any]):
        self.name = name
        self.args = reproducible_arguments

    @abstractmethod
    def _Fit(self, Train_Datasets) -> None:
        pass

    @abstractmethod
    def _Predict(self, Test_Datasets) -> Iterable[float]:
        pass

    def __Score(self, Test_Datasets) -> float:
        pred = self._Predict(Test_Datasets)
        true = []
        for Dataset in Test_Datasets:
            true.extend(extract_labels(Dataset=Dataset, max_row=-1))

        return f1_score(y_true=true, y_pred=pred)

    def __Fit_and_Score(self, Train_Datasets: Iterable[str], Test_Datasets: Iterable[str]):
        Legal_Datasets = Datasets.Legal_Datasets()
        if not all(train in Legal_Datasets for train in Train_Datasets) or not all(
                test in Legal_Datasets for test in Test_Datasets):
            raise ValueError(
                'Invalid Dataset inputs: Train_Datasets={}, Test_Datasetss={}'.format(Train_Datasets, Test_Datasets))
        self._Fit(Train_Datasets=Train_Datasets)
        return self.__Score(Test_Datasets=Test_Datasets)

    def Report_Model(self, Full=True):

        if Full:
            Train_Cases = [[Datasets.fold2], [Datasets.fold2, Datasets.fold1], [Datasets.fold2, Datasets.fold3], [Datasets.fold1], [Datasets.fold1, Datasets.fold3]]
            Test_Cases = [[Datasets.fold1, Datasets.fold3], [Datasets.fold3], [Datasets.fold1], [Datasets.fold2, Datasets.fold3], [Datasets.fold2]]
        else:
            Train_Cases = [[Datasets.fold2]]
            Test_Cases = [[Datasets.fold1]]

        record_identifiers = [self.name, self.args]
        performances = [self.__Fit_and_Score(Train_Datasets=Train, Test_Datasets=Test) for Train, Test in
                        zip(Train_Cases, Test_Cases)]
        summaries = [sum(performances)/len(performances)]
        result = [record_identifiers+performances+summaries]

        if Full:
            file = 'Ranking.csv'
        else:
            file = 'Partial_Cases_Ranking.csv'

        with open(Gitlab_Path + '/Ranking/'+file, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerows(result)

        return result
