from typing import List, Generator
from _0_DataCreation.Read_Data import load_data
from General.Paths import Data_Path


class Datasets:
    fold1 = 'fold1'
    fold2 = 'fold2'
    fold3 = 'fold3'
    test = 'testSet'

    @classmethod
    def Legal_Datasets(cls) -> List[str]:
        return [cls.fold1, cls.fold2, cls.fold3, cls.test]


def extract_labels(Dataset: str, max_row=-1) -> List[int]:
    return [label for id, label, data in Load_Main_Dataset(Dataset=Dataset, max_row=max_row)]


def Load_Main_Dataset(Dataset: str, max_row=-1) -> Generator:
    return load_data(filename=Data_Path + '/' + Dataset + '_NA.dat', max_row=max_row)



