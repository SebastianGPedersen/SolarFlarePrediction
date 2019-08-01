from General.Evaluation_Model import Evaluation_Model
from General.Conventions import Load_Main_Dataset


class Same_Label(Evaluation_Model):
    """
    Trivial model that just sets the label to either 0 or 1
    """

    def __init__(self, Label: int=1):
        if not Label in [0,1]:
            raise ValueError('Only labels 0 & 1 allowed, not: {}'.format(Label))
        super(Same_Label, self).__init__(name=self.__class__.__name__, reproducible_arguments={'Label': Label})
        self.Label = Label

    def _Fit(self, Train_Datasets):
        pass

    def _Predict(self, Test_Datasets):
        pred = []
        for Test_Dataset in Test_Datasets:
            pred.extend([self.Label for _ in Load_Main_Dataset(Dataset=Test_Dataset, max_row=-1)])

        return pred


if __name__ == '__main__':
    all_ones = Same_Label(Label=1)
    all_ones.Report_Model()
