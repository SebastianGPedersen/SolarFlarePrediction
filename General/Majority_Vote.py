def Probs_To_Preds(Probs, level):
    return [int(p >= level) for p in Probs]


def Majority_Vote_Binary(*predictions):
    """
    :param args: Iterables of predictions
    :return: Majority vote
    """

    x = len(predictions)
    return [sum(a) // x for a in zip(*predictions)]