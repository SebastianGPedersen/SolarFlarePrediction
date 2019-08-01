from typing import Optional, Tuple, Dict, Generator, List
from General.Paths import Data_Path
import pickle
from sys import getsizeof
from _0_DataCreation.Raw_Data_Transformations import scale_df
import pandas as pd
import numpy as np

def load_dataframe(filename: str) -> pd.DataFrame:
    """
    Created to load the foldX_extracted.dat files
    :param filename: Only name (not path including)
    :return: A dataframe of features
    """

    load_path = Data_Path + "/" + filename

    with open(load_path, "rb") as f:
        header = pickle.load(f)
    my_df = pd.DataFrame([line for line in load_data(load_path, max_row=-1, header=True)], columns=header)

    print("Size: " + str(round(getsizeof(my_df)/2**20, 2)) + "Mb")

    return my_df

def batch_generator(filename: str, batch_size: int, num_features: int=10,
                    time_steps: int=60, header: bool=False) -> Generator:
    
    while True:
        with open(filename, 'rb') as f:
            if header:
                _ = pickle.load(f)
            while True:
                try:
                    X = np.zeros((batch_size, time_steps, 16))
                    y = np.zeros(batch_size)
                    i = 0
                    while i < batch_size:
                        _, y[i], X[i, :, :] = pickle.load(f)
                        i += 1
                    X = X[:, :, 0:num_features]
                    yield X, y
                except EOFError:
                    break
                
def batch_generator_test(filename: str, batch_size: int, num_features: int=10,
                    time_steps: int=60, header: bool=False) -> Generator:
    
    while True:
        with open(filename, 'rb') as f:
            if header:
                _ = pickle.load(f)
            while True:
                try:
                    X = np.zeros((batch_size, time_steps, 16))
                    i = 0
                    while i < batch_size:
                        _, X[i, :, :] = pickle.load(f)
                        i += 1
                    X = X[:, :, 0:num_features]
                    yield X
                except EOFError:
                    break 

def batch_generator2(filenames: List, batch_size: int, num_features: int=10,
                    time_steps: int=60, header: bool=False) -> Generator:
    
    while True:
        for filename in filenames:
            with open(filename, 'rb') as f:
                if header:
                    _ = pickle.load(f)
                while True:
                    try:
                        X = np.zeros((batch_size, time_steps, 16))
                        y = np.zeros(batch_size)
                        i = 0
                        while i < batch_size:
                            _, y[i], X[i, :, :] = pickle.load(f)
                            i += 1
                        X = X[:, :, 0:num_features]
                        yield X, y
                    except EOFError:
                        break #Out of while loop and next filename


def load_data(filename: str, max_row: int=-1, header: bool=False) -> Generator:
    """
    Creates a generator reading (using pickle) from file.

    :param filename: file to read
    :param max_row: Max number of rows/observations to read. If -1, reads all
    :param header: If True, skips the first observation
    :return: A generator
    """

    if max_row == -1:
        with open(filename, "rb") as f:  # mode = '(r)ead (b)yte'
            if header:
                _ = pickle.load(f)
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:  # Stop at end of file
                    break

    elif max_row > 0:
        with open(filename, "rb") as f:  # mode = '(r)ead (b)yte'
            if header:
                _ = pickle.load(f)
            for _ in range(max_row):
                try:
                    yield pickle.load(f)
                except EOFError:  # Stop at end of file
                    pass
    else:
        raise ValueError('max_row={}, not allowed.'.format(max_row))


def repeat_iter(gen_func, args: Optional[Tuple]=None, kwargs: Optional[Dict]=None, n_times: int=-1):
    """
    Creates generator which starts over once iterated through
    :param gen_func: A function return a generator, such as load_data
    :param args: arguments passed to gen_func
    :param kwargs: keyword arguments passed to gen_func
    :param n_times: Number of times to repeat the iterator, if -1 (or negative in general), repeats forever
    :return: A generator
    """
    # Handle default arguments
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    i = 0
    while True:
            yield from gen_func(*args, **kwargs)
            if i == n_times:
                break
            i += 1


def load_data_batch(filename: str, max_row: int=-1, header: bool=False, batchsize=1):
    ret = []
    i = 0
    for data in load_data(filename=filename, max_row=max_row, header=header):
        ret.append(data)
        i += 1
        if i == batchsize:
            yield ret
            ret = []
            i = 0

    if ret:
        yield ret


fn = Data_Path + '/fold1_NA.dat'  # obs = 0..76773
fn2 = Data_Path + '/fold2_NA.dat'  # obs = 0..92481
fn3 = Data_Path + '/fold3_NA.dat'  # obs = 0..27006
fn4 = Data_Path + '/testSet_NA.dat'  # obs = 0..173512

if __name__ == '__main__':
    from time import time

    # fn == filename

    # Test examples
    # labels = []
    # start = time()
    # for id, label, data in load_data(filename=fn2, max_row=-1):
    #     labels.append(label)
    #
    # print(time() - start)
    if True:
        for id, label, data in load_data(filename=fn, max_row=1):
            print('id={id} label={label}'.format(id=id, label=label))
            test = scale_df(data)
            print(type(data))
            print(data.columns)
            # print(print(type(data)))
            #test = data_to_keep(data, 10, list(range(6))+list(range(start=11, stop=16)))
            #print(test)
            # print(type(test))

        

    # print('\nRecursively iterate 3')
    # repeat_data = repeat_iter(load_data, kwargs=dict(filename=fn, max_row=3), n_times=-1)
    # for _ in range(10):
    #     id, label, data = next(repeat_data)
    #     print('id={id} label={label}'.format(id=id, label=label))
    #     print(data[-1, ])

    # fold1_df = load_dataframe(filename='fold1_extracted.dat')

    if False:
        for obs in load_data_batch(filename=fn2, max_row=5, batchsize=2):
            print('new obs')
            for id, label, data in obs:
                print('id={id} label={label}'.format(id=id, label=label))
