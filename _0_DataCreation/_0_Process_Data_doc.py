from json import JSONDecoder, JSONDecodeError  # for reading the JSON data files
import re  # for regular expressions
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


def decode_obj(line, pos=0, decoder=JSONDecoder()):
    no_white_space_regex = re.compile(r'[^\s]')
    while True:
        match = no_white_space_regex.search(line, pos)
        if not match:
            return
        pos = match.start()
        try:
            obj, pos = decoder.raw_decode(line, pos)
        except JSONDecodeError as err:
            print('Oops! something went wrong. Error: {}'.format(err))
        yield obj


def one_obs_as_pandas(line, train_data: bool = True, sort: bool = False):
    obj = next(decode_obj(line))  # dict
    id = obj['id']
    data = pd.DataFrame.from_dict(obj['values'])  # pd.DataFrame
    data.set_index(data.index.astype(int), inplace=True)
    if sort:
        data.sort_index(inplace=True)

    if train_data:
        label = obj['classNum']
        return id, label, data
    else:
        return id, data


def json_to_pickle(file: str, out_file: str, train_data: bool = True):
    with open(file, 'r') as f, open(out_file, 'wb') as of:
        i = 0
        for line in f:
            # id, label, data = one_obs_as_pandas(line, sort=True)
            # pprint(data)
            # print('id={}, label={}'.format(id, label))
            # break
            tuple_data = one_obs_as_pandas(line, train_data=train_data, sort=True)
            pickle.dump(tuple_data, of)
            #if i == 5:
            #    break
            if (i % 1000) == 0:
                print(i)
            i += 1


if __name__ == '__main__':
    # from Non_Repo.Paths import Data_Path
    import pickle
    from pprint import pprint

    Data_Path = '/Users/SebastianGPedersen/Dropbox/KU/6. aar/LSDA/Kaggle/Data/'


    fn = Data_Path + 'fold1Training.json'  # obs = 0..76773
    fn2 = Data_Path + 'fold2Training.json'  # obs = 0..92481
    fn3 = Data_Path + 'fold3Training.json'  # obs = 0..27006
    fn4 = Data_Path + 'testSet.json'  # obs = 0..173512

    out_file = fn.replace('fold1Training.json', 'fold1.dat')
    out_file2 = fn.replace('fold1Training.json', 'fold2.dat')
    out_file3 = fn.replace('fold1Training.json', 'fold3.dat')
    out_file4 = fn.replace('fold1Training.json', 'testSet.dat')

    for f, of, train_data in zip([fn, fn2, fn3, fn4],
                                 [out_file, out_file2, out_file3, out_file4],
                                 [True, True, True, False]):
        json_to_pickle(f, of, train_data)

    # with open(out_file, 'rb') as of:
    #     a, b, c = pickle.load(of)
