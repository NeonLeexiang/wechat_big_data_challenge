"""
    date:       2021/6/14 11:53 上午
    written by: neonleexiang
"""
import os
import numpy as np
import pandas as pd

# pytorch package
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from my_matrix_factorization_by_pytorch import Matrixfactorization


root = './data/wechat_algo_data1'
BATCH_SIZE = 512


def read_test_data():
    # cols = ['userid', 'feedid', 'read_comment', 'like', 'click_avatar', 'forward']
    # drop_columns = ['date_', 'device', 'comment', 'play', 'stay', 'follow', 'favorite']
    drop_columns = ['device']
    test_data = pd.read_csv(os.path.join(root, 'test_a.csv')).drop(columns=drop_columns).astype(int)
    if not os.path.exists('test_a_pred.csv'):
        test_data.to_csv('test_a_pred.csv', index=False)
    # n_user, n_feed = test_data[['userid', 'feedid']].max()
    # return n_user, n_feed, test_data
    return test_data


class TestDataset(Dataset):
    def __init__(self, df):
        self.df_values = df.values

    def __getitem__(self, idx):
        _userid, _feedid = self.df_values[idx]
        return torch.LongTensor([_userid]), torch.LongTensor([_feedid])

    def __len__(self):
        return len(self.df_values)


def main(choose):
    test_data = read_test_data()
    test_a_df = pd.read_csv('test_a_pred.csv')
    pred_list = []

    test_loader = DataLoader(TestDataset(test_data),
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=0)
    print('successfully reading data...')
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device:', device)

    model = torch.load('./model_file/model_'+choose+'.pkl', map_location=torch.device('cpu')).to(device)
    print('adding model successfully...')

    # Set to not-training mode to disable dropout
    model.eval()
    print('begin to predict ->[ {} ]'.format(choose))
    with torch.no_grad():
        for user_pred, feed_pred in test_loader:
            user_pred, feed_pred = user_pred.to(device), feed_pred.to(device)
            pred = model(user_pred, feed_pred)
            pred_list += list(pred)
            # print('----> predicting user:[{}], feed:[{}]'.format(user_pred, feed_pred))

    pred_list = [float(c) + 0.5 for c in pred_list]

    test_a_df[choose] = pred_list
    test_a_df.to_csv('test_a_pred.csv', index=False)
    print('successfully predict ->[ {} ]'.format(choose))


if __name__ == '__main__':
    main('read_comment')
    main('like')
    main('click_avatar')
    main('forward')

    # test_a_df = pd.read_csv('test_a_pred.csv')


    # test_data = read_test_data()
    # test_loader = DataLoader(TestDataset(test_data),
    #                          batch_size=BATCH_SIZE,
    #                          shuffle=False,
    #                          num_workers=0)
    # print(test_loader.dataset.__getitem__(0))
