"""
    date:       2021/6/13 12:25 上午
    written by: neonleexiang
"""
# basic package
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# pytorch package
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# define hyperparameters
BATCH_SIZE = 512
dim = 100
LEARNING_RATE = 0.005
N_EPOCHES = 10
WEIGHT_DECAY = 0.002

root = './data/wechat_algo_data1'


# read_data methods
def read_data():
    # cols = ['userid', 'feedid', 'read_comment', 'like', 'click_avatar', 'forward']
    drop_columns = ['date_', 'device', 'comment', 'play', 'stay', 'follow', 'favorite']
    feed_drop_columns = ['authorid', 'videoplayseconds', 'description', 'ocr', 'asr', 'bgm_song_id', 'bgm_singer_id', 'manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list', 'description_char', 'ocr_char', 'asr_char']

    train_data = pd.read_csv(os.path.join(root, 'user_action.csv')).drop(columns=drop_columns).astype(int)

    feed_info = pd.read_csv(os.path.join(root, 'feed_info.csv')).drop(columns=feed_drop_columns).astype(int)

    n_user = train_data[['userid']].max()
    n_feed = feed_info[['feedid']].max()
    return int(n_user), int(n_feed), train_data[:1000]


# def read_test_data():
#     cols = ['userid', 'feedid', 'read_comment', 'like', 'click_avatar', 'forward']
#     drop_columns = ['date_', 'device', 'comment', 'play', 'stay', 'follow', 'favorite']
#     train_data = pd.read_csv(os.path.join(root, 'user_action.csv')).drop(columns=drop_columns).astype(int)
#     # test_data = pd.read_csv(os.path.join(root, 'u1.test'), sep='\t', names=cols).drop(columns=['timestamp']).astype(int)
#     n_user, n_feed = train_data[['userid', 'feedid']].max()
#     return n_user, n_feed, train_data


# create dataset
class wechat_datasets(Dataset):
    def __init__(self, df):
        self.df_values = df.values

    def __getitem__(self, idx):
        _userid, _feedid, _read_comment, _like, _click_avatar, _forward = self.df_values[idx]

        return torch.LongTensor([_userid]), torch.LongTensor([_feedid]), torch.FloatTensor([_read_comment]), torch.FloatTensor([_like]), torch.FloatTensor([_click_avatar]), torch.FloatTensor([_forward])

    def __len__(self):
        return len(self.df_values)


class test_datasets(Dataset):
    def __init__(self, df):
        self.df_values = df.values

    def __getitem__(self, idx):
        _userid, _feedid, _, _, _, _ = self.df_values[idx]
        return torch.IntTensor([_userid]), torch.IntTensor([_feedid])

    def __len__(self):
        return len(self.df_values)


# built model
class Matrixfactorization(nn.Module):
    def __init__(self, n_user, n_feed, mu=0.5, dim=30):
        super().__init__()

        self.user_latent = nn.Embedding(n_user, dim)
        self.feed_latent = nn.Embedding(n_feed, dim)
        self.user_bias = nn.Embedding(n_user, 1)
        self.feed_bias = nn.Embedding(n_feed, 1)
        self.init_embedding()
        self.mu = mu

    def init_embedding(self):
        self.user_latent.weight.data.normal_(0, 0.02)
        self.feed_latent.weight.data.normal_(0, 0.02)
        self.user_bias.weight.data.normal_(0, 0.5)
        self.feed_bias.weight.data.normal_(0, 0.5)

        return self

    def forward(self, users, feeds):
        # indexes of user and items start at 1
        # python start at 1

        u_latent = self.user_latent(users).squeeze()
        f_latent = self.feed_latent(feeds).squeeze()
        u_bias = self.user_bias(users).squeeze()
        f_bias = self.feed_bias(feeds).squeeze()

        res = u_latent.mm(f_latent.transpose(1, 0)).diag() + u_bias + f_bias + self.mu

        return res


def main(choose):
    n_user, n_feed, train_data = read_data()
    # print(n_user, n_feed)
    # dataset
    train_loader = DataLoader(wechat_datasets(train_data),
                              batch_size=BATCH_SIZE,
                              shuffle=True,

                              num_workers=0)
    # test_loader = DataLoader(MovieLens(test_data),
    #                          batch_size=BATCH_SIZE,
    #                          shuffle=False,
    #                          num_workers=0)

    # mu = train_data['rating'].mean()
    print('successfully reading data...')

    mu = 0.5
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device:', device)

    # model
    model = Matrixfactorization(n_user+1, n_feed+1, mu=mu, dim=dim).to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # begin training
    for epo in tqdm(range(N_EPOCHES)):
        training_loss = 0
        for users, feeds, read_comments, likes, click_avatars, forwards in train_loader:
            users, feeds, read_comments, likes, click_avatars, forwards = users.to(device), feeds.to(device), read_comments.to(device), likes.to(device), click_avatars.to(device), forwards.to(device)
            pre_results = model(users, feeds)
            if choose=='read_comment':
                choose_param = read_comments
            elif choose == 'like':
                choose_param = likes
            elif choose == 'click_avatar':
                choose_param = click_avatars
            elif choose == 'forward':
                choose_param = forwards
            else:
                raise Exception('Error Choose')
            loss = criterion(pre_results, choose_param.squeeze())
            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        with torch.no_grad():

            training_loss = 0
            for users, feeds, read_comments, likes, click_avatars, forwards in train_loader:
                users, feeds, read_comments, likes, click_avatars, forwards = users.to(device), feeds.to(
                    device), read_comments.to(device), likes.to(device), click_avatars.to(device), forwards.to(device)
                pre_results = model(users, feeds)
                if choose == 'read_comment':
                    choose_param = read_comments
                elif choose == 'like':
                    choose_param = likes
                elif choose == 'click_avatar':
                    choose_param = click_avatars
                elif choose == 'forward':
                    choose_param = forwards
                else:
                    raise Exception('Error Choose')
                loss = criterion(pre_results, choose_param.squeeze())
                training_loss += loss.item()

            # testing_loss = 0
            # for users, items, ratings in test_loader:
            #     users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            #     pre_ratings = model(users, items)
            #     loss = criterion(pre_ratings, ratings.squeeze())
            #     testing_loss += loss.item()

        training_rmse = np.sqrt(training_loss / train_loader.dataset.__len__())
        # testing_rmse = np.sqrt(testing_loss / test_loader.dataset.__len__())

        print('Epoch: {0:2d} / {1}, Traning RMSE: {2:.4f}'.format(epo + 1, N_EPOCHES, training_rmse))

    # save model
    if not os.path.exists('./model_file/'):
        os.mkdir('./model_file/')
    torch.save(model, './model_file/model_'+choose+'.pkl')


    # Set to not-training mode to disable dropout
    # model.eval()
    # _, _, test_data = read_test_data()
    # test_loader = DataLoader(test_datasets(test_data),
    #                           batch_size=BATCH_SIZE,
    #                           shuffle=False,
    #                           num_workers=0)
    #
    # with torch.no_grad():
    #     for user_pred, feed_pred in test_loader:
    #         user_pred, feed_pred = user_pred.to(device), feed_pred.to(device)
    #         pred = model(user_pred, feed_pred)
    #         print(pred)


if __name__ == '__main__':
    main('read_comment')
    main('like')
    main('click_avatar')
    main('forward')
