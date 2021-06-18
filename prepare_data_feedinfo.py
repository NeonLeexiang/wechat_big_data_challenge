# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
# 存储数据的根目录
ROOT_PATH = "../data"
#out_path='data'
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
# FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
"""
# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH =ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = "./data/wechat_algo_data1/user_action.csv"
FEED_INFO = "./data/wechat_algo_data1/feed_info.csv"
FEED_EMBEDDINGS ="data/wechat_algo_data1/feed_embeddings.csv"

# adding feature file path
USER_FEATURE = './data/feature_data/user_feature_617.csv'
FEED_FEATURE = './data/feature_data/feed_feature_617.csv'

# 测试集
TEST_FILE = "./data/wechat_algo_data1/test_a.csv"

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'key_1', 'key_2', 'key_3','key_4', 'key_5', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 10, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}


def process_embed(train):
    feed_embed_array = np.zeros((train.shape[0], 512))
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    train = pd.concat((train, temp), axis=1)
    return train

def change_feed_info(feed_info_df):
    key_tag=feed_info_df[['manual_keyword_list', 'manual_tag_list']]
    feed_info_df[['key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5']] = np.nan
    with tqdm(total=len(key_tag['manual_tag_list'])) as pbar:
        for index, line in enumerate(key_tag.values):
            line = list(line)
            key = line[0]
            tag = line[1]
            key = key.split(';') if type(key)!=float else [0]
            tag = tag.split(';') if type(tag)!=float else [0]
            feed_info_df.loc[index, 'key_1'] = int(key[0]) if len(key) >= 5 else 0
            feed_info_df.loc[index, 'key_2'] = int(key[1]) if len(key) >= 5 else 0
            feed_info_df.loc[index, 'key_3'] = int(key[2]) if len(key) >= 5 else 0
            feed_info_df.loc[index, 'key_4'] = int(key[3]) if len(key) >= 5 else 0
            feed_info_df.loc[index, 'key_5'] = int(key[4]) if len(key) >= 5 else 0
            feed_info_df.loc[index, 'tag_1'] = int(tag[0]) if len(tag) >= 5 else 0
            feed_info_df.loc[index, 'tag_2'] = int(tag[1]) if len(tag) >= 5 else 0
            feed_info_df.loc[index, 'tag_3'] = int(tag[2]) if len(tag) >= 5 else 0
            feed_info_df.loc[index, 'tag_4'] = int(tag[3]) if len(tag) >= 5 else 0
            feed_info_df.loc[index, 'tag_5'] = int(tag[4]) if len(tag) >= 5 else 0
            pbar.update(1)
    return feed_info_df


def prepare_data():
    feed_info_df = pd.read_csv(FEED_INFO)
    feed_info_df = change_feed_info(feed_info_df)

    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    # feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    test = pd.read_csv(TEST_FILE)
    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')

    # merge with feature file
    user_feature_617 = pd.read_csv(USER_FEATURE)
    feed_feature_617 = pd.read_csv(FEED_FEATURE)
    train = pd.merge(train, user_feature_617, on='userid', how='left')
    train = pd.merge(train, feed_feature_617, on='feedid', how='left')
    test = pd.merge(test, user_feature_617, on='userid', how='left')
    test = pd.merge(test, feed_feature_617, on='feedid', how='left')

    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    test.to_csv('test_data_tag_key.csv', index=False)
    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=1.0 / ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        #df_all["play"] = np.log(df_all["play"] + 1.0)
        df_all.to_csv(f'train_data_for_{action}_tag_key.csv', index=False)


if __name__ == "__main__":
    prepare_data()
