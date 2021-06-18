# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import *
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer

# 存储数据的根目录
ROOT_PATH = "./data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id','key_1','key_2','key_3','key_4','key_5','tag_1','tag_2','tag_3','tag_4','tag_5']
# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 10, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}

FEED_FEATURE_LIST = ['feed_interaction_num', 'feed_read_comment_num',
       'feed_like_num', 'feed_forward_num', 'feed_click_avatar_num',
       'feed_read_comment_rat', 'feed_like_rat', 'feed_forward_rat',
       'feed_click_avatar_rat', 'stay_min', 'stay_max', 'stay_mean',
       'stay_median', 'stay_std', 'play_min', 'play_max', 'play_mean',
       'play_median', 'play_std']

USER_FEATURE_LIST = ['user_interaction_num', 'user_read_comment_num',
       'user_like_num', 'user_forward_num', 'user_click_avatar_num',
       'user_read_comment_rat', 'user_like_rat', 'user_forward_rat',
       'user_click_avatar_rat', 'stay_min', 'stay_max', 'stay_mean',
       'stay_median', 'stay_std', 'play_min', 'play_max', 'play_mean',
       'play_median', 'play_std']

NEW_FEATURE = ['user_interaction_num', 'user_read_comment_num', 'user_like_num',
       'user_forward_num', 'user_click_avatar_num', 'user_read_comment_rat',
       'user_like_rat', 'user_forward_rat', 'user_click_avatar_rat',
       'stay_min_x', 'stay_max_x', 'stay_mean_x', 'stay_median_x',
       'stay_std_x', 'play_min_x', 'play_max_x', 'play_mean_x',
       'play_median_x', 'play_std_x', 'feed_interaction_num',
       'feed_read_comment_num', 'feed_like_num', 'feed_forward_num',
       'feed_click_avatar_num', 'feed_read_comment_rat', 'feed_like_rat',
       'feed_forward_rat', 'feed_click_avatar_rat', 'stay_min_y', 'stay_max_y',
       'stay_mean_y', 'stay_median_y', 'stay_std_y', 'play_min_y',
       'play_max_y', 'play_mean_y', 'play_median_y', 'play_std_y']
FEA_FEED_LIST += NEW_FEATURE


class MyBaseModel(BaseModel):

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):

        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=4, T_mult=2, eta_min=1e-6,
                                                                            last_epoch=-1)

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            index=0
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pred = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        optim.step()
                        if lr_scheduler != None:
                            lr_scheduler.step(epoch + (index + 1) / len(train_loader))
                        index+=1
                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    temp = metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                                except Exception:
                                    temp = 0
                                finally:
                                    train_result[name].append(temp)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return self.history

    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

# class DIN(MyBaseModel):
#     """Instantiates the Deep Interest Network architecture.
#     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
#     :param history_feature_list: list,to indicate  sequence sparse field
#     :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
#     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
#     :param dnn_activation: Activation function to use in deep net
#     :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
#     :param att_activation: Activation function to use in attention net
#     :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
#     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
#     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
#     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
#     :param init_std: float,to use as the initialize std of embedding vector
#     :param seed: integer ,to use as random seed.
#     :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
#     :param device: str, ``"cpu"`` or ``"cuda:0"``
#     :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
#     :return:  A PyTorch model instance.
#     """
#
#     def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
#                  dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
#                  att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
#                  l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
#                  seed=1024, task='binary', device='cpu', gpus=None):
#         super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
#                                   init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
#
#         self.sparse_feature_columns = list(
#             filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
#         self.varlen_sparse_feature_columns = list(
#             filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
#
#         self.history_feature_list = history_feature_list
#
#         self.history_feature_columns = []
#         self.sparse_varlen_feature_columns = []
#         self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
#
#         for fc in self.varlen_sparse_feature_columns:
#             feature_name = fc.name
#             if feature_name in self.history_fc_names:
#                 self.history_feature_columns.append(fc)
#             else:
#                 self.sparse_varlen_feature_columns.append(fc)
#
#         att_emb_dim = self._compute_interest_dim()
#
#         self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
#                                                        embedding_dim=att_emb_dim,
#                                                        att_activation=att_activation,
#                                                        return_score=False,
#                                                        supports_masking=False,
#                                                        weight_normalization=att_weight_normalization)
#
#         self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
#                        hidden_units=dnn_hidden_units,
#                        activation=dnn_activation,
#                        dropout_rate=dnn_dropout,
#                        l2_reg=l2_reg_dnn,
#                        use_bn=dnn_use_bn)
#         self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
#         self.to(device)
#
#
#     def forward(self, X):
#         _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
#
#         # sequence pooling part
#         query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
#                                           return_feat_list=self.history_feature_list, to_list=True)
#         keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
#                                          return_feat_list=self.history_fc_names, to_list=True)
#         dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
#                                               to_list=True)
#
#         sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
#                                                       self.sparse_varlen_feature_columns)
#
#         sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
#                                                       self.sparse_varlen_feature_columns, self.device)
#
#         dnn_input_emb_list += sequence_embed_list
#         deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
#
#         # concatenate
#         query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
#         keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]
#
#         keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
#                                     feat.length_name is not None]
#         keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]
#
#         hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]
#
#         # deep part
#         deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
#         deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)
#
#         dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
#         dnn_output = self.dnn(dnn_input)
#         dnn_logit = self.dnn_linear(dnn_output)
#
#         y_pred = self.out(dnn_logit)
#
#         return y_pred
#
#     def _compute_interest_dim(self):
#         interest_dim = 0
#         for feat in self.sparse_feature_columns:
#             if feat.name in self.history_feature_list:
#                 interest_dim += feat.embedding_dim
#         return interest_dim

class DeepFM(MyBaseModel):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param use_fm: bool,use FM part or not
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(512,256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(
            dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(
                dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, X):

        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(
                sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred


if __name__ == "__main__":
    submit = pd.read_csv('./data/wechat_algo_data1/test_a.csv')[['userid', 'feedid']]
    x=[3,3,2,2]
    y=0
    for action in ACTION_LIST:
        USE_FEAT = ['userid', 'feedid',action] + FEA_FEED_LIST[1:]
        train = pd.read_csv(f'train_data_for_{action}_tag_key.csv')[USE_FEAT]

        train = train.sample(frac=1, random_state=42).reset_index(drop=True)
        print("posi prop:")
        print(sum((train[action]==1)*1)/train.shape[0])
        test = pd.read_csv('test_data_tag_key.csv')[[i for i in USE_FEAT if i != action]]
        target = [action]
        test[target[0]] = 0
        test = test[USE_FEAT]
        data = pd.concat((train, test)).reset_index(drop=True)
        dense_features = ['videoplayseconds','tag_1','tag_2','tag_3','tag_4','tag_5'] + NEW_FEATURE
        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]

        data[sparse_features] = data[sparse_features].fillna(0)
        data[dense_features] = data[dense_features].fillna(0)

        # 1.Label Encoding for sparse features,and do simple Transformation for dense features
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        # 2.count #unique features for each sparse field,and record dense feature field name
        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = DeepFM(linear_feature_columns=linear_feature_columns,dnn_feature_columns=dnn_feature_columns,
                       task='binary',
                       l2_reg_embedding=1e-1, device=device)

        model.compile(optimizer=torch.optim.Adam(model.parameters(),lr=1e-3), loss="binary_crossentropy", metrics=["binary_crossentropy", "auc"])

        history = model.fit(train_model_input, train[target].values, batch_size=512, epochs=x[y], verbose=1,
                            validation_split=0.2)
        pred_ans = model.predict(test_model_input, 128)
        submit[action] = pred_ans
        torch.cuda.empty_cache()
        y+=1
    # 保存提交文件
    submit.to_csv("submit_base__deepfm.csv", index=False)
