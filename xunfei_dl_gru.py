import warnings
from datetime import timedelta, datetime

import gc
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Dense, Dropout, concatenate, Reshape
from keras.layers import Lambda, GaussianDropout, CuDNNGRU, BatchNormalization, PReLU
from keras.models import Model
from keras.optimizers import Adam

warnings.filterwarnings('ignore')

base_cols = ['ip']
media_cols = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype']
time_cols = ['hour']
location_cols = ['city']
device_cols = ['adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5', 'dvctype', 'model', 'make', 'ntt', 'carrier',
               'osv', 'orientation', 'ppi', 'screen_area', 'creative_dpi']
total_cate = [base_cols, media_cols, time_cols, location_cols, device_cols]

data_dir = '/home/galen/workspace/competition/data/'
print('read data')
df_test = pd.read_csv(data_dir + 'round1_iflyad_anticheat_testdata_feature.txt', sep='\t')
df_train = pd.read_csv(data_dir + 'round1_iflyad_anticheat_traindata.txt', sep='\t')
df_uni = pd.concat([df_train, df_test], ignore_index=True)
df_uni['label'] = df_uni['label'].fillna(-1).astype(int)

# 数据预处理
print('prework')
# 处理ip。ip 为空时，使用 reqrealip。
df_uni.ip.fillna(df_uni.reqrealip, inplace=True)
# 屏幕尺寸 合并成宽和高
df_uni['screen_area'] = (df_uni['w'] * df_uni['h']).astype('category')
df_uni['creative_dpi'] = df_uni['w'].astype(str) + "_" + df_uni['h'].astype(str)
# orientation 出现异常值 90度和2 归为 0
df_uni.orientation[(df_uni.orientation == 90) | (df_uni.orientation == 2)] = 0
# carrier  -1 就是0
df_uni.carrier[df_uni.carrier == -1] = 0
# ntt 网络类型。0 未知 -> 0 , 1 2 宽带 1 ,  4,5,6 移动网络 -> 2
df_uni.ntt[(df_uni.ntt <= 0) | (df_uni.ntt > 6)] = 0
df_uni.ntt[(df_uni.ntt <= 2) | (df_uni.ntt >= 1)] = 1
df_uni.ntt[(df_uni.ntt <= 6) | (df_uni.ntt >= 4)] = 2
# 运营商 carrier
df_uni.ntt[(df_uni.carrier <= 0) | (df_uni.carrier > 46003)] = 0


# make
def make_fix(x):
    """
    iphone,iPhone,Apple,APPLE>--apple
    redmi>--xiaomi
    honor>--huawei
    Best sony,Best-sony,Best_sony,BESTSONY>--best_sony
    :param x:
    :return:
    """
    x = x.lower()
    if 'iphone' in x or 'apple' in x:
        return 'apple'
    if '华为' in x or 'huawei' in x or "荣耀" in x:
        return 'huawei'
    if "魅族" in x:
        return 'meizu'
    if "金立" in x:
        return 'gionee'
    if "三星" in x:
        return 'samsung'
    if 'xiaomi' in x or 'redmi' in x:
        return 'xiaomi'
    if 'oppo' in x:
        return 'oppo'
    return x


df_uni['make'] = df_uni['make'].astype('str').apply(lambda x: x.lower())
df_uni['make'] = df_uni['make'].apply(make_fix)

print('feature time...')
# 处理时间
df_uni['datetime'] = pd.to_datetime(df_uni['nginxtime'] / 1000, unit='s') + timedelta(hours=8)
df_uni['hour'] = df_uni['datetime'].dt.hour
# 将天数归零成有序数列。[0,1,2,3,4,5,6]
df_uni['day'] = df_uni['datetime'].dt.day - df_uni['datetime'].dt.day.min()


def unique_count(index_col, feature, df_data):
    if isinstance(index_col, list):
        name = "{0}_{1}_nq".format('_'.join(index_col), feature)
    else:
        name = "{0}_{1}_nq".format(index_col, feature)
    print(name)
    gp1 = df_data.groupby(index_col)[feature].nunique().reset_index().rename(
        columns={feature: name})
    df_data = pd.merge(df_data, gp1, how='left', on=[index_col])
    return df_data.fillna(0)


# 设备下的媒体数  model_mediashowid_nq model_city_nq
df_uni = unique_count('model', 'mediashowid', df_uni)
df_uni = unique_count('model', 'city', df_uni)
# 设备
df_uni = unique_count('adidmd5', 'model', df_uni)
df_uni = unique_count('imeimd5', 'model', df_uni)
df_uni = unique_count('macmd5', 'model', df_uni)
df_uni = unique_count('openudidmd5', 'model', df_uni)
df_uni = unique_count('ip', 'model', df_uni)
df_uni = unique_count('reqrealip', 'model', df_uni)

# 屏幕密度
df_uni = unique_count('adidmd5', 'ppi', df_uni)
df_uni = unique_count('imeimd5', 'ppi', df_uni)
df_uni = unique_count('macmd5', 'ppi', df_uni)
df_uni = unique_count('openudidmd5', 'ppi', df_uni)
df_uni = unique_count('ip', 'ppi', df_uni)
df_uni = unique_count('reqrealip', 'ppi', df_uni)

# 网络类型
df_uni = unique_count('adidmd5', 'dvctype', df_uni)
df_uni = unique_count('imeimd5', 'dvctype', df_uni)
df_uni = unique_count('macmd5', 'dvctype', df_uni)
df_uni = unique_count('openudidmd5', 'dvctype', df_uni)
df_uni = unique_count('ip', 'dvctype', df_uni)
df_uni = unique_count('reqrealip', 'dvctype', df_uni)

# 地理位置
df_uni = unique_count('ip', 'city', df_uni)
df_uni = unique_count('reqrealip', 'city', df_uni)

# 用户下的ip数
df_uni = unique_count('adidmd5', 'ip', df_uni)
df_uni = unique_count('imeimd5', 'ip', df_uni)
df_uni = unique_count('macmd5', 'ip', df_uni)
df_uni = unique_count('openudidmd5', 'ip', df_uni)

# 统计数据
value_counts_col = [
    # 'adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5',
    'make', 'pkgname', 'adunitshowid', 'mediashowid', 'ip', 'city', 'model', 'hour',
    'screen_area', 'creative_dpi', 'h', 'w',
    'dvctype',
]


def gen_value_counts(data, col):
    """
    # 统计每个种类的个数。
    :param data:
    :param col:
    :return:
    """
    print('value counts', col)
    df_tmp = pd.DataFrame(data[col].value_counts().reset_index())
    df_tmp.columns = [col, 'tmp']
    r = pd.merge(data, df_tmp, how='left', on=col)['tmp']
    return r.fillna(0)


# 统计值
counts_col_name = []
for col_values in value_counts_col:
    new_name = 'vc_' + col_values
    df_uni[new_name] = gen_value_counts(df_uni, col_values)
    counts_col_name.append(new_name)

# ip
gp = df_uni[['ip', 'mediashowid', 'adunitshowid']].groupby(by=['ip', 'mediashowid'])[
    ['adunitshowid']].count().reset_index().rename(index=str, columns={'adunitshowid': 'ip_media_count_ad'})
df_uni = df_uni.merge(gp, on=['ip', 'mediashowid', ], how='left')
del gp
gc.collect()

gp = df_uni[['ip', 'mediashowid', 'dvctype', 'hour']].groupby(by=['ip', 'mediashowid', 'dvctype'])[
    ['hour']].var().reset_index().rename(
    index=str, columns={'hour': 'ip_media_dvctype_var_hour'})
df_uni = df_uni.merge(gp, on=['ip', 'mediashowid', 'dvctype'], how='left')
del gp
gc.collect()

gp = df_uni[['ip', 'mediashowid', 'dvctype', 'hour']].groupby(by=['ip', 'mediashowid', 'dvctype'])[
    ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_media_dvctype_mean_hour'})
df_uni = df_uni.merge(gp, on=['ip', 'mediashowid', 'dvctype'], how='left')
del gp

# make
gp = df_uni[['make', 'mediashowid', 'adunitshowid']].groupby(by=['make', 'mediashowid'])[
    ['adunitshowid']].count().reset_index().rename(index=str, columns={'adunitshowid': 'make_media_count_ad'})
df_uni = df_uni.merge(gp, on=['make', 'mediashowid', ], how='left')
del gp
gc.collect()

gp = df_uni[['make', 'mediashowid', 'dvctype', 'hour']].groupby(by=['make', 'mediashowid', 'dvctype'])[
    ['hour']].var().reset_index().rename(
    index=str, columns={'hour': 'make_media_dvctype_var_hour'})
df_uni = df_uni.merge(gp, on=['make', 'mediashowid', 'dvctype'], how='left')
del gp
gc.collect()

gp = df_uni[['make', 'mediashowid', 'dvctype', 'hour']].groupby(by=['make', 'mediashowid', 'dvctype'])[
    ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'make_media_dvctype_mean_hour'})
df_uni = df_uni.merge(gp, on=['make', 'mediashowid', 'dvctype'], how='left')
del gp

# model
gp = df_uni[['model', 'mediashowid', 'adunitshowid']].groupby(by=['model', 'mediashowid'])[
    ['adunitshowid']].count().reset_index().rename(index=str, columns={'adunitshowid': 'model_media_count_ad'})
df_uni = df_uni.merge(gp, on=['model', 'mediashowid', ], how='left')
del gp
gc.collect()

gp = df_uni[['model', 'mediashowid', 'dvctype', 'hour']].groupby(by=['model', 'mediashowid', 'dvctype'])[
    ['hour']].var().reset_index().rename(
    index=str, columns={'hour': 'model_media_dvctype_var_hour'})
df_uni = df_uni.merge(gp, on=['model', 'mediashowid', 'dvctype'], how='left')
del gp
gc.collect()

gp = df_uni[['model', 'mediashowid', 'dvctype', 'hour']].groupby(by=['model', 'mediashowid', 'dvctype'])[
    ['hour']].mean().reset_index().rename(index=str, columns={'hour': 'model_media_dvctype_mean_hour'})
df_uni = df_uni.merge(gp, on=['model', 'mediashowid', 'dvctype'], how='left')
del gp

# city dvctype
gp = df_uni[['city', 'dvctype']].groupby(by=['city'])[
    ['dvctype']].count().reset_index().rename(index=str, columns={'dvctype': 'city_count_dvctype'})
df_uni = df_uni.merge(gp, on=['city'], how='left')
del gp
gc.collect()

# 'dvctype', 'orientation', 'city'
gp = df_uni[['dvctype', 'orientation', 'city']].groupby(by=['dvctype', 'orientation'])[
    ['city']].count().reset_index().rename(index=str, columns={'city': 'dvctype_orientation_count_city'})
df_uni = df_uni.merge(gp, on=['dvctype', 'orientation'], how='left')
del gp
gc.collect()

# 'dvctype', 'ppi', 'city'
gp = df_uni[['dvctype', 'ppi', 'city']].groupby(by=['dvctype', 'ppi'])[
    ['city']].count().reset_index().rename(index=str, columns={'city': 'dvctype_ppi_count_city'})
df_uni = df_uni.merge(gp, on=['dvctype', 'ppi'], how='left')
del gp
gc.collect()

print("merging success...")
# 将种类编码成数字
print('post process')
cat_cols = [
    'model', 'make', 'ppi', 'screen_area', 'creative_dpi',
    'pkgname', 'ver', 'osv', 'city',
    'adidmd5', 'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5',
    'adunitshowid', 'mediashowid',
    'apptype', 'dvctype', 'ntt', 'carrier', 'orientation',
    'hour', 'reqrealip', 'ip', 'h', 'w', 'lan',
]
print(set(df_uni.columns) - (set(cat_cols) | set(counts_col_name)))
for col_values in cat_cols:
    # 将种类进行  映射成唯一编码 {"A":1"}    .unique() 获得唯一值。
    df_uni[col_values] = df_uni[col_values].map(
        dict(zip(df_uni[col_values].unique(), range(0, df_uni[col_values].nunique()))))

# print('model', df_uni['model'].max())
# 数据集索引。最后一天数据用于预测，不提供“是否作弊”标识，其余日期的数据作为训练数据。
all_train_index = (df_uni['day'] <= 6).values
test_index = (df_uni['day'] == 7).values
train_label = df_uni['label']

train_df = df_uni.iloc[all_train_index, :]
y_train = train_label.iloc[all_train_index].values

test_df = df_uni.iloc[test_index, :]


def get_keras_data(dataset, cate_list, num_list):
    X = {
        'category_inp': dataset[cate_list].values,
        'continous_inp': dataset[num_list].values,
    }
    return X


category = [
    # 'adidmd5', 'idfamd5', 'imeimd5', 'macmd5', 'openudidmd5', 'ip', 'reqrealip',
    # 'idfamd5',
    'adunitshowid', 'apptype', 'carrier', 'city', 'dvctype', 'make', 'model', 'mediashowid', 'ntt',
    'orientation', 'osv', 'pkgname', 'ppi', 'hour',
    'screen_area', 'creative_dpi', 'ver', 'h', 'w', 'lan',
]

numerical = [
    'ip_media_count_ad', 'ip_media_dvctype_var_hour', 'ip_media_dvctype_mean_hour',
    'make_media_count_ad', 'make_media_dvctype_var_hour', 'make_media_dvctype_mean_hour',
    'model_media_count_ad', 'model_media_dvctype_var_hour', 'model_media_dvctype_mean_hour',
    'city_count_dvctype', 'dvctype_orientation_count_city', 'dvctype_ppi_count_city',

    'model_mediashowid_nq',
    'model_city_nq',
    # model
    'adidmd5_model_nq',
    'ip_model_nq',
    'imeimd5_model_nq',
    'macmd5_model_nq',
    'openudidmd5_model_nq',
    'reqrealip_model_nq',

    # ppi
    'adidmd5_ppi_nq',
    'ip_ppi_nq',
    'imeimd5_ppi_nq',
    'macmd5_ppi_nq',
    'openudidmd5_ppi_nq',
    'reqrealip_ppi_nq',

    # dvctype
    'adidmd5_dvctype_nq',
    'ip_dvctype_nq',
    'imeimd5_dvctype_nq',
    'macmd5_dvctype_nq',
    'openudidmd5_dvctype_nq',
    'reqrealip_dvctype_nq',

    'ip_city_nq',
    'reqrealip_city_nq',

    'adidmd5_ip_nq',
    'imeimd5_ip_nq',
    'macmd5_ip_nq',
    'openudidmd5_ip_nq',
]


def gru_model():
    emb_n = 64
    category_num = {
        # 'adidmd5': (780369, emb_n),
        # 'idfamd5': (360, emb_n),
        # 'imeimd5': (1021836, emb_n),
        # 'macmd5': (329184, emb_n),
        # 'openudidmd5': (85051, emb_n),
        # 'ip': (813719, emb_n),
        # 'reqrealip': (9748, emb_n),
        'adunitshowid': (800, emb_n),
        'apptype': (91, emb_n),
        'carrier': (4, emb_n),
        'city': (331, emb_n),
        'dvctype': (3, emb_n),
        'model': (5923, emb_n),  # 7957 7958  5922
        'make': (1704, emb_n),
        'mediashowid': (313, emb_n),
        'ntt': (7, emb_n),
        'orientation': (2, emb_n),
        'osv': (185, emb_n),
        'pkgname': (2368, emb_n),
        'ppi': (119, emb_n),
        'ver': (3268, emb_n),
        'screen_area': (1396, emb_n),
        'creative_dpi': (1763, emb_n),
        'hour': (24, emb_n),
        'lan': (33, emb_n),
        'h': (985, emb_n),
        'w': (449, emb_n),

    }
    # 类别型变量输入
    category_inp = Input(shape=(len(category),), name='category_inp')
    cat_embeds = []
    for idx, col in enumerate(category):
        x = Lambda(lambda x: x[:, idx, None])(category_inp)
        x = Embedding(category_num[col][0], category_num[col][1], input_length=1)(x)
        cat_embeds.append(x)
    embeds = concatenate(cat_embeds, axis=2)
    embeds = GaussianDropout(0.5)(embeds)
    # 数值型变量输入
    numerical_inp = Input(shape=(len(numerical),), name='continous_inp')
    print('numerical', len(numerical) // 8 * 8 + 8)
    x2 = Dense(len(numerical) // 8 + 8, activation='relu', kernel_initializer='random_uniform',
               bias_initializer='zeros')(
        numerical_inp)
    x2 = Dropout(0.5)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Reshape([1, int(x2.shape[1])])(x2)
    x = concatenate([embeds, x2], axis=2)
    # 主干网络
    x = CuDNNGRU(128)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    x = Dense(64, activation='relu', kernel_initializer='random_uniform')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    x = Dense(32, activation='relu', kernel_initializer='random_uniform')(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    out_p = Dense(1, activation='sigmoid')(x)
    return Model(inputs=[category_inp, numerical_inp], outputs=out_p)


model = gru_model()
# model.summary()

batch_size = 1024  # 20000 512
epochs = 20

steps = int(len(train_df) / batch_size) * epochs
exp_decay = lambda init, fin, steps: (init / fin) ** (1 / (steps - 1)) - 1
lr_init, lr_fin = 0.001, 0.0001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy', optimizer=optimizer_adam, metrics=['accuracy'])

train_df = get_keras_data(train_df, category, numerical)

early_stopping = EarlyStopping(monitor='va', patience=3)
model.fit(train_df, y_train, callbacks=[early_stopping], validation_split=0.2, batch_size=batch_size, epochs=epochs,
          shuffle=True, verbose=1)

test_df = get_keras_data(test_df, category, numerical)

print("predicting....")
test_y = model.predict(test_df, batch_size=batch_size)

test_list = test_y.flatten().tolist()
result = []
for d in test_list:
    if d > 0.5:
        result.append(1)
    else:
        result.append(0)

df_sub = pd.concat([df_test['sid'], pd.Series(result)], axis=1)
df_sub.columns = ['sid', 'label']
save_path = 'submit-{}.csv'.format(datetime.now().strftime('%m%d_%H%M%S'))
print(save_path)
df_sub.to_csv(save_path, sep=',', index=False)
