import numpy as np
import pandas as pd

def time_features(data):
    #выделим время суток
    data.loc[(data['trans_time'] > 0) & (data['trans_time'] < 600), 'night_time'] = 1
    data.loc[(data['trans_time'] >= 600) & (data['trans_time'] < 1200), 'morning_time'] = 1
    data.loc[(data['trans_time'] >= 1200) & (data['trans_time'] < 1700), 'day_time'] = 1
    data.loc[(data['trans_time'] >= 1700) & (data['trans_time'] < 2400), 'evening_time'] = 1
    data.fillna(0, inplace=True)
    data = data.groupby('user_id')['night_time', 'morning_time', 'day_time', 'evening_time'].mean().reset_index()
    return data

def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def new_item_features(data, item_features, items_emb_df):
    """Новые признаки для продуктов"""
    new_item_features = item_features.merge(data, on='item_id', how='left')

    ##### Добавим имбеддинги
    item_features = item_features.merge(items_emb_df, how='left')

    ##### discount
    mean_disc = new_item_features.groupby('item_id')['coupon_disc'].mean().reset_index().sort_values('coupon_disc')
    item_features = item_features.merge(mean_disc, on='item_id', how='left')

    ###### manufacturer
    rare_manufacturer = item_features.manufacturer.value_counts()[item_features.manufacturer.value_counts() < 20].index
    item_features.loc[item_features.manufacturer.isin(rare_manufacturer), 'manufacturer'] = 999999999
    item_features.manufacturer = item_features.manufacturer.astype('object')

    ##### 1 Количество продаж и среднее количество продаж товара
    item_qnt = new_item_features.groupby(['item_id'])['quantity'].count().reset_index()
    item_qnt.rename(columns={'quantity': 'quantity_of_sales'}, inplace=True)
    item_features = item_features.merge(item_qnt, on='item_id', how='left')

    ## добавим экспоненциальное сглаживание продаж

    #продажи по неделям каждого товара
    d2 = data.groupby(['item_id', 'week_no'])['sales_value'].sum().reset_index()
    r1 = d2.groupby('item_id')['sales_value']
    r1_dict = []
    for item, sales in r1:
        r1_dict.append({'item_id': item, 'sales_value': sales.tolist()})
    r1_df = pd.DataFrame(r1_dict)
    r1_df['rowIndex'] = r1_df.apply(lambda x: x.name, axis=1)

    ## добавим экспоненциальное сглаживание продаж и среднее значение продаж по товару за неделю
    r1_df['exp_sales_0_1'] = r1_df['rowIndex'].apply(lambda x: \
                                                         exponential_smoothing \
                                                             (pd.Series(r1_df['sales_value'].iloc[x]), 0.1)[-1])
    r1_df['exp_sales_0_01'] = r1_df['rowIndex'].apply(lambda x: \
                                                          exponential_smoothing \
                                                              (pd.Series(r1_df['sales_value'].iloc[x]), 0.01)[-1])
    r1_df['exp_sales_0_05'] = r1_df['rowIndex'].apply(lambda x: \
                                                          exponential_smoothing \
                                                              (pd.Series(r1_df['sales_value'].iloc[x]), 0.05)[-1])
    r1_df['sales_mean'] = r1_df['rowIndex'].apply(lambda x: (pd.Series(r1_df['sales_value'].iloc[x]).mean()))
    r1_df = r1_df[['item_id', 'exp_sales_0_1', 'exp_sales_0_01', 'exp_sales_0_05', 'sales_mean']]
    item_features = item_features.merge(r1_df, on='item_id', how='left')

    return item_features


def new_user_features(data, user_features, users_emb_df):
    """Новые признаки для пользователей"""
    data['price'] = data['sales_value'] / data['quantity']

    new_user_features = user_features.merge(data, on='user_id', how='left')

    ##### Добавим имбеддинги
    user_features = user_features.merge(users_emb_df, how='left')

    ##### Обычное время покупки
    time = new_user_features.groupby('user_id')['trans_time'].mean().reset_index()
    time.rename(columns={'trans_time': 'mean_time'}, inplace=True)
    time = time.astype(np.float32)
    user_features = user_features.merge(time, how='left')

    ##### Возраст
    user_features['age'] = user_features['age_desc'].replace(
        {'65+': 70, '45-54': 50, '25-34': 30, '35-44': 40, '19-24': 20, '55-64': 60}
    )
    user_features = user_features.drop('age_desc', axis=1)

    ##### Доход
    user_features['income'] = user_features['income_desc'].replace(
        {'35-49K': 45,
         '50-74K': 70,
         '25-34K': 30,
         '75-99K': 95,
         'Under 15K': 15,
         '100-124K': 120,
         '15-24K': 20,
         '125-149K': 145,
         '150-174K': 170,
         '250K+': 250,
         '175-199K': 195,
         '200-249K': 245}
    )
    user_features = user_features.drop('income_desc', axis=1)

    ##### Дети
    user_features['kids'] = 0
    user_features.loc[(user_features['kid_category_desc'] == '1'), 'kids'] = 1
    user_features.loc[(user_features['kid_category_desc'] == '2'), 'kids'] = 2
    user_features.loc[(user_features['kid_category_desc'] == '3'), 'kids'] = 3
    user_features = user_features.drop('kid_category_desc', axis=1)

    ##### Средний чек, средний чек в неделю

    ##### Средний чек, средний чек в неделю
    basket = new_user_features.groupby(['user_id'])['price'].sum().reset_index()
    baskets_qnt = new_user_features.groupby('user_id')['basket_id'].count().reset_index()
    baskets_qnt.rename(columns={'basket_id': 'baskets_qnt'}, inplace=True)
    average_basket = basket.merge(baskets_qnt)
    average_basket['average_basket'] = average_basket.price / average_basket.baskets_qnt
    average_basket['sum_per_week'] = average_basket.price / new_user_features.week_no.nunique()
    average_basket = average_basket.drop(['price', 'baskets_qnt'], axis=1)
    user_features = user_features.merge(average_basket, how='left')

    ## время
    user_time_data = time_features(data)
    user_features = user_features.merge(user_time_data, on='user_id', how='left')

    return user_features