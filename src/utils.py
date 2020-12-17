import pandas as pd
import numpy as np


def prefilter_items(data, item_features, take_n_popular=5000): #,  margin_slice_rate=0.9

    # Уберем товары с нулевыми продажами
    data = data[(data.sales_value >0) & (data.quantity >0)]

    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity['share'] = popularity['share_unique_users'] / popularity['share_unique_users'].sum() * 100
    top_popular = popularity[popularity['share'] > 0.1].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые популярные товары (их и так купят)
    # popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    # popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    # top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
    # data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users']<=2].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 6 месяцев
    notsold = data['item_id'][(data['week_no'] <= 6 * 4) & (data['sales_value'] == 0)]
    data = data[~data['item_id'].isin(notsold)]

    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 128].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 1]

    # Уберем слишком дорогие товары
    data = data[data['price'] < 15]

    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data


def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""

    popular = data.groupby('item_id')['quantity'].count().reset_index()
    popular.sort_values('quantity', ascending=False, inplace=True)
    popular = popular[popular['item_id'] != 999999]
    recs = popular.head(n).item_id
    return recs.tolist()



def postfilter_items(recommendations, data_train, user, item_features, N=5):
    """Пост-фильтрация товаров

    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """
    # dataframe  товар - цена = средняя цена на товар
    item_price = data_train.groupby('item_id')['price'].mean().reset_index()

    # Уникальность
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]

    # Стоимость каждого рекомендованного товара > 1 доллар
    unique_recommendations = [x for x in unique_recommendations if x \
                              in item_price['item_id'].loc[item_price['price'] > 1].tolist()]
    unique_recommendations = [x for x in unique_recommendations if x != 999999]
    # Разные категории
    categories_used = []
    final_recommendations = []

    CATEGORY_NAME = 'sub_commodity_desc'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]

        if category not in categories_used:
            final_recommendations.append(item)

        unique_recommendations.remove(item)
        categories_used.append(category)

    # 1 дорогой товар > 7 долларов
    rec_7USD = [x for x in item_price['item_id'].loc[item_price['price'] > 7]. \
        tolist() if x in final_recommendations][:1]
    if len(rec_7USD) < 1:
        rec_7USD = [x for x in popularity_recommendation(data_train, 100) if
                    x in item_price['item_id'].loc[item_price['price'] > 7].tolist()][:1]
    else:
        pass

    # 2 новых товара (юзер никогда не покупал)
    rec_new2 = [x for x in final_recommendations if x not in data_train['item_id'].loc[data_train['user_id'] == user].unique() and x not in rec_7USD][:2]
    adds  = list(set(rec_new2 +rec_7USD))

    #дописать что если нет 2 новых

    #все, кроме новых и товара за 7 долларов
    final_recs_1 = [x for x in final_recommendations  if x not in adds]
    final_recs_2 = adds  + final_recs_1

    # Для каждого юзера 5 рекомендаций (иногда модели могут возвращать < 5)

    n_rec = len(final_recs_2 )
    if n_rec < N:
        # Если меньше, то дополняем топом популярных
        final_recs_2  = final_recs_2 .extend(popularity_recommendation(data_train, N))
    else:
        final_recs_2  = final_recs_2 [:N]

    assert len(final_recs_2) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recs_2