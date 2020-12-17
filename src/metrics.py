import numpy as np


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, df_price, k=5):
    prices_recommended = []
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    [prices_recommended.append(df_price.loc[df_price['item_id'] == item]['price'].values) for item in
     recommended_list]
    prices_recommended = np.array(prices_recommended[:k])
    # определяем номера позиций, которые купили
    flags = np.isin(recommended_list, bought_list)
    prices = flags @ prices_recommended
    precision = prices.sum() / prices_recommended.sum()
    return float("{0:.4f}".format(precision))

def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if k < len(recommended_list):
        recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_list = np.array(prices_recommended[:k])

    # определяем номера позиций, которые купили
    flags = np.isin(bought_list, recommended_list)
    # определяем позиции цен, по которым купили
    position_price = np.isin(recommended_list, bought_list[flags])
    # выбираем цены по позициям
    prices_chosen = prices_list[position_price]
    # доход от рекомендованных, который купили делим на реальный доход от всех покупок (не только рекомендации)
    recall = np.sum(prices_chosen) / np.sum(prices_bought)
    return recall