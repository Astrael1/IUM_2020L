# /usr/bin/python3
import pandas as pd
import numpy as np
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import random


def categoryListIntoSeries(listSeries, name):
    return listSeries.apply(lambda x: name in x)


def getListOfCategories(product_data):
    a = set()
    for categories in product_data.category_path:
        for item in categories:
            a.add(item)
    return sorted(list(a))


def preprocess_all(product_data, session_data, users_data):
    # find products with price lower or equal 0 and greater than 1000000 and those which have '#','&' or ';' in name
    incorrect_products = product_data[
        (product_data["price"] <= 0) |
        (product_data["price"] > 1000000) |
        (product_data['product_name'].str.contains('#|;|&', regex=True))].index

    # find values of their 'product_id' column

    # drop incorrect products
    product_data.drop(incorrect_products, inplace=True)
    # change strings with categories into list of categories
    product_data["category_path"] = product_data["category_path"].apply(lambda x: x.split(';'))

    # drop rows with null value of columns 'user_id' or 'product_id'
    session_data.dropna(subset=['user_id', 'product_id'], inplace=True)
    session_data.drop(columns=['purchase_id'], inplace=True)

    # replace floats with ints
    session_data.user_id = session_data.user_id.astype(int)
    session_data.product_id = session_data.product_id.astype(int)
    # drop rows with products that are incorrect
    incorrect_sessions = session_data[session_data["product_id"].isin(incorrect_products)].index
    session_data.drop(incorrect_sessions, inplace=True)

    users_data.drop(columns=['name', 'street'], inplace=True)
    users_data.set_index('user_id', inplace=True)
    session_ids = session_data.session_id.unique()
    random.shuffle(session_ids)
    delimiter = int(len(session_ids) * 0.7)
    train_ids = session_ids[: delimiter]
    test_ids = session_ids[delimiter:]
    sessions_train = session_data[session_data.session_id.isin(train_ids)]
    sessions_test = session_data[session_data.session_id.isin(test_ids)]
    return product_data, sessions_train, sessions_test, users_data


def fit(product_data, session_data, users_data):
    category_weights = {
        'Gry i konsole': 1 / 3,
        'Gry komputerowe': 2 / 3,
        'Gry na konsole': 1 / 3,
        'Gry PlayStation3': 1 / 3,
        'Gry Xbox 360': 1 / 3,
        'Komputery': 1 / 3,
        'Drukarki i skanery': 1 / 3,
        'Biurowe urządzenia wielofunkcyjne': 1 / 3,
        'Monitory': 1 / 3,
        'Monitory LCD': 1 / 3,
        'Tablety i akcesoria': 1 / 3,
        'Tablety': 1 / 3,
        'Sprzęt RTV': 1 / 3,
        'Audio': 1 / 3,
        'Słuchawki': 1 / 3,
        'Przenośne audio i video': 1 / 3,
        'Odtwarzacze mp3 i mp4': 1 / 3,
        'Video': 1 / 3,
        'Odtwarzacze DVD': 1 / 3,
        'Telewizory i akcesoria': 1 / 6,
        'Anteny RTV': 1 / 6,
        'Okulary 3D': 1 / 6,
        'Telefony i akcesoria': 1 / 3,
        'Akcesoria telefoniczne': 1 / 3,
        'Zestawy głośnomówiące': 1 / 3,
        'Zestawy słuchawkowe': 1 / 3,
        'Telefony komórkowe': 2 / 3,
        'Telefony stacjonarne': 2 / 3
    }

    all_categories = getListOfCategories(product_data)
    category_weight_list = [category_weights[all_categories[i]] for i in range(len(all_categories))]

    def getProductHotness(index, products, all_categories, category_weights):
        product_evaluated = products.loc[index]
        interesting_categories = [category for category in all_categories if product_evaluated[category]]
        score = 0
        for category in interesting_categories:
            matching_products = products.loc[(products.index != index) & (products[category] == True)]
            if not matching_products.empty:
                score += (matching_products.price - product_evaluated.price).mean() * category_weights[category]
        return score

    def getProductFrequency(products, sessions):
        data = {}
        for product_id in products.index:
            isBought = sessions.loc[sessions.product_id == product_id, 'event_type'] == 'BUY_PRODUCT'
            frequency = isBought.sum() * 2 / len(isBought) if len(isBought) > 0 else 0
            data[product_id] = frequency
        return pd.DataFrame.from_dict(data, orient='index', columns=['frequency'])

    def preprocess_products(products, sessions, all_categories, category_weights):
        for category in all_categories:
            newSeries = categoryListIntoSeries(products.category_path, category)
            products[category] = newSeries
        grp = products.groupby(all_categories)
        for line in sorted([group['category_path'].iloc[0] for name, group in grp]):
            print(line)

        products['hotness'] = np.nan
        getProductHotness(1001, products, all_categories, category_weights)
        for index in products.index:
            products.loc[index, 'hotness'] = getProductHotness(index, products, all_categories, category_weights)

        frequencyFrame = getProductFrequency(products, sessions)
        products['frequency'] = frequencyFrame['frequency']

    preprocess_products(product_data, session_data, all_categories, category_weights)

    def getTopCategories(howMany, all_categories, sessions, products):
        data = {}
        for category in all_categories:
            category_products = products.loc[products[category]].index
            category_transactions = sessions.loc[
                                        sessions.product_id.isin(category_products), 'event_type'] == 'BUY_PRODUCT'
            score = category_transactions.mean()
            data[category] = score
        scoreFrame = pd.DataFrame.from_dict(data, orient='index', columns=['score'])
        return scoreFrame.sort_values(by='score').index[:howMany].tolist()

    detailed_categories = [
        'Gry Xbox 360',
        'Biurowe urządzenia wielofunkcyjne',
        'Monitory LCD',
        'Tablety',
        'Słuchawki',
        'Odtwarzacze mp3 i mp4',
        'Odtwarzacze DVD',
        'Anteny RTV',
        'Okulary 3D',
        'Zestawy głośnomówiące',
        'Zestawy słuchawkowe',
        'Telefony komórkowe',
        'Telefony stacjonarne']
    topCategories = getTopCategories(3, detailed_categories, session_data, product_data)

    def getTopProductSeries(products, topCategories):
        return products[topCategories].any(axis=1)

    # add information if product is of top category
    topProductSeries = getTopProductSeries(product_data, topCategories)
    product_data['is_top'] = topProductSeries

    def preprocess_users(users, sessions, products, all_categories, category_weights):
        user_ids = users.index
        data = []
        for user_id in user_ids:
            user_purchases_ids = sessions.loc[
                (sessions.user_id == user_id) & (sessions.event_type == 'BUY_PRODUCT'), 'product_id']
            user_products = products.loc[user_purchases_ids]
            category_sums = user_products[all_categories].sum()
            category_total = category_sums.sum()
            category_sums /= category_total
            calculation_result = category_sums.to_dict()
            calculation_result['user_id'] = user_id
            data.append(calculation_result)
        category_frame = pd.DataFrame(data)
        category_frame.fillna(0, inplace=True)
        category_frame.set_index('user_id', inplace=True)
        users = users.join(category_frame)
        return users

    users_data = preprocess_users(users_data, session_data, product_data, all_categories, category_weights)
    return product_data, session_data, users_data, all_categories, category_weights


def make_training_set(product_data, session_data, users_data, all_categories, category_weights):
    groupped_sessions = session_data.groupby('session_id')

    def getSessionAttributes(session, products, users, all_categories, category_weights):
        s_id = session['session_id'].unique()[0]
        user_id = session['user_id'].unique()[0]

        timestamps = session.iloc[[0, -1]].timestamp.values
        duration = (timestamps[1] - timestamps[0]) / np.timedelta64(1, 's')

        event = session.iloc[-1].event_type
        isSuccessfull = True if event == 'BUY_PRODUCT' else False

        discount = session['offered_discount'].values

        product_ids = group['product_id']
        viewed_products_data = products.loc[product_ids].drop(columns=all_categories).drop(
            columns=['product_name', 'category_path'])
        viewed_products_data.insert(1, 'offered_discount', discount, True)
        viewed_products_data.insert(2, 'discount_value',
                                    viewed_products_data.price * viewed_products_data.offered_discount, True)
        mean_values = viewed_products_data.mean()

        is_top = viewed_products_data.is_top.any()
        avg_price = mean_values.price
        avg_discount_percent = mean_values.offered_discount
        avg_discount_value = mean_values.discount_value
        avg_hotness = mean_values.hotness

        sum_values = viewed_products_data.sum()
        total_discount_percent = sum_values.offered_discount
        total_discount_value = sum_values.discount_value
        total_hotness = sum_values.hotness

        # count customer 'liking'
        products_category_total = products.loc[product_ids, all_categories].sum()
        user_preferences = users.loc[user_id, all_categories]
        data = {}
        for category in all_categories:
            score = products_category_total[category] * user_preferences[category] * category_weights[category]
            data[category] = score
        totalScore = pd.DataFrame.from_dict(data, orient='index', columns=['score']).sum().values[0]

        result = {
            'id': s_id,
            'duration': duration,
            'avg_price': avg_price,
            'avg_discount_percent': avg_discount_percent,
            'total_discount_percent': total_discount_percent,
            'avg_discount_value': avg_discount_value,
            'total_discount_value': total_discount_value,
            'avg_hotness': avg_hotness,
            'total_hotness': total_hotness,
            'is_top': is_top,
            'liking_score': totalScore,
            'success': isSuccessfull
        }
        return result

    data = []
    i = 100
    for name, group in groupped_sessions:
        data.append(getSessionAttributes(group, product_data, users_data, all_categories, category_weights))
        # i -= 1
        # if(i == 0):
        #     break

    sessiondf = pd.DataFrame(data)
    sessiondf.set_index('id', inplace=True)

    def simplifyColumn(df, column):
        df.loc[df[column] > 0, column] = df.loc[df[column] > 0, column].apply(np.log10).apply(np.floor)
        df.loc[df[column] < 0, column] = df.loc[df[column] < 0, column].apply(lambda x: -1 * x).apply(np.log10).apply(
            lambda x: -1 * x).apply(np.floor)

    sessiondf_p = sessiondf.copy()
    sessiondf_p.liking_score *= 100
    for column in ['duration', 'avg_price', 'avg_discount_percent', 'total_discount_percent', 'avg_discount_value',
                   'total_discount_value', 'total_hotness', 'avg_hotness', 'liking_score']:
        simplifyColumn(sessiondf_p, column)

    grouped_p = sessiondf_p.groupby(
        ['duration', 'avg_price', 'avg_discount_percent', 'total_discount_percent', 'avg_discount_value',
         'total_discount_value', 'avg_hotness', 'total_hotness', 'liking_score', 'is_top'])
    sessiondf['probability'] = np.nan

    for name, group in grouped_p:
        result = group['success'].sum() / len(group)
        sessiondf.loc[group.index, 'probability'] = result
        print(name, result)
    print("groups: ", len(grouped_p))
    sessiondf.drop(inplace=True, columns=['success'])
    return sessiondf


def visualise(sessiondf):
    corr = sessiondf.corr()
    print(corr.probability)

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    fig = plt.figure(figsize=(20, 20))
    sns.heatmap(corr, mask=mask, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


if __name__ == '__main__':
    product_data = pd.read_json('../resources/products.jsonl', lines=True)
    product_data.set_index('product_id', inplace=True)
    session_data = pd.read_json('../resources/sessions.jsonl', lines=True)
    users_data = pd.read_json('../resources/users.jsonl', lines=True)

    product_data, sessions_train, sessions_test, users_data = preprocess_all(product_data, session_data, users_data)
    product_data, sessions_train, users_data, all_categories, category_weights = fit(
        product_data,
        sessions_train,
        users_data)

    sessiondf_train = make_training_set(product_data, sessions_train, users_data, all_categories, category_weights)

    sessiondf_test = make_training_set(product_data, sessions_test, users_data, all_categories, category_weights)

    visualise(sessiondf_train)

    # sessiondf.to_csv(path_or_buf='session_preprocessed.csv')
    sessiondf_train.to_csv(path_or_buf='session_train.csv')
    sessiondf_test.to_csv(path_or_buf='session_test.csv')
    users_data.to_csv(path_or_buf='users_preprocessed.csv')
    product_data.to_csv(path_or_buf='products_preprocessed.csv')
