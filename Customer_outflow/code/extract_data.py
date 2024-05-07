import pandas as pd
from tqdm import tqdm

action = pd.read_csv('../data/train_action_df.csv')
pay = pd.read_csv('../data/train_pay_df.csv')

#   all clients from 205 till 212 period inclusive
train_pay = pay[(pay['period'] >= 205) & (pay['period'] <= 212)]
train_action = action[action['user_id'].isin(set(train_pay['user_id']))]

def date_in_period(date, period):
    year, month, day = date.split('-')
    year1 = (period - 205) // 4 + 2021
    quarter = (period - 205) % 4 + 1
    if int(year) == year1 and quarter - 1 < int(month) / 3 <= quarter:
        return True
    else:
        return False

def make_DB(type, period_from=205, period_to=212):
    cols = ['user_id', 'action_1', 'action_2',
              'action_3', 'action_4', 'action_5',
              'prev', 'current_tariff', 'prepayment', 'total']
    if type == 'train':
        cols += ['will_stay']

    result_db = pd.DataFrame(columns=cols)
    for p in tqdm(range(period_from, period_to)):
        pay_current = train_pay[train_pay['period'] == p].reset_index(drop=True)
        if type == 'train':
            pay_next = train_pay[train_pay['period'] == p + 1].reset_index(drop=True)

        current_users = set(pay_current['user_id'])

        action_current = action[(action['user_id'].isin(current_users))
                                & (action['action_date'].apply(date_in_period, args=[p]))].reset_index(drop=True)

        df_pivot = pd.pivot_table(
            action_current[['user_id', 'action_type', 'cnt_actions']],
            index='user_id',
            columns="action_type",
            values='cnt_actions',
            aggfunc='sum').reset_index()

        df_pivot.columns = ['user_id', 'action_1', 'action_2', 'action_3', 'action_4', 'action_5']
        df_pivot.fillna(0, inplace=True)
        df_pivot = pd.concat([df_pivot, pd.DataFrame([[id, 0, 0, 0, 0, 0]
                                                      for id in current_users - set(df_pivot['user_id'])],
                                                        columns=df_pivot.columns)]).reset_index(drop=True)

        user_to_period = pd.pivot_table(
                train_pay[(train_pay['period'] < p + 1)
                          & (train_pay['user_id'].isin(current_users))][['user_id', 'period']],
                index='user_id',
                values="period",
                aggfunc=lambda x: len(x.unique())).reset_index()
        user_to_period.columns = ['user_id', 'period']

        df_pivot['prev'] = df_pivot['user_id'].apply(lambda x:
                                                     user_to_period[user_to_period['user_id'] == x]['period'].values[0])
        df_pivot['current_tariff'] = df_pivot['user_id'].apply(lambda x:
                                                   pay_current[pay_current['user_id'] == x].tail(1)['tariff'].values[0])
        df_pivot['prepayment'] = df_pivot['user_id'].apply(lambda x:
                          not date_in_period(pay_current[pay_current['user_id'] == x].tail(1)['pay_date'].values[0], p))

        df_pivot['total'] = df_pivot['action_1'] + df_pivot['action_2'] + df_pivot['action_3']\
                            + df_pivot['action_4'] + df_pivot['action_5']
        if type == 'train':
            df_pivot['will_stay'] = df_pivot['user_id'].apply(lambda x: int(x in set(pay_next['user_id'])))

        result_db = pd.concat([result_db, df_pivot], axis=0)

    return result_db.reset_index(drop=True)

#   year before target period
data_train = make_DB('train', 208, 212)
data_test = make_DB('test', 212, 213)

data_train.to_csv('main_raw_data_train.csv')
data_test.to_csv('main_raw_data_test.csv')
