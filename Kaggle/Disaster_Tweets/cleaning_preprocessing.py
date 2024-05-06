import numpy as np
import pandas as pd
import re


data_train = pd.read_csv('../data/train.csv')
data_test = pd.read_csv('../data/test.csv')

data_train.fillna(' ', inplace=True)
data_test.fillna(' ', inplace=True)

def split_hashtag(text):
    inds = [0]
    for i in range(len(text) - 1):
        if text[i].isupper() and text[i + 1].islower() and inds[-1] != i:
            inds += [i]
        elif text[i].islower() and text[i + 1].isupper() and inds[-1] != i + 1:
            inds += [i + 1]
    inds += [len(text)]
    return ' '.join([text[inds[i]:inds[i+1]].lower() for i in range(len(inds) - 1)])

def get_info(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    links = re.findall('https?\:\/\/[\w\.]+\/\w+', text)

    hashtags = [split_hashtag(i[1:]) for i in re.findall('\#\w+', text)]

    replys = [i[1:] for i in re.findall('\@\w+', text)]

    return pd.Series(map(lambda x: ' '.join(x) if len(x) else np.nan, [emoticons, links, hashtags, replys]))

def func(word):
    if word == "can't":
        return "can not"
    elif word.endswith("n't"):
        return word[:-3] + ' not'
    else:
        return word

def cleaning(text):
    text = text.lower()
    text = re.sub('https?\:\/\/[\w\.]+\/\w+', ' ', text)#   links

    text = re.sub('\#\w+', ' ', text)#    hashtags

    text = re.sub('\@\w+', ' ', text)#    replys

    text = ' '.join([func(word) for word in text.split()])# paste 'not'

    text = re.sub('[\W]+', ' ', text, flags=re.ASCII).strip()#    trash

    text = ' '.join([word for word in text.split() if len(word) > 2])#  remove if len <= 2

    return text


data_train[['emoticons', 'links', 'hashtags', 'replys']] = data_train['text'].apply(get_info)
data_train['text'] = data_train['text'].apply(cleaning)
data_train['location'] = data_train['location'].apply(lambda x: re.sub('[\W]+', ' ', x.lower(), flags=re.ASCII).strip())

data_test[['emoticons', 'links', 'hashtags', 'replys']] = data_test['text'].apply(get_info)
data_test['text'] = data_test['text'].apply(cleaning)
data_test['location'] = data_test['location'].apply(lambda x: re.sub('[\W]+', ' ', x.lower(), flags=re.ASCII).strip())
# print(data_train.head(100).to_string())

def pivot_staying_rate(data, target_column):
    data.loc[:, data.columns[~data.columns.isin(['id', 'target'])]] = ~data[data.columns[~data.columns.isin(['id', 'target'])]].isna()

    df_pivot = pd.pivot_table(
        data[['id', target_column, 'target']],
        index=[target_column],
        columns=["target"],
        aggfunc='count',
        fill_value=0) \
        .reset_index()

    df_pivot.columns = [target_column, 'spam', 'disaster']
    df_pivot['users'] = df_pivot['disaster'] + df_pivot['spam']
    df_pivot['disaster_rate'] = df_pivot['disaster'] / df_pivot['users'] * 100

    print(df_pivot.to_markdown())
    print()

print(data_train.isna().sum())
# for i in ['keyword', 'location', 'emoticons', 'links', 'hashtags', 'replys']:
# # for i in ['location']:
#     print(i)
#     pivot_staying_rate(data_train.copy(), i)
#     print()

data_train['links'] = (~data_train['links'].isna()).astype('int')
data_train['replys'] = (~data_train['replys'].isna()).astype('int')
data_train['emoticons'] = (~data_train['emoticons'].isna()).astype('int')
data_test['links'] = (~data_test['links'].isna()).astype('int')
data_test['replys'] = (~data_test['replys'].isna()).astype('int')
data_test['emoticons'] = (~data_test['emoticons'].isna()).astype('int')

print(data_train.isna().sum())

data_train.to_csv('train_processed.csv', index=False)
data_test.to_csv('test_processed.csv', index=False)
