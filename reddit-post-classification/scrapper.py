import pandas as pd
import numpy as np

p = {"sort_type":['score','score','created_utc'],
     "score":[">100",">100",">100"],
     "sort":["desc","asc","desc"],
    }

def reddit_scrape(subreddit,params=p,before=1559347200,after=1514764800):
    '''
    This function scrapes reddit using the pushshift database to extract posts which follow criteria given by params
    dictionary.

    Before default date - 1 Jun 2019
    After default date - 1 Jan 2018

    We will use this date range to extract 1,500 posts from each subreddit for our training dataset. We will test on
    posts from 1 Jun 2019 onwards. There may be duplicates in this requested 1500 posts, the function will drop duplicates
    before returning

    Params argument must be a dict. Keys must be pushshift api parameters and values must be a list, each of equal length

    This function will return a dataframe
    '''
    dic = {'title':[],
           'subreddit':[],
            'score':[]}
    p = params
    assert len(p[list(p)[0]]) == len(p[list(p)[1]])
    assert len(p[list(p)[1]]) == len(p[list(p)[2]])
    its = len(p[list(p)[2]])
    url = "https://api.pushshift.io/reddit/search/submission/?after={}&before={}&limit=500".format(after,before)
    url +="&subreddit={}".format(subreddit)
    for i in range(its):
        temp_url = url + "&" + list(p)[0] + "=" + p[list(p)[0]][i]
        temp_url = temp_url +  "&" + list(p)[1] + "=" + p[list(p)[1]][i]
        temp_url = temp_url +  "&" + list(p)[2] + "=" + p[list(p)[2]][i]
        h = requests.get(temp_url,headers={'User-agent': "Agent1"}).json()
        for d in h['data']:
            dic['title'].append(d['title'])
            dic['subreddit'].append(d['subreddit'])
            dic['score'].append(d['score'])
    df = pd.DataFrame(dic)
    df=df.drop_duplicates('title')
    return df
