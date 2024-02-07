import pickle
import json
import numpy as np
# import pandas as pd
import os
import openai
import tiktoken
import datetime
import logging

# logging config
def Beijing_TimeZone_Converter(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp
# logging.Formatter.converter = time.gmtime
logging.Formatter.converter = Beijing_TimeZone_Converter

def save_pickle(obj, filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p','.data']:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    elif ext == '.npy':
        if not isinstance(obj, np.ndarray):
            obj = np.array(obj)
        np.save(filename, obj)
    else:
        pass # raise Error

def load_pickle(filename):
    _, ext = os.path.splitext(filename)
    if ext in ['.pkl','.p','.data']:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    elif ext == '.npy':
        return np.load(filename)
    else:
        return None # raise Error

def call_openai_gpt(input: str, model_type: str = "gpt-4-1106-preview", stream: bool = False):
    """
    stream: 是否流式响应 建议开启 会减少超时情况，整体任务处理更快
    """
    openai.base_url = 'http://openai.infly.tech/v1/'
    # 不要替换这个 无实际意义, 但是openai sdk不允许这个值为空, 所以随便设置一个值就行
    openai.api_key = 'no-modify' # 不是改这个值，改下面的

    # 不建议指定use_openai, 指定时强制选择请求目标，true:强制使用openai false:强制使用azure
    # extra['use_openai'] = False
    extra = {}

    msg = [
        {"role":"user", "content": input}
    ]

    response = openai.chat.completions.create(
        model = model_type,
        messages = msg,
        extra_headers = {'apikey' : 'sk-ChWx7gQo0pCq4f5Fq8i6ryQhUQytTScWK1eIGrmlwB5fFB9V'},
        extra_body = extra,
        stream = stream,
    )

    if not stream:
        try:
            response_text = response.choices[0].message.content
            return response_text
        except Exception as e:
            print("Error info = ", e)
            print("response = ", response)
            return None
    else:
        encoding = tiktoken.encoding_for_model(model_type)
        complete = 0
        content = ''
        for chunk in response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                # print(chunk.choices[0].delta.content)
                complete += len(encoding.encode(chunk.choices[0].delta.content))
                content += chunk.choices[0].delta.content
        # print(complete)
        # print(content)
        return content

def call_gpt_multiple_times(input, call_num=5, model_type="gpt-4-1106-preview", stream=False):
    while True:
        res = call_openai_gpt(input, model_type=model_type, stream=stream)
        if res:
            return res
        else:
            call_num -= 1
            if call_num == 0:
                return None


if __name__ == "__main__":
    # 统计信息
    # 有大约2263914条推文, 不太能每条都做检索
    # 每个用户有大约54.72条推文
    cascade_dict = load_pickle("/home/wangzejian/data/twitter-withtexts/cascadewithcontent_dict.data")
    # cascade_dict = load_pickle("/home/wangzejian/data/twitter-withtexts/cascade_withtext.pkl")

    user_tweets_dict = {}
    for tag, cascades in cascade_dict.items():
        # print(tag)
        # print(cascades['content'][0])
        for user, tweet in zip(cascades['user'], cascades['content']):
            if user not in user_tweets_dict:
                user_tweets_dict[user] = []
            user_tweets_dict[user].append(tweet)

    logger.info(len(user_tweets_dict))

    # Strategy1: 对每个用户的每条推文做话题分析

    # Strategy2: 对每个用户做话题分析

    prompt_template = "We have several tweets from one particulr user on Twitter. \
    We hope to extract some topics from these tweets, the number of the topics is no more than ten. \n\
    Remember you only have to keep the keywords of these extracted topics, i.e. Movies, Entertainment, etc. \
    Put all topics in a list, and seperate each other with commas.\n\
    Here are the Tweets:"
    logger.info(f"prompt_template: {prompt_template}")

    logger.info("Begin...")
    with open("/home/wangzejian/code/topic_llm/user_topics.jsonl", 'w') as f:
        for user, tweets in user_tweets_dict.items():
            all_tweets = "\n".join(tweets)
            input = f"{prompt_template} {all_tweets}" + "\n\nTopics:"
            res = call_openai_gpt(input, model_type="gpt-4-1106-preview", stream=False)
            logger.info(f"user={user}, res={res}")

            result = {
                "meta_info": {"user": user, "prompt": prompt_template},
                "tweets": all_tweets,
                "topics": res,
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info("Finished...")
