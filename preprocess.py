import pandas as pd
from sklearn.model_selection import train_test_split
import tokenization
import run_classifier
import emoji
from functools import partial

PATHS = {
    'twitter': 'data/sarcasm_detection_shared_task_twitter_training.jsonl',
    'reddit': 'data/sarcasm_detection_shared_task_reddit_training.jsonl'
}


def context_process(context, context_extent):
    if context_extent == 'last':
        return context[-1]
    elif context_extent == 'all':
        return " ".join(context)


def data_loader(data_type):
    """
    A basic function which reads jsonlines twitter and reddit data
    :param data_type: 'twitter, 'reddit' or 'both'
    :return:
    """
    # Read the data set into a pandas data frame
    res = pd.DataFrame()
    if data_type == 'both':
        datasets = ['twitter', 'reddit']
    else:
        datasets = [data_type]

    for dataset in datasets:
        data = pd.read_json(PATHS[dataset], lines=True)
        res = res.append(data, ignore_index=True)

    return res


def standard_loader(data_type):
    # Use some cute functional binding to reuse our standard_loader
    return partial(data_loader, data_type)


def load_data(
        load_fn=standard_loader('twitter'),
        bert_pair_shape=True, split=None, context_extent='last',
        emoji_totext=False
):
    """
    :param load_fn: a loader function which when executed  should return data frame with 'label',
    'response' and 'context'
    :param bert_pair_shape: True if you want it in shape for bert
    :param split: If none full dataset will be returned, else train, test where split corresponds to frac of test
    :param context_extent: 'last' or 'all'
    :param emoji_totext: process emojis to text if True
    :return:
    """

    # Why do we bother doing this ? We want to pass some arbitrary data not tied to any specific file later.
    res = load_fn()

    # Process emojis if we'd like to
    if emoji_totext:
        res['response'] = res['response'].apply(lambda resp: emoji.demojize(resp))
        res['context'] = res['context'].apply(lambda contexts: [emoji.demojize(c) for c in contexts])

    # Get it into a BERT friendly format
    new_res = []
    if bert_pair_shape:
        i = 1
        for _ind, row in res.iterrows():
            guid = "%d" % i  # A Unique id for each row.
            text_a = tokenization.convert_to_unicode(row['response'])  # The main text which we wish to classify
            # in this Sentence pair classification task, our context is the second sentence.
            text_b = tokenization.convert_to_unicode(context_process(row['context'], context_extent))
            # Sarcasm Label
            label = row['label']

            # Convert every row into a BERT InputExample Object
            new_res.append(run_classifier.InputExample(guid, text_a, text_b, label))
    else:
        new_res = res

    # Split data for test if needed
    if split is None:
        return new_res
    else:
        train, test = train_test_split(new_res, test_size=split, random_state=42)
        return train, test
