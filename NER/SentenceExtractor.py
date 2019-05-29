from pandas import DataFrame


class SentenceExtractor(object):

    def __init__(self, sentence_label: str):
        self.sentence_label = sentence_label

        self.group_function = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                                      s["POS"].values.tolist(),
                                                                      s["Tag"].values.tolist())]

    def extract(self, data: DataFrame):
        grouped_sentences = data.groupby(self.sentence_label).apply(self.group_function)

        return [sentence for sentence in grouped_sentences]
