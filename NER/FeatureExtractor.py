class FeatureExtractor(object):

    def sentences2features(self, sentences: list):
        return [self.sentence2features(sentence) for sentence in sentences]

    def sentences2labels(self, sentences: list):
        return [self.sentence2labels(sentence) for sentence in sentences]

    def sentence2labels(self, sentence: list):
        return [label for token, postag, label in sentence]

    def sentence2features(self, sentence: list):
        return [self.word2features(sentence, i) for i in range(len(sentence))]

    def word2features(self, sentence: list, i: int):
        word = sentence[i][0]
        postag = sentence[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sentence[i - 1][0]
            postag1 = sentence[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sentence) - 1:
            word1 = sentence[i + 1][0]
            postag1 = sentence[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features
