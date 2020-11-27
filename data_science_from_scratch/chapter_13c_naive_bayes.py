from typing import List
from chapter_13_naive_bayes import Message, NaiveBayesClassifier
from chapter_11_machine_learning import split_data
from collections import Counter
import glob, re, random, pprint

path = "spam_data/*/*"
data: List[Message] = []

random.seed(0)
ct = 0
for filename in glob.glob(path):
    is_spam = 'ham' not in filename
    with open(filename, errors='ignore') as email:
        for line in email:
            if 'subject' in line.lower():
                ct += 1
                txt = line.lstrip('Subject: ')
                data.append(Message(txt, is_spam))

train_messages, test_messages = split_data(data, 0.7)

print(len(data), data[0])
print(len(train_messages), len(test_messages))

model = NaiveBayesClassifier()
model.train(train_messages)
predictions = [(message, model.predict(message.text)) for message in test_messages]
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5) 
        for message, spam_probability in predictions)

print(predictions[200])
print(confusion_matrix)

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    prob_if_spam, prob_if_ham = model._probabilities(token)
    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print('Spammiest', words[-50:])
print('Hammiest', words[:50])
