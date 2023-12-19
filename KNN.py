import nltk
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
nltk.download('stopwords')

from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer  

import sys
print(sys.executable)

import os                     
import pandas as pd         

from math import log    


def process_mail1(mail):
    mail = mail.lower()
    stopWords = set(stopwords.words('english'))
    stopWords.add('subject')
    words = word_tokenize(mail)
    wordsFiltered = []

    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    stemmer = PorterStemmer()
    wordsFiltered = [stemmer.stem(word) for word in wordsFiltered]
    str = ""
    b = 0
    for i in range(len(wordsFiltered) - 1):
        str += wordsFiltered[i] + ' '
        b += 1
    str += wordsFiltered[b]
    return str

train_mail = pd.DataFrame(columns=['label', 'mail'])

directory = os.path.normpath(".../training/ham/")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f = open(os.path.join(subdir, file), 'r')
            text = f.read()
            text=process_mail1(text)
            new_row = {'label': 0, 'mail': text}
            train_mail = train_mail.append(new_row, ignore_index=True)
            f.close()

directory = os.path.normpath(".../training/spam")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_mail1(text)
            new_row = {'label': 1, 'mail': text}
            train_mail = train_mail.append(new_row, ignore_index=True)
            f.close()


test_mail= pd.DataFrame(columns=['label', 'mail'])

directory = os.path.normpath(".../development/ham")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_mail1(text)
            new_row = {'label': 0, 'mail': text}
            test_mail = test_mail.append(new_row, ignore_index=True)
            f.close()

directory = os.path.normpath(".../development/spam")
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            f=open(os.path.join(subdir, file),'r')
            text = f.read()
            text=process_mail1(text)
            new_row = {'label': 1, 'mail': text}
            test_mail = test_mail.append(new_row, ignore_index=True)
            f.close()

trainData = train_mail
testData = test_mail

class SpamKNN(object):
    def __init__(self, trainData,k_value):
        self.mails, self.labels = trainData['mail'], trainData['label']
        self.k_value=k_value
        self.train()


    def train(self):
        print('knn education...')
        print('The frequency of words and their probability of appearing in all messages are calculated.....')
        self.calc_TF_and_IDF()
        self.calc_TF_IDF()
        print('All training data is scored according to calculations....')
        self.create_neighbour_puan()


    def calc_TF_and_IDF(self):
        noOfMail = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(noOfMail):
            mail_processed = word_tokenize(self.mails[i])
            count = list()
            for word in mail_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1


    def calc_TF_IDF(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log((self.spam_mails + self.ham_mails) \
                                                              / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log((self.spam_mails + self.ham_mails) \
                                                            / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails


    def create_neighbour_puan(self):
        self.komsu_puanlar = pd.DataFrame(columns=['label', 'puan_spam', 'puan_ham'])
        noOfMail = self.mails.shape[0]
        for i in range(noOfMail):
            processed_mail = word_tokenize(self.mails[i])
            puan = self.classify(processed_mail)
            new_row = {'label': self.labels[i], 'puan_spam': puan[0], 'puan_ham': puan[1]}
            self.komsu_puanlar = self.komsu_puanlar.append(new_row, ignore_index=True)
        data_types_dict = {'label': int}
        self.komsu_puanlar = self.komsu_puanlar.astype(data_types_dict)

    def classify(self, processed_mail):
        pSpam, pHam = 0, 0
        for word in processed_mail:
            if word in self.prob_spam:
                pSpam += log(self.prob_spam[word])
            else:
                pSpam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                pHam += log(self.prob_ham[word])
            else:
                pHam -= log(self.ham_words + len(list(self.prob_ham.keys())))
            pSpam += log(self.prob_spam_mail)
            pHam += log(self.prob_ham_mail)
        return [pSpam, pHam]


    def predict(self, testData):
        print('Test data is scored according to calculationsr...')
        print('The differences between spam_score and raw_score between the test data and the train data are calculated with eucledian....')
        print('The majority of neighbors are considered according to the knn value given...')
        print('is classified, predictions are listed...')
        print('This step takes 3 minutes...')

        result = dict()
        for (i, message) in enumerate(testData):
            processed_message = word_tokenize(message)
            result[i] = int(self.knn(processed_message))
        return result


    def knn(self, mail):
        testmail_puan = self.classify(mail)
        return self.oklid_hesaplama(testmail_puan)


    def oklid_hesaplama(self, test_puan):
        puan_spam = self.komsu_puanlar['puan_spam'].values
        puan_ham = self.komsu_puanlar['puan_ham'].values
        labels =  self.komsu_puanlar['label'].values
        noOfMail = labels.shape[0]
        oklid_hesapla = pd.DataFrame(columns=['label', 'eucledian'])

        for i in range(noOfMail):
            eucledian = ((test_puan[0] - puan_spam[i])**2 + abs(test_puan[1] - puan_ham[i])**2)**0.5
            new_row = {'label': labels[i], 'eucledian': eucledian}
            oklid_hesapla = oklid_hesapla.append(new_row, ignore_index=True)

        sorted = oklid_hesapla.sort_values(by=['eucledian'], axis=0)[:self.k_value]['label'].values
        ham_score = 0
        spam_score = 0
        for i in sorted:
            if (i == 1.0):
                spam_score += 1
            else:
                ham_score += 1
        if spam_score > ham_score:
            return 1
        else:
            return 0

def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    Fscore = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    print('\n')
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", Fscore)
    print("Accuracy: ", accuracy)


sc_knn = SpamKNN(trainData,9)
preds_knn = sc_knn.predict(testData['mail'])
metrics(testData['label'], preds_knn)