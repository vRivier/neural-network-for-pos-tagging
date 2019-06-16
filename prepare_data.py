from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle



def extract_corpus(filename):
	corpus = list()

	with open('fr-ud-v1.3/fr-ud-'+filename+'.conllu', 'r', encoding='utf-8') as flow:
		for line in flow:
			if len(line)>1:
				if line[0]!='#':
					word, tag = line.split()[1], line.split()[3]
					corpus.append((word, tag))

	print(corpus[0])
	return corpus


train_corpus = extract_corpus('train')
dev_corpus = extract_corpus('dev')
test_corpus = extract_corpus('test')


def get_features(corpus):
	x = list()
	y = list()
	for i in range(len(corpus)):
		word_dict = dict()
		word_dict['previous_word'] = '' if i == 0 else corpus[i-1][0]
		word_dict['current_word'] = corpus[i][0]
		word_dict['next_word'] = '' if i == len(corpus)-1 else corpus[i+1][0]
		# word_dict['previous_tag'] = 
		x.append(word_dict)
		y.append(corpus[i][1])
	return x, y


# {feature: value}
X_train, y_train = get_features(train_corpus)
X_dev, y_dev = get_features(dev_corpus)
X_test, y_test = get_features(test_corpus)

print(len(X_train))


dict_vectorizer = DictVectorizer()
dict_vectorizer.fit(X_train + X_test + X_dev)

X_train = dict_vectorizer.transform(X_train)
X_dev = dict_vectorizer.transform(X_dev)
X_test = dict_vectorizer.transform(X_test)

label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_test + y_dev)

y_train = label_encoder.transform(y_train)
y_dev = label_encoder.transform(y_dev)
y_test = label_encoder.transform(y_test)

print(type(X_train))

pickle.dump(dict_vectorizer, open('d_v','wb'))
pickle.dump(label_encoder, open('l_e', 'wb'))

pickle.dump(X_train, open('xdata','wb'))
pickle.dump(X_dev, open('xdev','wb'))

pickle.dump(y_train, open('ydata','wb'))
pickle.dump(y_dev, open('ydev','wb'))


