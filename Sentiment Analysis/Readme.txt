Here are the steps of our work:

We translated the text and processed them with the Translate_and_sentiments.py : 
	Spliting the text into fittable chunks in order not to loose data
	Then using nltk 'vader_lexicon' to process the sentiments
The results shows a lot of neutrality which could be logic if we supposed that the words translated from hebrew 'loose' meaning with the translation,
for example while 'חמאס' is 96% negative in Hebert it's 100% neutral on nltk

We then used Hebert.py, loading the Bert model from git to analyse sentiment on the datas:
	Spliting the text into fittable chunks in order  not to loose data (512 tokens maximum)
	Then using the pipeline showed in the git to calculate sentiments
The results shows lot of negativity wich seems logic

We then used CNN-token model from the github https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew
	We trained the CNN-token model with 512 tokens instead of 100 
	We got a 0.88 accuracy 
	We splited the data into 512 maximum tokens in order not to loose data,
	adding a weigh function to keep the weight of each chunk in the calculation
	We used the model on A B and C 
The results shows lot of negativity wich seems logic

We then used NER_dicta-il.py to change the names in the A B and C lists of texts
	We changed every 'PER' group to a random first and last name from the moodle xlsx files
	We saved the files to be used with Hebert and CNN-neural

We reused Hebert.py with the files where the names are changed
The result did not really changed

We reused CNN-neural-sentiments.py with the files where the names are changed
The result did not really changed, Here it seems anyway that the names are out of vocabulary,
because after further inspections all the words that are not in the model vocabulary are erased in the predictions,
the changes in the result are probably resulting from the split to chunks function performed in the NER_dicta-il.py 
(the token spliting is not the same, since in the CNN model it's erasing the words out of vocabulary before spliting into chunks)

Note on the CNN-token model from github:
The model is trained with 10 000 lines only that are probably resulting from facebook comments so it didn't really fitted to our exercise,
the training on the model seems to be mainly focus on positives and negatives probabilities (there is very few '2' lines in the training 
wich correspong to neutral), we did some test and the model really struggles in neutrals sentances, resulting in scores around 45/45/10 pos/neg/neu.

Results in results.xlsx
Lior and Mendel