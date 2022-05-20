Mechanism for predicting Customer Reviews:

![image](https://user-images.githubusercontent.com/73421334/169522173-34af4518-287a-41b9-bfff-d918cbfadd4f.png)


Stemming is widely used for text processing. Stemming is the process of finding root of words by removing either prefix or suffix. It links words with similar meanings to one word. 
Text before stemming: amazing
Text after Stemming: amaze

A machine learning model can’t understand the textual input data on which it has to perform computations to make prediction results. So, We can’t transfer textual data directly to any ML model. Thus, we have to convert the textual data into the numerical form. So, to perform this process we need to use some of the proper NLP (natural language processing) techniques. The key mechanism involved in doing this is to use a bag-of-words (BOW) model.

The bag-of-words model removes or discards the all information involved in the textual data and position of the textual characters present in each word of the sentence and only considers the frequent occurrences of the word. It converts specific kind of textual documents to the specific length list of numbers.

A global vector is implemented by considering all possible words that can occur in English Language,
And each word is assigned with a unique number (commonly index of an array) besides the frequency count which demonstrating the number of occurrences of that specific word in the given sentence. This kind of encoding mechanism on the textual data, in which key focus is to demonstrate the representation of the data but not on the position of the data.

Count Vectorizer is responsible for the tokenization of the textual sentences. This breaks the each textual sentence into collection of key words after performing the important data preprocessing techniques which involve removal of special characters, transforming the all words into lowercase, removing stopwords, applying stemming and lemmatization which find root words of the given word which are further referred to as keywords, etc.


			Fig: Mechanism for predicting customer reviews


The prediction of customer’s textual review as whether +ve /-ve can be done by our collective blended model which involves the following phases.
•	Data preprocessing
•	Prediction using individual models
•	Developing blended model
•	Performing final prediction

Data Preprocessing:
An ML model can’t make predictions on textual data as it operates only on numerical data. So, to transform a textual data to the numerical data, different Natural Language Processing techniques were used. Firstly, the removal of stopwords (like ‘the’, ‘and’, ‘an’,….) from the customer’s textual review gets happened
followed by stemming the each word in the review. Finally, all the selected words from the review can undergo count vectorization technique to transform into the numerical data. The numerical data is formed on the basis of relative positions and corresponding frequencies of all words with respect to all other words in English language.

Prediction using individual models:
Each individual model which is already trained on training data from the dataset make predictions on review given by the customer after performing NLP techniques on review.

Developing Blended model:
Every trained individual model make predictions on cross validation and test set data and the predicted results by each individual model for each record of data are extracted as new features to the same record. Now the new attributes in the cross validation and test sets are (review, LR, DT, SVM, XGB, ANN). Now, a new kind of training data is created which is used to train the meta classifier.



Review	XLR	XDT	XSVM	XXGB	XANN	Y
Numerical_review	Pred_result (LR)	Pred_result (DT)	Pred_result (SVM)	Pred_result
(SVM)	Pred_result (ANN)	Actual_result





Performing final prediction:
Now the trained blended model take the use of prediction results of each individual models and make final prediction based on given review and results predicted by each individual models. In this way the final prediction result is fetched through the blended model.

RECOMMENDING TOP RELEVANT REVIEWS:

We also recommend top relevant positive and negative reviews based on the reviews given by the customers to our system. The important and key words present in each review are extracted and corresponding frequencies are maintained in a database. Now, the frequency cost is calculated for each review as the sum of the frequencies of all keywords present in the customer review. And finally the positive reviews with highest frequency costs are recommended as top positive relevant reviews and the negative reviews with highest frequency costs are recommended as top negative relevant reviews to the owner of the restaurant through our system which gets displayed on the dashboard.

freqCost [ review ] = ∑ freq [ keyword i ]   

where keywordi  is the ith key word present in the given customer review.

