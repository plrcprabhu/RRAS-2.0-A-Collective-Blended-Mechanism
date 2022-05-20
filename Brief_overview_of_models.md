Our Proposed Methodology involves the mechanism of the Collective Blended Model, which takes the use of the blending ensemble method. The following are the algorithms involved in our Collective Blended Model.
●	Decision Tree
●	Logistic Regression
●	Support Vector Machine
●	XG Boost
●	Artificial Neural Networks
Every individual model provides results to some extent. But, the results provided by an individual model might not be optimal. To acquire optimal or better accurate results, there is a need to use a model which combines the properties of all individual models i.e, an ensemble model. So, we introduce a “Collective blended model” which involves the mechanism of the Blending ensemble method. Let us dive into the working process involved in each individual model.

Detailed Overview of each individual model:
Decision Tree :
Decision Trees are one of the most widely used machine learning algorithms and they can be implemented in solving both classification and regression problems. Whether the data present is linear or non-linear, they perform better in every aspect. But, these are mostly used in solving non-linear data. As  the name suggests, each node present in the tree performs decision making which is acquired from the data and its characteristics. The decison trees are easy to implement and derive based on the features derived from the dataset which result in corresponding predictions.
A decision tree is a hierarchical representation of all feasible solutions to a condition or decision based on particular conditions which is done by the different features involved in the data.



![image](https://user-images.githubusercontent.com/73421334/169519610-9a83e9e1-8eab-4e5e-8121-59ce04eb7828.png)



	 


Entropy: It is responsible for identifying the impurity or randomness present in the data. It is given by:

Entropy = -P(class A) * log(P(class A)) – P(class B) * log(P(class C))

where P denotes the probability.

Information Gain: The reduction achieved in the entropy after the data gets splitted based on a feature is known as the “Information gain”. The main task to develop or construct a decision tree is to finding a feature or an attribute which on splitting returns the highest information gain. So, It helps in identifying which attribute or feature will play the key role in creating the most crucial deciding internal node at a specific point.
It is given by:
Information Gain = Entropy(s) — [ Weighted average * (Entropy of each feature/attribute) ]

Decision trees plays a key role in solving any kind of prediction based tasks whether regression or classification as they develop the base of the Ensemble learning techniques especially bagging and boosting, which are the widely used algorithms in the current machine learning domain.


Logistic Regression:

Logistic regression (LR) is one of the most widely used machine learning algorithm for performing classification tasks (especially binary classification) on the linear type of data. The  analysis made by the logistic regression algorithm is key aspect in predicting the likelihood/feasibility of an event happening or a decision being made. This algorithm preserves the marginal probabilities that got achieved of during the training phase of the data. This model is used when we have a binary outcome variable. For example, given the parameters, will the student pass or fail? Will it rain or not? etc., Logistic regression is a special architecture of neural network.

![image](https://user-images.githubusercontent.com/73421334/169519770-12218b69-690f-4e77-bbdd-26ef2151c762.png) ![image](https://user-images.githubusercontent.com/73421334/169519789-4b666b15-f4bc-4b3e-b630-41641e436941.png)



Logistic regression equation:
p = 1 / (1 + e ^ - (a0 + a1x1 + a2 x2 +... + e))
where, p, the probability of achieving outcome
            y, the predicted result
            a0, the intercept or bias parameter term

Sigmoid or Logistic Function:
The logistic function is also known as  the sigmoid function. It is an S-shaped curve that can receive any kind of real-valued number and map it into a value between fractional values of 0 and 1, but never exactly at those limits. The logistic function is described as follows:
	p = 1 / (1 + e ^ -(realValue))
Where, e is known as the base of the natural logarithms and realValue is represented as the original numerical real value that we want to convert or tranform.

The conditional probability of the output variabale (y) :
![image](https://user-images.githubusercontent.com/73421334/169519858-cb2d8192-4e44-4700-ba86-d80aac3fb3cb.png)

			 
maximizing the likelihood of the model:
![image](https://user-images.githubusercontent.com/73421334/169519904-6386bada-3ec8-404a-9c0b-71ca93e54e67.png)

			 
 
Which can be restated as the minimization of the following regularized negative log-likelihood:
![image](https://user-images.githubusercontent.com/73421334/169519920-d58fd841-3c4e-46ea-a087-1e58875dd335.png)



One of the best merits of Logistic Regression is that when we have a complex linear problem and not a whole lot of data it's still even able to produce and achieve much more useful predictions. On the other side, however, the traditional modelling predictions and decisions may lead to underfitting for complex  and  rich datasets.

Support Vector Machine (SVM)
Support Vector Machines (SVMs) are best useful for finding the best line in the 2D geometry and best plane in greater than 2D (like 3D,...) to differentiate the data space into different kind of classes. These are best useful for implementing classification problems, however it can also solve regression problems.
Before the introduction or evolution of the boosting ensemble algorithms (Extreme Gradient Boosting,  etc..) SVM algorithms had been popularly used earlier. The hyperplane is identified or determined through the depiction of maximum margin, i.e., the maximum possible distance between data objects of the all kind of classes.
Hard Margin
Selecting the two hyperplanes with max possible distance among data objects of all kind of classes to separate the data objects according to the corresponding classes.
Soft Margin
Allowance of margin violation under the usage of non-linearly separable data.
Linear SVM : When the data is linearly separable, it is used.
Non-linear SVM :  When data is non-linear, it is used with the use of kernels.

![image](https://user-images.githubusercontent.com/73421334/169519957-f65f423d-1d8b-476f-b3f4-89a3ec6887f7.png)

                                Support Vector Machine for linearly Separable data

![image](https://user-images.githubusercontent.com/73421334/169519980-ed9fce6f-839e-4824-afe1-dbe797997d06.png)

			Fig. Types of Kernel in Support Vector Machine


Extreme Gradient Boosting (XG Boosting):
XGBoost is an efficient and reliable machine learning ensemble technique which is implemented based on the working of gradient boosting. It uses gradient descent algorithm to perform minimization of loss occurred during training and updates the new parameters. It is a decision tree ensemble based on gradient boosting. Variation in loss function  results in identifying and controlling the complexity of performance when constructing the decision trees.

![image](https://user-images.githubusercontent.com/73421334/169520020-0ea194b0-6831-4f9e-a059-5929a4cadc0a.png)

			 
		
Where T = No. of leaves of the tree
	w = Output scores of the leaves.
This loss function plays a crucial role in the integration of the split mechanism involved in the decision trees which lead to the strategical process of pre-pruning. The simpler trees are formed on the basis of higher values of γ result. The minimum loss reduction gain can be found to be controlled by using the values of γ which is highly needed to split a node. An important benefit of using these kind of algorithms is that the models require less storage space and trained relatively faster as compared to other kind of models.
Additionally, randomization techniques are highly implemented in Extreme Gradient Boosting to minimize the probability of occurrence of overfitting and to increase or improve the training process speed. Random subsamples to train the individual decision trees and Column subsampling at decision tree and tree node levels are some of the randomization techniques or mechanisms that were included in the Extreme Gradient Boosting (XGBoost).
				 
![image](https://user-images.githubusercontent.com/73421334/169520044-1a60c5c8-a37e-4a15-a0ed-9fbad13fe77c.png)



Artificial Neural Network (ANN):

An Artificial Neural Network (ANNs) contains collection of layers in linear manner in which every layer consists of some no.of  units which performs a specific task under the usage of an activation function. The input layer takes the data different kind of forms. The data transfers from the input layer units into the one or more hidden unit layers. The hidden units are responsible performing mathematical computations and transform the data from the input unit to the output unit. The artificial neural networks are depicted as fully connected networks from one layer to another layer as they contain some millions of neurons which transfer the data. Some weights are added to each connection. As each information passes through each layer unit in the network is learning more about the information or input data. At output unit’s side, the network responds to the given data and process it.
		
![image](https://user-images.githubusercontent.com/73421334/169520077-44e2054a-cc05-48d6-85f9-a9f7a85afbc5.png)

		 
  		Fig. Artificial Neural Networks for Binary Classification
In the above ANN Architecture, the primary layer in the hidden layer set is used as Universal Sentence Encoder Layer. As we know hidden layer is less responsible for learning of the model for sentiment analysis on numerical data. So, we use Universal Sentence Encoder Layer to encode the review sentences from the training data and convert them into numerical data which is operable by ANN model.

Universal Sentence Encoder:
The model architecture consists of two encoders, which have two different goals. First one focusses on
Higher accuracy, but requires huge resource consumption. The other focusses on efficient inference, but
with little lesser accuracy. The model transformer, a textual sentence  model used for encoding frames the every textual sentence embedding process using the encoding sub-graph in the architecture of the transformer. The encoder receives the tokenized string in the lowercase order as the input data and produces a 512 dimensional vector as the sentence embedding in the form of output. The transformed based sentence encoding model achieves overall best task performance. But it requires more computational time and also consumes more memory when the sentence length is more.
The second encoding model which is known as the Deep Averaging Network (DAN) where input textual data embeddings for the words and bi-grams are formed as an average and then transferred through a feed-forward deep neural network to generate the finest textual sentence embeddings. DAN achieves strong baseline performance for classification of text reviews. For training encoder data, we have collected a huge responses from customers in different restaurants through online web sources and thus created a dataset.
		
![image](https://user-images.githubusercontent.com/73421334/169520104-db5b2174-b53f-484d-9398-6841b75640b6.png)


	Fig. Similarity scores of the different kind of sentences using corresponding embeddings from the universal sentence encoder layer.
Perceptron: This is the simple form of the artificial neural network which contains m number of inputs, just one neuron, and one output, where m is represented as the no.of  attributes present in our dataset.
Backpropagation: Backpropagation, which is also known as the backward propagation of errors, is an algorithm which is used for calculating the gradient of the error loss function under the consideration of weights of each layer in the network. 
Optimization:  Optimization is the process of selecting the best weights in the each layer which provide best accurate results and perform great predictions. The selection of best bias of the perceptron involves the usage of implementing the gradient descent algorithm. The weights and bias gets updated until the convergence occurs during the optimization process. Learning rate (α) controls the how much impact should be done on the weights and bias by acting as a hyper parameter. The weights and bias will go under  the updation process till convergence got occurred between the parameters in the data.
	
![image](https://user-images.githubusercontent.com/73421334/169520126-07419f16-52b1-4f56-9884-e602c3a189f8.png)



Working of Collective Blended Model:

Collective Blended model uses the Blending ensemble method in the process of sentiment analysis of customer reviews and predicting the accurate status of the customer reviews. Following is the brief overview about blending technique.

“Blending” is a kind of ensemble technique which is also known as stacking generalization. In blending,
Firstly, each individual model perform under training phase on the training data and performs predictions on the validation set and test set data. The results made by the each individual model are formed as new features and added to the corresponding subsequent input data present on the validation set and test set. Now, the meta classifier (collective blended model) model gets trained on the predictions results made by each individual model (base model)  that were made on a separate cross validation set of the dataset instead of full and folded training set. And finally, our collective blended model which is also known as the meta classifier model perform predictions on newly formed test set with added features as base model predictions. It makes the predictions which are more accurate than the results achieved by each and every single individual model used. 

Steps involved in Working Mechanism of Blending Ensemble Model:

Step-1: Training dataset is split into base train data, validation data and test data.
Step-2: Train the individual models i.e., Decision Tree, Logistic Regression, Support Vector Machine (SVM), XG Boost, Artificial Neural Networks (ANN) on the training data of our dataset and perform predictions on validation set and test set data. The creation of new predictions made by this mechanism.
Step-3: A Collective Blended Model, a new meta-classifier model is then fitted on validation/holdout set using individual model prediction attributes which are the results made by each individual model. For this, both actual features/attributes and new meta features from the holdout or validation  set will be used.
Step-4: Finally, this trained collective blended model is used to perform the crucial final predictions on the test set data which involves the usage of original and new blended model (meta) features.
	
![image](https://user-images.githubusercontent.com/73421334/169520164-5f6bfcce-6157-450a-b3da-c0bf2a634419.png)

 
