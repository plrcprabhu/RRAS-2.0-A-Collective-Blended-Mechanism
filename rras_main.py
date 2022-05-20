#################     RESTAURANT   REVIEW   ANALYSIS    SYSTEM     ##################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from math import *
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tkinter import *
from tkinter import messagebox
import sqlite3

############

conn=sqlite3.connect('Restaurant_food_data.db')
c=conn.cursor()

train_ratio = 0.70
validation_ratio = 0.20
test_ratio = 0.10

Cust_Review_Data = pd.read_csv('Customer_Review_Data.tsv', delimiter = '\t', quoting = 3)
preprocessed_reviews = []
rras_code="Wyd^H3R"
food_rev={}
food_perc={}
for i in range(0, 5530):
	review = re.sub('[^a-zA-Z]', ' ', Cust_Review_Data['Review'][i])
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	all_stopwords = stopwords.words('english')
	all_stopwords.remove('not')
	review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
	review = ' '.join(review)
	preprocessed_reviews.append(review)

cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(preprocessed_reviews).toarray()
y = Cust_Review_Data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
Y_test=y_test
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=5)

model_1 = DecisionTreeClassifier(criterion = 'entropy')
model_2 = LogisticRegression()
model_3 = SVC(kernel="linear")
model_4 = XGBClassifier()

# training all the model on the train dataset
 
# training first model
model_1.fit(x_train, y_train)
val_pred_1 = model_1.predict(x_val)
test_pred_1 = model_1.predict(x_test)
 
# converting to dataframe
#val_pred_1 = pd.DataFrame({'NB':val_pred_1.tolist()})
#test_pred_1 = pd.DataFrame({'NB':test_pred_1.tolist()})

val_pred_1.tolist()
test_pred_1.tolist()
val_pred_1=list(map(int,val_pred_1))
test_pred_1=list(map(int,test_pred_1))
Y_pred1=np.concatenate((val_pred_1,test_pred_1),axis=0)
#print("DT : ",accuracy_score(Y_test,Y_pred1))
#print("DT : ",accuracy_score(y_test,test_pred_1))

# training second model
model_2.fit(x_train, y_train)
val_pred_2 = model_2.predict(x_val)
test_pred_2 = model_2.predict(x_test)
Y_pred2=np.concatenate((val_pred_2,test_pred_2),axis=0)
#print("LR : ",accuracy_score(Y_test,Y_pred2))
#print("LR : ",accuracy_score(y_test,test_pred_2))
# converting to dataframe
#val_pred_2 = pd.DataFrame({'LR':val_pred_2.tolist()})
#test_pred_2 = pd.DataFrame({'LR':test_pred_2.tolist()})

val_pred_2.tolist()
test_pred_2.tolist()
val_pred_2=list(map(int,val_pred_2))
test_pred_2=list(map(int,test_pred_2))

# training third model
model_3.fit(x_train, y_train)
val_pred_3 = model_3.predict(x_val)
test_pred_3 = model_3.predict(x_test)
Y_pred3=np.concatenate((val_pred_3,test_pred_3),axis=0)
#print("SVC : ",accuracy_score(Y_test,Y_pred3)) 
#print("SVC : ",accuracy_score(y_test,test_pred_3))
# converting to dataframe
#val_pred_3 = pd.DataFrame({'SVC':val_pred_3.tolist()})
#test_pred_3 = pd.DataFrame({'SVC':test_pred_3.tolist()})

val_pred_3.tolist()
test_pred_3.tolist()
val_pred_3=list(map(int,val_pred_3))
test_pred_3=list(map(int,test_pred_3))

# training fourth model
model_4.fit(x_train, y_train)
val_pred_4 = model_4.predict(x_val)
test_pred_4 = model_4.predict(x_test)
Y_pred4=np.concatenate((val_pred_4,test_pred_4),axis=0)
#print("XGB : ",accuracy_score(Y_test,Y_pred4)) 
#print("XGB : ",accuracy_score(y_test,test_pred_4))
# converting to dataframe
#val_pred_4 = pd.DataFrame({'XGB':val_pred_4.tolist()})
#test_pred_4 = pd.DataFrame({'XGB':test_pred_4.tolist()})

val_pred_4.tolist()
test_pred_4.tolist()

val_pred_4=list(map(int,val_pred_4))
test_pred_4=list(map(int,test_pred_4))

#ANN

path = 'Customer_Review_Data.tsv'
df = pd.read_csv(path, sep='\t')
df_sentence = df["Review"]
df_label = df["Liked"]
# Split the sentences and labels into training and validation sets
#print(type(df_sentence))
train_sentences, val_sentences, train_labels, val_labels = train_test_split(df_sentence.to_numpy(),
                                                                            df_label.to_numpy(),
                                                                            test_size=0.3, 
                                                                            random_state=42)
#print(type(val_sentences))
use_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"

sentence_encoder_layer = hub.KerasLayer(use_url,
                                        # shape of inputs coming to our model 
                                        input_shape=[],
                                        # data type of inputs coming to the USE layer
                                        dtype=tf.string,
                                        # keep the pretrained weights (this is a feature 
                                        # extractor without fine-tuning)
                                        trainable=False, 
                                        name="USE")
# Build model

model_5 = tf.keras.Sequential([
  sentence_encoder_layer,
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
], name="model_USE_large_v5")

ckpt_path="./train_model/ann_checkpoint.ckpt"


# Compile model
'''model_5.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])



model_checkpoint = ModelCheckpoint(filepath=ckpt_path,
                                   # set to False to save the entire model
                                   save_weights_only=True,
                                   # set to False to save every model every epoch
                                   save_best_only=True,
                                   save_freq="epoch",
                                   monitor="val_loss",
                                   verbose=1)

early_stopping = EarlyStopping(patience=5, 
                               monitor="val_loss",
                               verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=2, 
                              min_lr=0.0005,
                              verbose=1)

callbacks=[model_checkpoint, 
           early_stopping, 
           reduce_lr]
# Train the model
EPOCHS = 50

history_model = model_5.fit(train_sentences,
                          train_labels,
                          epochs=EPOCHS,
                          validation_data=(val_sentences, val_labels),
	          callbacks=callbacks)'''

# Loads the weights
model_5.load_weights(ckpt_path)
#print(len(val_sentences))
# Re-evaluate the model
#loss, acc = model_5.evaluate(val_sentences, verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

val_sentences, test_sentences, val_labels, test_labels = train_test_split(val_sentences,
                                                                            val_labels,
                                                                            test_size=test_ratio/(test_ratio + validation_ratio), 
                                                                            random_state=5)
#print(val_sentences)
#print(type(val_sentences))
model_pred_val_probs = model_5.predict(val_sentences)
model_val_preds = tf.squeeze(tf.round(model_pred_val_probs))

model_pred_test_probs = model_5.predict(test_sentences)
model_test_preds = tf.squeeze(tf.round(model_pred_test_probs))

#val_pred_5 = pd.DataFrame({'ANN':model_val_preds.numpy().tolist()})
#test_pred_5 = pd.DataFrame({'ANN':model_test_preds.numpy().tolist()})

model_val_preds.numpy().tolist()
model_test_preds.numpy().tolist()

val_pred_5=list(map(int,model_val_preds))
#print(val_pred_5)

test_pred_5=list(map(int,model_test_preds))
Y_pred5=np.concatenate((val_pred_5,test_pred_5),axis=0)
#print("ANN : ",accuracy_score(Y_test,Y_pred5))
#print("ANN : ",accuracy_score(y_test,test_pred_5))

#Calculate the metrics Accuracy, Precision, Recall and F1-Score
def calculate_results(y_true, y_pred):
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, 
                                                                                 y_pred, 
                                                                                 average="weighted")
    model_results = {"accuracy": round(model_accuracy, 4),
                  "precision": round(model_precision, 4),
                  "recall": round(model_recall, 4),
                  "f1": model_f1, "Confusion Matrix" : confusion_matrix(y_true,y_pred)}
    return model_results

#print(calculate_results(val_labels, model_preds))
#res=calculate_results(val_labels, model_preds)
#acc5=round(res["accuracy"],3)

# concatenating validation dataset along with all the predicted validation data (meta features)

#x_val_df=pd.DataFrame({"X": x_val.tolist()})
#x_test_df=pd.DataFrame({"X": x_test.tolist()})

x_val=x_val.tolist()
x_test=x_test.tolist()
#x_val=list(map(int,x_val))
#x_test=list(map(int,x_test))
#val_models=[val_pred_1,val_pred_2,val_pred_3,val_pred_4,val_pred_5]
#test_models=[test_pred_1,test_pred_2,test_pred_3,test_pred_4,test_pred_5]
#print(type(x_val))

val_models=[val_pred_1,val_pred_2,val_pred_3,val_pred_4,val_pred_5]
test_models=[test_pred_1,test_pred_2,test_pred_3,test_pred_4,test_pred_5]

#print(x_val)
#print(val_models)
#print(test_models)
for i in range(len(x_val)):
	extra_features=[]
	for j in range(len(val_models)):
		extra_features+=[val_models[j][i]]
	x_val[i]+=extra_features

for i in range(len(x_test)):
	extra_features=[]
	for j in range(len(test_models)):
		extra_features+=[test_models[j][i]]
	x_test[i]+=extra_features

fin_x_val=np.array(x_val)
fin_x_test=np.array(x_test)

#df_val = pd.concat([x_val_df, val_pred_1, val_pred_2, val_pred_3,val_pred_4,val_pred_5], axis=1)
#df_test = pd.concat([x_test_df, test_pred_1, test_pred_2, test_pred_3,test_pred_4,test_pred_5], axis=1)

#print(df_val)
#print(y_val)

# making the final model using the meta features

final_model = LogisticRegression()

final_model.fit(fin_x_val, y_val)

#BLENDING
final_pred = final_model.predict(fin_x_test)
#print("#########################")
#print("LR : ",accuracy_score(y_test, final_pred) * 100)
#print("#########################")

####Data Visualization####

test_preds=[test_pred_1,test_pred_2,test_pred_3,test_pred_4,test_pred_5]
#dt,lr,svm,xgb,ann,model=round(accuracy_score(y_test,test_pred_1),2)*100,round(accuracy_score(y_test,test_pred_2),2)*100,round(accuracy_score(y_test,test_pred_3),2)*100,round(accuracy_score(y_test,test_pred_4),2)*100,round(accuracy_score(y_test,test_pred_5),2)*100,round(accuracy_score(y_test,final_pred),2)*100
#dt,lr,svm,xgb,ann,model=accuracy_score(y_test,test_pred_1)*100,accuracy_score(y_test,test_pred_2)*100,accuracy_score(y_test,test_pred_3)*100,accuracy_score(y_test,test_pred_4)*100,accuracy_score(y_test,test_pred_5)*100,accuracy_score(y_test,final_pred)*100

scores=[]
for test_pred in test_preds:
	acc=accuracy_score(y_test,test_pred)*100
	model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_test, test_pred,average="weighted")
	scores.append([acc,model_precision,model_recall,model_f1])

fin_acc=accuracy_score(y_test,final_pred)*100
fin_model_precision, fin_model_recall, fin_model_f1, _ = precision_recall_fscore_support(y_test, final_pred,average="weighted")
scores.append([fin_acc,fin_model_precision,fin_model_recall,fin_model_f1])

dt,lr,svm,xgb,ann,cbmodel=round(scores[0][0],4),round(scores[1][0],4),round(scores[2][0],4),round(scores[3][0],4),round(scores[4][0],4),round(scores[5][0],4)

data={"Decision Tree":dt,"Logistic Regression":lr,"Support Vector Machine":svm,"XGBoost":xgb,"Artificial Neural Networks":ann,"Collective Blended Model":cbmodel}
Models = ["DT","LR","SVM","XGB","ANN","Blended Model"]
Accuracies = list(data.values())
k=0
for model in data:
	print("\n\n##########################    "+model+"     #########################\n")
	print("Accuracy\tPrecision\tRecall\tF1-Score")
	for x in range(len(scores[k])):
		print(str(round(scores[k][x],4)),end="        ")
	k+=1
print("\n")
print("Confusion Matrix(Blended Model) : ")
print(confusion_matrix(y_test,final_pred))
print("\n\n") 
  
fig = plt.figure(figsize = (8,8))
 
# creating the bar plot
plt.bar(Models, Accuracies, color ='black',
        width = 0.4)

'''for i in range(1,len(Models)+1):
	plt.annotate(Accuracies[i-1],(i,Accuracies[i-1]),ha="center")''' 
plt.xlabel("Classification Models")
plt.ylabel("Accuracy Measures")
plt.title("Performance of different ML models in classifying customer reviews")
plt.show()

###################

variables=[]
clr_variables=[]


c.execute("SELECT *,oid FROM item")
records=c.fetchall()


foods=[list(record)[0] for record in records] 
for i in foods:
	food_rev[i]=[]
	food_perc[i]=[0.0,0.0]

bgcolor="#96C3EB"

def estimate(s):

	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	review = re.sub('[^a-zA-Z]', ' ', s)
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()
	all_stopwords = stopwords.words('english')
	all_stopwords.remove('not')
	review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
	review = ' '.join(review)
	X = cv.transform([review]).toarray()	
	nb=model_1.predict(X).tolist()[0] #list
	lr=model_2.predict(X).tolist()[0]
	svm=model_3.predict(X).tolist()[0]
	xgb=model_4.predict(X)[0]
	s=np.array([s])
	mpp = model_5.predict(s)
	mp = tf.squeeze(tf.round(mpp))
	mp=mp.numpy().tolist()
	ann=int(mp)
	X=X.tolist()
	X[0]+=[nb,lr,svm,xgb,ann]	
	res=final_model.predict(X)
	selected_foods=[]
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	#print([variables[i].get() for i in range(len(variables))])
	for i in range(len(foods)):
		if variables[i].get()==1:
			selected_foods.append(foods[i])  
	#update data
	for i in records:
		rec=list(i)
		if rec[0] in selected_foods:
			n_cust=int(rec[1])+1
			n_pos=int(rec[2])
			n_neg=int(rec[3])
			if res[0]==1:
				n_pos+=1
			else:
				n_neg+=1
			pos_percent=round((n_pos/n_cust)*100,1)
			neg_percent=round((n_neg/n_cust)*100,1)
			c.execute("""UPDATE item SET No_of_customers=:no_of_customers,No_of_positive_reviews=:no_of_positives,No_of_negative_reviews=:no_of_negatives,Positive_percentage=:pos_perc,Negative_percentage=:neg_perc  where Item_name=:item_name""",
				{
					'item_name': rec[0],
					'no_of_customers': str(n_cust),
					'no_of_positives': str(n_pos),
					'no_of_negatives': str(n_neg),
					'pos_perc': str(pos_percent)+"%",
					'neg_perc': str(neg_percent)+"%"      
				}
			)
	selected_foods=[]
	stemmed_review=review
	actual_review=str(s[0])
	c.execute("select *,oid from freqdata")
	records=c.fetchall()
	words=[list(record)[0] for record in records]
	up_freq=0
	freq={}
	for wrd in stemmed_review:
		if wrd in words:
			c.execute("select Count from freqdata where Word=:wwrd",
				{
					'wwrd':wrd	
				}
			)
			ccnt=list(c.fetchall()[0])[0]
			c.execute("""UPDATE freqdata SET Count=:cnt where Word=:wwrd""",
				{
					'cnt': int(ccnt)+1,
					'wwrd': wrd
				}
			)
			freq[wrd]=int(ccnt)+1
		else:
			c.execute("INSERT INTO freqdata VALUES(:wwrd,:cnt)",
				{
					'wwrd': wrd,
					'cnt': "1"
				}	
			)
			freq[wrd]=1
	for wrd in freq:
		up_freq+=freq[wrd]
	c.execute("select *,oid from reviewdata")
	records=c.fetchall()
	reviews=[list(record)[0] for record in records]
	if actual_review not in reviews:
		c.execute("INSERT INTO  reviewdata VALUES(:act_rev,:stat,:cost)",
			{
				'act_rev': actual_review,
				'stat': str(res[0]),
				'cost': up_freq
			}
		)
	conn.commit()
	conn.close()
	
root1=Tk()
main="Restaurant Review Analysis System/"
root1.title(main+"Welcome Page")
root1["bg"]=bgcolor
# getting screen's height in pixels
'''height = root1.winfo_screenheight()
 
# getting screen's width in pixels
width = root1.winfo_screenwidth()
root1.geometry('%dx%d+0+0' % (width,height))'''

def init_data():

	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	for i in range(len(foods)):  
		c.execute("INSERT INTO item VALUES(:item_name,:no_of_customers,:no_of_positives,:no_of_negatives,:pos_perc,:neg_perc)",
				{
					'item_name': foods[i], 
					'no_of_customers':"0",
					'no_of_positives':"0",
					'no_of_negatives':"0",
					'pos_perc':"0.0%",
					'neg_perc':"0.0%"
				}
			)
	conn.commit()
	conn.close()

def clr_data():
	
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	for i in range(len(clr_variables)):
		if clr_variables[i].get()==1:
			
			c.execute("""UPDATE item SET Item_name=:item_name,No_of_customers=:no_of_customers,No_of_positive_reviews=:no_of_positives,No_of_negative_reviews=:no_of_negatives,Positive_percentage=:pos_perc,Negative_percentage=:neg_perc  where oid=:Oid""",
				{
					'item_name': foods[i],  
					'no_of_customers': "0",
					'no_of_positives': "0",
					'no_of_negatives': "0",
					'pos_perc': "0.0%",
					'neg_perc': "0.0%",
					'Oid': i+1
				}
			)

	conn.commit()
	conn.close()

def init_limitData():

	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	for i in range(len(foods)):  
		c.execute("INSERT INTO limitt VALUES(:item_name,:cust_lim)",
				{
					'item_name': foods[i], 
					'cust_lim': 5
				}
			)	
	conn.commit()
	conn.close()
	
def popup():
	messagebox.showerror("Error Message!","Incorrect code!")

def notifyOwner(val,root):
	if(val==1):
		messagebox.showerror("Error Message!","Item Already Exists!!",parent=root)
	else:
		messagebox.showerror("Error Message!","Item doesn't exist!!",parent=root)	

def show_selected_food(var,foods):
	sel=[]
	for i in range(len(foods)):
		if var[i].get()==1:
			sel.append(foods[i])
	#print(sel)
	return sel
def access_data():

	root5=Toplevel()
	root5.title(main+"Restaurant_Database")
	root5["bg"]=bgcolor
	label=Label(root5,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
	title1=Label(root5,text="S.NO",font=('Arial',10,'bold','underline'),bg="black",fg="white")
	title2=Label(root5,text="FOOD ITEM",font=('Arial',10,'bold','underline'),bg=bgcolor)
	title3=Label(root5,text="NO.OF CUSTOMERS",font=('Arial',10,'bold','underline'),bg=bgcolor)
	title4=Label(root5,text="NO.OF POSITIVE REVIEWS",font=('Arial',10,'bold','underline'),bg=bgcolor)
	title5=Label(root5,text="NO.OF NEGATIVE REVIEWS",font=('Arial',10,'bold','underline'),bg=bgcolor)
	title6=Label(root5,text="POSITIVE RATE",font=('Arial',10,'bold','underline'),bg=bgcolor)
	title7=Label(root5,text="NEGATIVE RATE",font=('Arial',10,'bold','underline'),bg=bgcolor)
	
	label.grid(row=0,column=0,columnspan=7)
	title1.grid(row=1,column=0,sticky=W+E)
	title2.grid(row=1,column=1)
	title3.grid(row=1,column=2)
	title4.grid(row=1,column=3)
	title5.grid(row=1,column=4)
	title6.grid(row=1,column=5)
	title7.grid(row=1,column=6)
	
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	
	c.execute("SELECT *,oid from item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]	
	pos_rates=[]
	#print(records)
	for i in range(len(records)):
		rec_list=list(records[i])
		fooditem=foods[i]
		cust=int(rec_list[1])
		c.execute("SELECT Cust_limit from limitt where Food_item=:itemname",
				{
					'itemname': fooditem
				}
			)
		cust_lim=c.fetchall()
		cust_lim=int(list(cust_lim[0])[0])
		if(cust>=cust_lim):
			pos_rates.append(float(rec_list[-3][:-1]))

	#print(pos_rates)
	max_pos=max(pos_rates)
	min_pos=min(pos_rates)
	k=0
	not_eligible_foods=[]
	for i in range(len(records)):
		rec_list=list(records[i])
		fooditem=foods[i]
		cust=int(rec_list[1])
		c.execute("SELECT Cust_limit from limitt where Food_item=:itemname",
				{
					'itemname': fooditem
				}
			)
		cust_lim=c.fetchall()
		cust_lim=int(list(cust_lim[0])[0])
		
		if(cust>=cust_lim):
			if str(max_pos)+"%"==rec_list[-3]:
				rec_lab=[Label(root5,text=str(k+1),font=("Helevetica",10,"bold"),bg="green")]
				for item in rec_list[:-1]:
					lab=Label(root5,text=item,font=("Helevetica",10,"bold"),bg="green")
					rec_lab.append(lab)
			elif str(min_pos)+"%"==rec_list[-3]:
				rec_lab=[Label(root5,text=str(k+1),font=("Helevetica",10,"bold"),bg="red")]
				for item in rec_list[:-1]:
					lab=Label(root5,text=item,font=("Helevetica",10,"bold"),bg="red")
					rec_lab.append(lab)
			else:
				rec_lab=[Label(root5,text=str(k+1),font=("Helevetica",10,"bold"),bg="black",fg="white")]
				for item in rec_list[:-1]:
					lab=Label(root5,text=item,font=("Helevetica",10,"bold"))
					rec_lab.append(lab)
		
			for j in range(len(rec_lab)):
				rec_lab[j].grid(row=i+2,column=j,sticky=W+E)
			k+=1
		else:
			not_eligible_foods.append(fooditem)	
	r=len(records)+10
	flag=0
	if(not_eligible_foods):
		lab1=Label(root5,text="Food Items which didn't reach their customer limits are :",font=("Helevetica",10,"bold","underline"),bg=bgcolor)
		lab1.grid(row=r,column=0,columnspan=4,sticky=W)
		l=len(not_eligible_foods)
		for x in range(l//7):
			for y in range(7):
				itemm=not_eligible_foods[x*7+y]
				lab=Label(root5,text=itemm,bg=bgcolor)
				lab.grid(row=r+x+1,column=y)
		extn=l-(l//7)*7
		for y in range(extn):
			itemm=not_eligible_foods[(l//7*7)+y]
			lab=Label(root5,text=itemm,bg=bgcolor)
			lab.grid(row=(l//7+r+1),column=y)
			flag=1
	
	exit_btn=Button(root5,text="Back",command=root5.destroy)
	exit_btn.grid(row=len(records)+len(not_eligible_foods)//7+flag+15,column=3)
	root5.state('zoomed')
	conn.commit()
	conn.close()

def clr_itemdata():
	root6=Toplevel()
	root6.title(main+"clear_item_data")
	root6["bg"]=bgcolor
	label=Label(root6,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	req1=Label(root6,text="Pick the items to clear their corresponding item data....",bg=bgcolor)
	chk_list=[]
	global clr_variables
	clr_variables=[]
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	for i in range(len(foods)):   
		var=IntVar()
		chk=Checkbutton(root6,text=foods[i],variable=var,bg=bgcolor)    
		clr_variables.append(var)
		chk_list.append(chk)
	label.grid(row=0,column=0,columnspan=4)
	req1.grid(row=1,column=0,columnspan=4,sticky=W+E)
	req1.config(font=("Helvetica",30))
	n=len(foods)
	for i in range(n//4):
		for j in range(4):
			c=chk_list[i*4+j]
			c.grid(row=i+3,column=j,columnspan=1,sticky=W)
	j=0
	for i in range((n//4)*4,n):
		c=chk_list[i]
		c.grid(row=(n//4)+3,column=j,columnspan=1,sticky=W)
		j+=1
	clr_item=Button(root6,text="Clear",font=('Arial',20),padx=100,pady=20,command=lambda : [clr_data(),root6.destroy()])
	clr_item.grid(row=(n//4)+5,column=0,columnspan=4)
	root6.state('zoomed')
	conn.commit()
	conn.close()

def clr_alldata(root):
	confirm=messagebox.askquestion("Confirmation","Are you sure to delete all data??",parent=root)
	if confirm=="yes":
		conn=sqlite3.connect('Restaurant_food_data.db')
		c=conn.cursor()
		c.execute("SELECT *,oid FROM item")
		records=c.fetchall()
		foods=[list(record)[0] for record in records]
		for i in range(len(foods)):  
			c.execute("""UPDATE item SET Item_name=:item_name,No_of_customers=:no_of_customers,No_of_positive_reviews=:no_of_positives,No_of_negative_reviews=:no_of_negatives,Positive_percentage=:pos_perc,Negative_percentage=:neg_perc  where oid=:Oid""",
					{
						'item_name': foods[i],  
						'no_of_customers':"0",
						'no_of_positives':"0",
						'no_of_negatives':"0",
						'pos_perc':"0.0%",
						'neg_perc':"0.0%",
						'Oid':i+1					
					}
				)
		conn.commit()
		conn.close()

def addFoodItem(foodItem,field,root):
	
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0].lower() for record in records]
	if(foodItem and foodItem.lower() not in foods):
		c.execute("INSERT INTO item VALUES(:item_name,:no_of_customers,:no_of_positives,:no_of_negatives,:pos_perc,:neg_perc)",
				{
					'item_name': foodItem,
					'no_of_customers':"0",
					'no_of_positives':"0",
					'no_of_negatives':"0",
					'pos_perc':"0.0%",
					'neg_perc':"0.0%"
				}
			)
		c.execute("INSERT INTO limitt VALUES(:item_name,:cust_lim)",
				{
					'item_name': foodItem,
					'cust_lim': 5
				}
			)
		messagebox.showinfo("showinfo", "Food item successfully added!!",parent=root)
		field.delete(0,END)
	else:
		notifyOwner(1,root)
	
	conn.commit()
	conn.close()

def setLimit(root):

	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]	
	for i in range(len(foods)):  
		c.execute("""UPDATE limitt SET Cust_limit=:cust_lim where Food_item=:item_name""",
				{
					'item_name': foods[i],
					'cust_lim': str(scl_list[i].get())				
				}
			)
	messagebox.showinfo("showinfo","Successfully Updated Customer limits!!",parent=root)
	conn.commit()
	conn.close()

def askToAdd():
	root7=Toplevel()
	root7.title(main+"add_food_item")
	root7["bg"]=bgcolor
	label=Label(root7,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
	add_lab=Label(root7,text="Enter food Item",bg=bgcolor)
	item_tf=Entry(root7,width=125,borderwidth=5)
	add_btn=Button(root7,text="Add",font=('Arial',20),padx=20,pady=2,command=lambda : [addFoodItem(item_tf.get(),item_tf,root7)])
	exit_btn=Button(root7,text="Back",command=root7.destroy)
	root7.state("zoomed")
	label.grid(row=0,column=0)
	add_lab.grid(row=2,column=0)
	add_lab.config(font=("Helvetica",30))														
	item_tf.grid(row=3,column=0,sticky=S)
	add_btn.grid(row=4,column=0)
	exit_btn.grid(row=6,column=0)

def deleteFoodItem(root):
	
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	selected_foods=[]
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	for i in range(len(foods)):
		if variables[i].get()==1:
			selected_foods.append(foods[i])
	n=len(selected_foods)
	for itemname in selected_foods:
		qry1="DELETE from item where Item_name=?"
		c.execute(qry1,(itemname,))
		qry2="DELETE from limitt where Food_item=?"
		c.execute(qry2,(itemname,))

	extn="s" if n!=1 else ""	
	messagebox.showinfo("showinfo", "Successfully deleted "+str(n)+" food item"+extn+" !",parent=root)
	conn.commit()
	conn.close()
	
def setCustLimit():
	
	root9=Toplevel()
	root9.title(main+"set_customer_limit")
	root9["bg"]=bgcolor
	label=Label(root9,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
	
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	selected_foods=[]
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	label.grid(row=0,column=0,columnspan=4)
	n=len(foods)
	global scl_list
	scl_list=[]
	c.execute("SELECT *,oid FROM limitt")
	limitt_records=c.fetchall()
	#print(limitt_records)
	limits=[list(limitt_record)[1] for limitt_record in limitt_records]
	for i in range(n):
		r,c=(i//4),i-(i//4)*4
		if(r%2==c%2):
			s=Scale(root9,bd=2,orient=HORIZONTAL,from_=1,to=30,troughcolor="black",fg="black",bg="violet",length=300,label=foods[i],cursor="arrow")
		else:
			s=Scale(root9,bd=2,orient=HORIZONTAL,from_=1,to=30,troughcolor="black",length=300,label=foods[i],cursor="arrow")
		s.set(limits[i])
		scl_list.append(s)
	for i in range(n//4):
		for j in range(4):
			c=scl_list[i*4+j]
			c.grid(row=i+3,column=j,columnspan=1,sticky=W)
	j=0
	for i in range((n//4)*4,n):
		c=scl_list[i]
		c.grid(row=(n//4)+3,column=j,columnspan=1,sticky=W)
		j+=1
	rno=(n//4)+5
	set_btn=Button(root9,text="Set",font=('Arial',20),padx=100,pady=20,command=lambda : [setLimit(root9)])
	set_btn.grid(row=rno,column=1,rowspan=3,columnspan=2,sticky=S)
	exit_btn=Button(root9,text="Back",command=root9.destroy)
	exit_btn.grid(row=rno+3,column=1,columnspan=2)
	root9.state('zoomed')
	conn.commit()
	conn.close()
	
def askToDel(root):
	root8=Toplevel()
	root8.title(main+"delete_food_item")
	root8["bg"]=bgcolor
	label=Label(root8,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")	
	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	req1=Label(root8,text="Choose the item(s) you want to delete",bg=bgcolor)
	global variables
	variables=[]
	chk_list=[]
	for i in range(len(foods)):  
		var=IntVar()
		chk=Checkbutton(root8,text=foods[i],variable=var,bg=bgcolor)  
		variables.append(var)
		chk_list.append(chk)
	label.grid(row=0,column=0,columnspan=4)
	req1.grid(row=1,column=0,columnspan=4,sticky=W+E)
	req1.config(font=("Helvetica",30))
	
	n=len(foods)
	for i in range(n//4):
		for j in range(4):
			c=chk_list[i*4+j]
			c.grid(row=i+3,column=j,columnspan=1,sticky=W)
	j=0
	for i in range((n//4)*4,n):
		c=chk_list[i]
		c.grid(row=(n//4)+3,column=j,columnspan=1,sticky=W)
		j+=1
	rno=(n//4)+5		
	del_btn=Button(root8,text="Delete",font=('Arial',20),padx=100,pady=20,command=lambda : [deleteFoodItem(root)])
	root8.state('zoomed')
	exit_btn=Button(root8,text="Back",command=root8.destroy)
	del_btn.grid(row=rno,column=1,rowspan=3,columnspan=2,sticky=S)
	exit_btn.grid(row=rno+3,column=1,columnspan=2)
	conn.commit()
	conn.close()

def view_details(s):
	if(s!=rras_code):
		popup()
	else:
		root4=Toplevel()
		root4.title(main+"view_details")
		root4["bg"]=bgcolor
		conn=sqlite3.connect('Restaurant_food_data.db')
		c=conn.cursor()
		c.execute("SELECT *,oid from item")
		records=c.fetchall()
		foods=[list(record)[0] for record in records]
		min_pos_rate=90.0
		for i in range(len(records)):
			rec_list=list(records[i])
			fooditem=foods[i]
			cust=int(rec_list[1])
			c.execute("SELECT Cust_limit from limitt where Food_item=:itemname",
					{
						'itemname': fooditem
					}
				)
			cust_lim=c.fetchall()
			cust_lim=int(list(cust_lim[0])[0])
			if(cust>=cust_lim):
				min_pos_rate=min(min_pos_rate,float(rec_list[-3][:-1]))
		dangz_foods=[]
		for i in range(len(records)):
			rec_list=list(records[i])
			fooditem=foods[i]
			cust=int(rec_list[1])
			c.execute("SELECT Cust_limit from limitt where Food_item=:itemname",
					{
						'itemname': fooditem
					}
				)
			cust_lim=c.fetchall()
			cust_lim=int(list(cust_lim[0])[0])
			if(cust>=cust_lim):
				if(min_pos_rate==float(rec_list[-3][:-1])):
					dangz_foods.append(Label(root4,text=fooditem,bg=bgcolor))
		label=Label(root4,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
		sug=Label(root4,text="Items in Danger Zone....",bg=bgcolor)
		acc_btn=Button(root4,text="View Data",font=('Arial',15),padx=50,pady=10,command=access_data)
		additem_btn=Button(root4,text="Add Food Item",font=('Arial',15),padx=50,pady=10,command=askToAdd)
		delitem_btn=Button(root4,text="Delete Food Item",font=('Arial',15),padx=50,pady=10,command=lambda : [askToDel(root4)])
		itemclr_btn=Button(root4,text="Clear Item Data",font=('Arial',15),padx=50,pady=10,command=clr_itemdata)
		allclr_btn=Button(root4,text="Clear All Data",font=('Arial',15),padx=50,pady=10,command=lambda : [clr_alldata(root4)])
		custlim_btn=Button(root4,text="Set Customer Limit",font=('Arial',15),padx=50,pady=10,command=setCustLimit)
		exit_btn=Button(root4,text="LogOut",command=root4.destroy)
		root4.state('zoomed')
		acc_btn.config(height=1,width=6)
		additem_btn.config(height=1,width=6)
		delitem_btn.config(height=1,width=6)
		itemclr_btn.config(height=1,width=6)
		allclr_btn.config(height=1,width=6)
		custlim_btn.config(height=1,width=6)
		label.grid(row=0,column=0,columnspan=6,sticky=W+E)

		acc_btn.grid(row=1,column=0)
		additem_btn.grid(row=1,column=1)
		delitem_btn.grid(row=1,column=2)
		itemclr_btn.grid(row=1,column=3)
		allclr_btn.grid(row=1,column=4)
		custlim_btn.grid(row=1,column=5)
				
		i=3
		sug.grid(row=i,column=0,columnspan=6,sticky=W+E)
		sug.config(font=("Helvetica",20))
		i+=1
		for item in dangz_foods:
			item.grid(row=i,column=0,columnspan=6)
			item.config(font=("Helvetica",20))
			i+=1
		
		
		c.execute("SELECT *,oid from reviewdata")
		records=c.fetchall()
		reviews=[list(record)[0] for record in records]
		for k in range(len(reviews)):
			review = re.sub('[^a-zA-Z]', ' ', reviews[k])
			review = review.lower()
			review = review.split()
			ps = PorterStemmer()
			all_stopwords = stopwords.words('english')
			all_stopwords.remove('not')
			review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
			review = ' '.join(review)
			#reviews[k]=review
			up_cost={}
			summarized_cost=0
			for wrd in review:
				c.execute("select Count from freqdata where Word=:wwrd",
					{
						'wwrd': wrd
					}
				)
				cnt=list(c.fetchall()[0])[0]
				if wrd not in up_cost:
					up_cost[wrd]=int(cnt)
					summarized_cost+=int(cnt)
			c.execute("""UPDATE reviewdata SET Freq_cost=:cost where Item_review=:rev""",
				{
					'cost': str(summarized_cost),
					'rev': reviews[k]
				}
			)

		c.execute("select Item_review,Freq_cost from reviewdata where Status=:stat",
				{
					'stat': "1"
				}
			)
		pos_records=c.fetchall()
		all_pos_reviews=[list(record) for record in pos_records]
		for record in all_pos_reviews:
			record[1]=int(record[1])

		c.execute("select Item_review,Freq_cost from reviewdata where Status=:stat",
				{
					'stat': "0"
				}
			)
		neg_records=c.fetchall()
		all_neg_reviews=[list(record) for record in neg_records]
		for record in all_neg_reviews:
			record[1]=int(record[1])
		
		plen,nlen=len(all_pos_reviews), len(all_neg_reviews)
		if(plen>=5 and nlen>=5):
			relevant_pos_reviews=sorted(all_pos_reviews,key= lambda rec : rec[1])[::-1]
			relevant_neg_reviews=sorted(all_neg_reviews,key= lambda rec : rec[1])[::-1]
			top_pos_reviews=[k[0] for k in relevant_pos_reviews]
			top_neg_reviews=[k[0] for k in relevant_neg_reviews]
			phead=Label(root4,text="TOP RELEVANT POSITIVE REVIEWS",font=('Arial',20,'bold','underline'),bg=bgcolor)
			nhead=Label(root4,text="TOP RELEVANT NEGATIVE REVIEWS",font=('Arial',20,'bold','underline'),bg=bgcolor)
			phead.grid(row=i,column=0,columnspan=3)
			nhead.grid(row=i,column=3,columnspan=3)
			i+=2
			for k in range(5):
				pos_review,neg_review=top_pos_reviews[k],top_neg_reviews[k]
				plabel=Label(root4,text=pos_review,font=("Helevetica",10,"bold"),bg=bgcolor)
				nlabel=Label(root4,text=neg_review,font=("Helevetica",10,"bold"),bg=bgcolor)
				plabel.grid(row=i,column=0,rowspan=2,columnspan=3)
				nlabel.grid(row=i,column=3,rowspan=2,columnspan=3)
				i+=2
		
		exit_btn.grid(row=i,column=0,columnspan=6,sticky=S)
				
		conn.commit()
		conn.close()
				
def take_review():
	root2=Toplevel()
	root2.title(main+"give review")
	root2["bg"]=bgcolor
	label=Label(root2,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
	req1=Label(root2,text="Select the item(s) you have taken.....",bg=bgcolor)
	'''foods=["Idly","Dosa","Vada","Roti","Meals","Veg Biryani","Egg Biryani","Chicken Biryani","Mutton Biryani","Ice Cream","Noodles","Manchooriya","Orange juice","Apple Juice","Pineapple juice","Banana juice"]
	for i in foods:
		food_rev[i]=[]
		food_perc[i]=[0.0,0.0]'''

	conn=sqlite3.connect('Restaurant_food_data.db')
	c=conn.cursor()
	
	chk_list=[]
	selected_foods=[]
	req2=Label(root2,text="Give your review below....",bg=bgcolor)
	rev_tf=Entry(root2,width=125,borderwidth=5)
	req3=Label(root2,text="NOTE : Use not instead of n't.",bg=bgcolor)
	c.execute("SELECT *,oid FROM item")
	records=c.fetchall()
	foods=[list(record)[0] for record in records]
	global variables
	variables=[]
	#for req in range(len(list(reqtf.items()))):
	chk_btns=[]
	#for i in range(len(list(reqtf.items()))):
	#	lab1=Label(root2,text="Add Food Item", bd=2,font=('Arial',47,'bold'))
	#	lab2=Label(root2,text="Delete Food Item", bd=2, font=('Arial',42,'bold')
	for i in range(len(foods)):  
		var=IntVar()
		chk=Checkbutton(root2,text=foods[i],variable=var,bg=bgcolor)  
		variables.append(var) 
		chk_list.append(chk)
	
	label.grid(row=0,column=0,columnspan=4)
	req1.grid(row=1,column=0,columnspan=4,sticky=W+E)
	req1.config(font=("Helvetica",30))

	n=len(foods)
	for i in range(n//4):
		for j in range(4):
			c=chk_list[i*4+j]
			c.grid(row=i+3,column=j,columnspan=1,sticky=W)
	j=0
	for i in range((n//4)*4,n):
		c=chk_list[i]
		c.grid(row=(n//4)+3,column=j,columnspan=1,sticky=W)
		j+=1
	rno=(n//4)+5
	selected_foods=[]
	submit_review=Button(root2,text="Submit Review",font=('Arial',20),padx=100,pady=20,command=lambda : [estimate(rev_tf.get()),root2.destroy()])
	root2.state('zoomed')
	req2.grid(row=rno,column=0,columnspan=4,sticky=W+E)
	req2.config(font=("Helvetica",20))
	rev_tf.grid(row=rno+1,column=1,rowspan=3,columnspan=2,sticky=S)
	req3.grid(row=rno+4,column=1,columnspan=2)
	submit_review.grid(row=rno+5,column=0,columnspan=4)
	conn.commit()
	conn.close()

def login():
	root3=Toplevel()
	root3.title(main+"owner verfication")
	root3["bg"]=bgcolor
	label=Label(root3,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
	label2=Label(root3,text="VERIFY OWNERSHIP",bd=1,font=('Helvetica',30,'bold','underline'),bg=bgcolor)
	label3=Label(root3,text="To verify your ownership, please enter your restaurant's private rras passcode....",bd=1,font=('Helvetica',20,'bold'),bg=bgcolor)
	ent=Entry(root3,show="*",borderwidth=2)
	submit_code=Button(root3,text="Submit",font=('Arial',20),padx=80,pady=20,command=lambda : [view_details(ent.get()),root3.destroy()])
	root3.state('zoomed')
	label.grid(row=0,column=0,columnspan=3)
	label2.grid(row=1,column=0,sticky=W+E,columnspan=3)
	label3.grid(row=2,column=0,sticky=W,columnspan=3)
	ent.grid(row=3,column=1,columnspan=1)
	submit_code.grid(row=5,column=1,columnspan=1)

#init_data()

label=Label(root1,text="RESTAURANT REVIEW ANALYSIS SYSTEM",bd=2,font=('Arial',47,'bold','underline'),bg="black",fg="white")
ques=Label(root1,text="Are you a Customer or Owner ???",bg=bgcolor)
cust=Button(root1,text="Customer",font=('Arial',20),padx=80,pady=20,command=take_review)
owner=Button(root1,text="Owner",font=('Arial',20),padx=100,pady=20,command=login)

'''conn=sqlite3.connect('Restaurant_food_data.db')
c=conn.cursor()
c.execute("CREATE TABLE item (Item_name text,No_of_customers text,No_of_positive_reviews text,No_of_negative_reviews text,Positive_percentage text,Negative_percentage text) ")
c.execute("CREATE TABLE limitt (Food_item text, Cust_limit text)")
conn.commit()
conn.close()'''

#init_limitData()
#c.execute("DELETE FROM item")
root1.attributes("-fullscreen",True)
#root1.state('zoomed')
label.grid(row=0,column=0)
ques.grid(row=1,column=0,sticky=W+E)
ques.config(font=("Helvetica",30))
cust.grid(row=2,column=0)
owner.grid(row=3,column=0)
conn.commit()
conn.close()
root1.mainloop()