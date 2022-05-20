
import sqlite3

conn=sqlite3.connect('Restaurant_food_data.db')
c=conn.cursor()

#c.execute("CREATE TABLE item (Item_name text,No_of_customers text,No_of_positive_reviews text,No_of_negative_reviews text,Positive_percentage text,Negative_percentage text) ")

#c.execute("CREATE TABLE limitt (Food_item text, Cust_limit text)")

#c.execute("CREATE TABLE reviewdata (Item_review text, Status text, Freq_cost text)")

#c.execute("CREATE TABLE freqdata (Word text, Count text)")


conn.commit()
conn.close()



