import pandas as pd
import pickle
from sklearn import linear_model
import matplotlib.pyplot as plt


# DATA EXPLORATION
df = pd.read_csv('Walmart.csv')
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# select relevant columns for linear regression
x = df[['Temperature', 'Fuel_Price']]
y = df[['Weekly_Sales']]

# creating an object of LinearRegression class
LR = linear_model.LinearRegression()
# fitting the training data
LR.fit(x.values, y.values)


# store model
pickle.dump(LR, open('model.pkl', "wb"))
model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict([[89, 4.30]]))


# ----------------method used to create visual elements-------------------
# def build_figs():
#     plt.clf()
#     plt.bar(df['Temperature'], df['Weekly_Sales'], width=3)
#     plt.xlabel('Average Temperature (Degrees Fahrenheit)')
#     plt.ylabel('Sales Amount USD (Millions) ')
#     plt.title('Sales by Temperature')
#     plt.savefig('plot1.png')
#     plt.clf()
#
#     plt.bar(df['Fuel_Price'], df['Weekly_Sales'], width=0.1)
#     plt.xlabel('Average Fuel Price USD')
#     plt.ylabel('Sales Amount USD (Millions) ')
#     plt.title('Sales by Fuel Price')
#     plt.savefig('plot2.png')
#     plt.clf()
#
#     plt.scatter(df['Fuel_Price'], df['Temperature'])
#     plt.xlabel('Average Fuel Price USD')
#     plt.ylabel('Average Temperature (Degrees Fahrenheit)')
#     plt.title('Fuel Price and Temperature')
#     plt.savefig('plot3.png')
#     plt.clf()
