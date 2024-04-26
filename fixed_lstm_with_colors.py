# Download all libraries in the terminal.

import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.express as px
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Walmart 45 Stores", page_icon=":bar_chart:",layout="wide")

st.title(" :bar_chart: Walmart 45 Store Sales prediction")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)


print("Current working directory:", os.getcwd())
print("List of files in directory:", os.listdir())

# Attempt to read the file with flexible date parsing
try:
    df = pd.read_csv("walmart-sales-dataset-of-45stores.csv", parse_dates=['Date'], infer_datetime_format=True)
    print("Data loaded successfully with inferred date format.")
except FileNotFoundError as e:
    print("File not found error:", e)
except Exception as e:
    print("An error occurred:", e)


fl = st.file_uploader(":file_folder: Upload a file",type=(["csv","txt","xlsx","xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename)
else:
    os.chdir(r"/Users/zeel/Downloads/SP24-DSCI-D590 Time Series Analysis/Project")
    df = pd.read_csv("walmart-sales-dataset-of-45stores.csv")

col1, col2 = st.columns((2))
df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')

# Getting the min and max date 
startDate = pd.to_datetime(df["Date"]).min()
endDate = pd.to_datetime(df["Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Date"] >= date1) & (df["Date"] <= date2)].copy()

# Sidebar
st.sidebar.title("Dashboard Menu")

# Add dropdown menu for viewing dataset in checkbox 1

if st.sidebar.checkbox("Dataset"):
    # Display the dataset
    st.subheader("Walmart 45 Store Dataset")
    st.write(df)


# Add dropdown menu for selecting statistics in checkbox 2
if st.sidebar.checkbox("Checkbox 2", key='stats_checkbox_2'):
    selected_stat_2 = st.selectbox("Select Statistics for Checkbox 2", ["Skewness", "Outlier Boxplots", "Shape", "Description", "Null Values"])

# Show selected statistics if checkbox 2 is selected
if st.sidebar.checkbox("Show Checkbox 2 Stats", key='show_stats_checkbox_2'):
    if selected_stat_2 == "Skewness":
        st.write("Skewness of the Dataset")
        st.write(df.skew())

    elif selected_stat_2 == "Outlier Boxplots":
        st.write("Outlier Boxplots for Every Column")
        for col in df.columns:
            st.write(f"Column: {col}")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)

    elif selected_stat_2 == "Shape":
        st.write("Shape of the Dataset")
        st.write(df.shape)

    elif selected_stat_2 == "Description":
        st.write("Description of the Dataset")
        st.write(df.describe())

    elif selected_stat_2 == "Null Values":
        st.write("Null Values in the Dataset")
        st.write(df.isnull().sum())

# Add dropdown menu for selecting statistics in checkbox 3
if st.sidebar.checkbox("Sales Distribution by Store", key='sales_distribution_checkbox'):
    selected_stat_3 = st.selectbox("Select Distribution Data", ["Distribution of Data and Skewness Score of every column"])

# Show selected statistics if checkbox 3 is selected
if st.sidebar.checkbox("Checkbox 3"):
    if selected_stat_3 == "Distribution of Data":
        st.write("Distribution of Data for Every Column")
        # Creating a new DataFrame with only the numeric columns leaving categorical ones out
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        # Plotting the histograms for each numeric column
        for col in numeric_df:
            plt.figure(figsize=(10, 4))
            sns.histplot(numeric_df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

            skewness = numeric_df[col].skew()
            print(" ")
            print(f"Skewness of {col}: {skewness}")

# Add dropdown menu for selecting statistics in checkbox 4
if st.sidebar.checkbox("Checkbox 4", key='graphs_for_comparison'):
    selected_stat_4 = st.selectbox("Graphs for Comparison", ["Fuel_Price vs CPI", "Change CPI per Week", "CPI vs Unemployment over Time", "Average Weekly Sales by Store", "Average Weekly Sales by Store over Time", "Weekly_sales vs Unemployement in stores", "Weekly_sales vs Fuel_Price", "Weekly_sales vs Temperature", "Weekly_sales vs holiday_flag", "Pearson Correlation Coefficient"])

# Show selected statistics if checkbox 4 is selected
if st.sidebar.checkbox("Show Checkbox 4"):
    if selected_stat_4 == "Fuel_Price vs CPI":
        st.write("Graph for Fuel_Price vs CPI")
        fig = px.scatter(df, x='CPI', y='Fuel_Price', title='Fuel_Price vs CPI')
        fig.show()

    elif selected_stat_4 == "Change CPI per Week":
        st.write("Graph for Change CPI per Week")
        df_3 = df.groupby('Date')[['CPI']].sum()
        fig = px.line(df_3, x=df_3.index, y=df_3['CPI'], title='change CPI per week')
        fig.show()

    elif selected_stat_4 == "CPI vs Unemployment over Time":
        st.write("Graph for CPI vs Unemployment over Time")
        fig = px.scatter(df, x='CPI', y='Unemployement', title=' CPI vs Unemployment over Time')
        fig.show()

    elif selected_stat_4 == "Average Weekly Sales by Store":
        st.write("Graph for Average Weekly Sales by Store") 
        df_grouped = df.groupby(['Date', 'Store'])['Weekly_Sales'].mean().unstack().reset_index()
        fig = px.line(df_grouped, x='Date', y=df_grouped.columns[1:], title='Average Weekly Sales by Store over Time')
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Weekly Sales')
        fig.show()

    elif selected_stat_4 == "Average Weekly Sales by Store over Time":
        st.write("Graph for Average Weekly Sales by Store over Time")
        fig = px.line(df.groupby('Date')['Weekly_Sales'].mean(), title='Average Weekly Sales over Time')
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Weekly Sales')
        fig.show()

    elif selected_stat_4 == "Weekly_sales vs Unemployement in stores":
        st.write("Graph for Weekly_sales vs Unemployement in stores")
        plt.figure(figsize = (20,5))
        fig = px.scatter(df, x="Weekly_Sales", y='Unemployment', color="Store", title="Relation between Unemployment and weeklysales within stores" , color_continuous_scale=px.colors.sequential.Viridis)
        fig.show()

    elif selected_stat_4 == "Weekly_sales vs Fuel_Price":
        st.write("Graph for Weekly_sales vs Fuel_Price")
        fig = px.scatter(df, x="Fuel_Price", y="Weekly_Sales", title="Weekly Sales vs Fuel Price")
        fig.show()

    elif selected_stat_4 == "Weekly_sales vs Temperature":
        st.write("Graph for Weekly_sales vs Temperature")
        plt.figure(figsize = (20,5))
        fig = px.scatter(df, x="Weekly_Sales", y="Temperature", color="Store", title="Relation between Temperature and weeklysales within stores")
        fig.show()

    elif selected_stat_4 == "Weekly_sales vs holiday_flag":
        st.write("Graph for Weekly_sales vs holiday_flag")
        plt.figure(figsize = (20,5))
        fig = px.strip(df, x="Weekly_Sales", y="Holiday_Flag", orientation="h", color="Store" , title = 'relation between weekly sales and holiday_flag')
        fig.show()

    elif selected_stat_4 == "Pearson Correlation Coefficient":
        st.write("Graph for Pearson Correlation Coefficient")
        # Calculate the Pearson correlation coefficient between all columns
        correlation_matrix = df.corr(method='pearson')

        # Convert correlation matrix to tidy format
        correlation_tidy = correlation_matrix.stack().reset_index()
        correlation_tidy.columns = ['Column1', 'Column2', 'Correlation']

        # Create interactive heatmap using Plotly Express
        fig = px.imshow(correlation_matrix, labels=dict(color='Pearson Correlation'),
                x=correlation_matrix.index,
                y=correlation_matrix.columns,
                title='Pearson Correlation Coefficients Between Columns',
                color_continuous_scale='RdBu_r')

        fig.show()

# LSTM Model Display
st.subheader("LSTM Model Predictions")
# Put data in traditional x-y format for ML
df_lstm = df.drop(["Store"], axis = 1)
df_lstm.sort_values("Date", inplace = True)
#df_lstm.set_index("Date", inplace = True)
x = df_lstm.drop(["Weekly_Sales"], axis = 1)
y = df_lstm[["Date", "Weekly_Sales"]]

# Resource used for TimeSeriesSplit
# https://medium.com/@Stan_DS/timeseries-split-with-sklearn-tips-8162c83612b9

# split data into train-test
tss = TimeSeriesSplit(n_splits = 3)
for train_index, test_index in tss.split(x):
    x_train, x_test = x.iloc[train_index, :], x.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
       
# the model itself doesn't like the Date feature
# but copies of the train-test splits will be kept for Date uses later
x_train = x_train.drop(["Date"], axis = 1)
x_test = x_test.drop(["Date"], axis = 1)
y_pres_train = y_train # preserve y_train data
y_train = y_train.drop(["Date"], axis = 1)
y_pres_test = y_test # preserve y_test data
y_test = y_test.drop(["Date"], axis = 1)

# reshape the data first
x_train = np.asarray(x_train).astype('float32')
x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
x_test = np.asarray(x_test).astype('float32')
x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))

# create the model
n_features = 5
n_steps = 1
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(len(x_train), input_shape = (x_train.shape[1],), activation='relu'))
opt = keras.optimizers.AdamW(learning_rate=0.01)
model.compile(optimizer=opt, loss='mse')

# fit the model
# takes 5 minutes to run
model.fit(x_train, y_train, epochs=200, verbose=0)

# make predictions
y_pred = model.predict(x_test, verbose=0)

# put predictions in a format for comparing to test set
y_pred = y_pred.astype("float64")
pd.options.display.float_format = '{:.0f}'.format
df_pred = pd.DataFrame(y_pred)
df_pred = df_pred.iloc[:, 2]
df_pred = df_pred.to_frame()
df_pred = df_pred.rename(columns={2:"Weekly_Sales"})

# create Weekly Sales sums for use in plotting and model evaluation
y_pres_test.reset_index(inplace = True)
df_pred["Date"] = y_pres_test["Date"]
df_pred_sum = df_pred.groupby("Date")[["Weekly_Sales"]].sum()
y_pres_test_sum = y_pres_test.groupby("Date")[["Weekly_Sales"]].sum()
y_pres_train_sum = y_pres_train.groupby("Date")[["Weekly_Sales"]].sum()

# Need to turn things into series to put into a dataframe
train_new = y_pres_train_sum['Weekly_Sales']
actual_new = y_pres_test_sum['Weekly_Sales']
pred_new = df_pred_sum['Weekly_Sales']

# Make new index series for dataframe
train_new_index = train_new.index
test_new_index = actual_new.index
train_new_index_series = train_new_index.to_series()
test_new_index_series = test_new_index.to_series()
new_index = pd.concat([train_new_index_series, test_new_index_series])

# Create DataFrame for Plotly Express
plot_data = pd.DataFrame({
    'training': train_new,
    'actual': actual_new,
    'forecast': pred_new,
    'index': new_index
})

# Melt the DataFrame to have 'variable' and 'value' columns
plot_data_melted = plot_data.melt(id_vars='index', var_name='Type', value_name='Value')

# # Create interactive line plot using Plotly Express
# fig = px.line(plot_data_melted, x='index', y='Value', color='Type', title='Forecast vs Actuals')
        
# fig.show()

# Create interactive line plot using Plotly Express with specified colors
fig = px.line(plot_data_melted, x='index', y='Value', color='Type', title='Forecast vs Actuals',
              color_discrete_map={
                  'training': 'blue',
                  'actual': 'green',  
                  'forecast': 'red'  
              })

fig.show()