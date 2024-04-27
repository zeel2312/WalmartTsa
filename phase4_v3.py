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
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
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
    df = pd.read_csv("./walmart-sales-dataset-of-45stores.csv")

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
if st.sidebar.checkbox("View Statistics", key='stats_checkbox_2'):
    selected_stat_2 = st.selectbox("Select Statistics for Checkbox 2", ["Outlier Boxplots", "Shape", "Description", "Null Values"])

# Show selected statistics if checkbox 2 is selected
if st.sidebar.checkbox("Show Chosen Statistics", key='show_stats_checkbox_2'):
    st.write("There is a known memory issue. If the web page crashes, please reload and return to this feature without observing the other features first.")
    if selected_stat_2 == "Outlier Boxplots":
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
if st.sidebar.checkbox("Show Distribution"):
    st.write("There is a known memory issue. If the web page crashes, please reload and return to this feature without observing the other features first.")
    if selected_stat_3 == "Distribution of Data and Skewness Score of every column":
        st.write("Distribution of Data for Every Column")
        # Creating a new DataFrame with only the numeric columns leaving categorical ones out
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        # Plotting the histograms for each numeric column
        for col in numeric_df:
            plt.figure(figsize=(10, 4))
            sns.histplot(numeric_df[col], kde=True)
            plt.title(f"Distribution of {col}")
            st.pyplot(plt)

            skewness = numeric_df[col].skew()
            st.write(" ")
            st.write(f"Skewness of {col}: {skewness}")


# Add dropdown menu for selecting statistics in checkbox 4
if st.sidebar.checkbox("Comparison Graphs", key='graphs_for_comparison'):
    selected_stat_4 = st.selectbox("Graphs for Comparison", ["Fuel_Price vs CPI", "Change CPI per Week", "CPI vs Unemployment over Time", "Average Weekly Sales by Store", "Average Weekly Sales by Store over Time", "Weekly_sales vs Unemployement in stores", "Weekly_sales vs Fuel_Price", "Weekly_sales vs Temperature", "Weekly_sales vs holiday_flag", "Pearson Correlation Coefficient"])

# Show selected statistics if checkbox 4 is selected
if st.sidebar.checkbox("Show Comparison Graph"):
    st.write("There is a known memory issue. If the web page crashes, please reload and return to this feature without observing the other features first.")
    if selected_stat_4 == "Fuel_Price vs CPI":
        st.write("Graph for Fuel_Price vs CPI")
        fig = px.scatter(df, x='CPI', y='Fuel_Price', title='Fuel_Price vs CPI')
        st.plotly_chart(fig)

    elif selected_stat_4 == "Change CPI per Week":
        st.write("Graph for Change CPI per Week")
        df_3 = df.groupby('Date')[['CPI']].sum()
        fig = px.line(df_3, x=df_3.index, y=df_3['CPI'], title='change CPI per week')
        st.plotly_chart(fig)

    elif selected_stat_4 == "CPI vs Unemployment over Time":
        st.write("Graph for CPI vs Unemployment over Time")
        fig = px.scatter(df, x='CPI', y='Unemployment', title=' CPI vs Unemployment over Time')
        st.plotly_chart(fig)

    elif selected_stat_4 == "Average Weekly Sales by Store":
        st.write("Graph for Average Weekly Sales by Store") 
        df_grouped = df.groupby(['Date', 'Store'])['Weekly_Sales'].mean().unstack().reset_index()
        fig = px.line(df_grouped, x='Date', y=df_grouped.columns[1:], title='Average Weekly Sales by Store over Time')
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Weekly Sales')
        st.plotly_chart(fig)

    elif selected_stat_4 == "Average Weekly Sales by Store over Time":
        st.write("Graph for Average Weekly Sales by Store over Time")
        fig = px.line(df.groupby('Date')['Weekly_Sales'].mean(), title='Average Weekly Sales over Time')
        fig.update_layout(xaxis_title='Date', yaxis_title='Average Weekly Sales')
        st.plotly_chart(fig)

    elif selected_stat_4 == "Weekly_sales vs Unemployement in stores":
        st.write("Graph for Weekly_sales vs Unemployement in stores")
        plt.figure(figsize = (20,5))
        fig = px.scatter(df, x="Weekly_Sales", y='Unemployment', color="Store", title="Relation between Unemployment and weeklysales within stores" , color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    elif selected_stat_4 == "Weekly_sales vs Fuel_Price":
        st.write("Graph for Weekly_sales vs Fuel_Price")
        fig = px.scatter(df, x="Fuel_Price", y="Weekly_Sales", title="Weekly Sales vs Fuel Price")
        st.plotly_chart(fig)

    elif selected_stat_4 == "Weekly_sales vs Temperature":
        st.write("Graph for Weekly_sales vs Temperature")
        plt.figure(figsize = (20,5))
        fig = px.scatter(df, x="Weekly_Sales", y="Temperature", color="Store", title="Relation between Temperature and weeklysales within stores")
        st.plotly_chart(fig)

    elif selected_stat_4 == "Weekly_sales vs holiday_flag":
        st.write("Graph for Weekly_sales vs holiday_flag")
        plt.figure(figsize = (20,5))
        fig = px.strip(df, x="Weekly_Sales", y="Holiday_Flag", orientation="h", color="Store" , title = 'relation between weekly sales and holiday_flag')
        st.plotly_chart(fig)

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

        st.plotly_chart(fig)

# Add dropdown menu for decompositions
if st.sidebar.checkbox("Choose Decomposition", key="decomposition"):
    selected_decomposition = st.selectbox("Select Decomposition", ["Additive", "Multiplicative"])

# Show decompositions
if st.sidebar.checkbox("Show Decomposition"):
    st.write("There is a known memory issue. If the web page crashes, please reload and return to this feature without observing the other features first.")
    if selected_decomposition == "Additive":
        st.write("Additive Decomposition Plots")
        additive_decomposition = seasonal_decompose(df['Weekly_Sales'], model='additive', period=300)
        plt.rcParams.update({'figure.figsize': (12,10)})
        additive_decomposition.plot()
        st.pyplot(plt)
    
    elif selected_decomposition == "Multiplicative":
        st.write("Multiplicative Decomposition Plots")
        multiplicative_decomposition = seasonal_decompose(df['Weekly_Sales'], model='multiplicative', period=300)
        plt.rcParams.update({'figure.figsize': (12,10)})
        multiplicative_decomposition.plot()
        st.pyplot(plt)

# Add dropdown menu for selecting LSTM changes
# if st.sidebar.checkbox("Choose LSTM Predictions", key='compare_lstm_changes'):
#     selected_lstm_change = st.selectbox("Select Data Changes for LSTM", ["None", "Increase Temperature", "Decrease Temperature", "Increase Fuel Price", "Decrease Fuel Price", "Increase CPI", "Decrease CPI", "Increase Unemployment", "Decrease Unemployment"])

# Show LSTM
if st.sidebar.checkbox("Show LSTM"):
    st.write("There is a known memory issue. If the web page crashes, please reload and return to this feature without observing the other features first.")
    st.subheader("LSTM Model Predictions")
    st.write("Please allow up to 10 minutes for predictions.")
    df_lstm = df.drop(["Store"], axis = 1)
    df_lstm.sort_values("Date", inplace = True)
    def predictions(df):
        x = df.drop(["Weekly_Sales"], axis = 1)
        y = df[["Date", "Weekly_Sales"]]
        tss = TimeSeriesSplit(n_splits = 3)
        for train_index, test_index in tss.split(x):
            x_train, x_test = x.iloc[train_index, :], x.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train = x_train.drop(["Date"], axis = 1)
        x_test = x_test.drop(["Date"], axis = 1)
        y_pres_train = y_train # preserve y_train data
        y_train = y_train.drop(["Date"], axis = 1)
        y_pres_test = y_test # preserve y_test data
        y_test = y_test.drop(["Date"], axis = 1)

        x_train = np.asarray(x_train).astype('float32')
        x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))
        x_test = np.asarray(x_test).astype('float32')
        x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))

        n_features = 5
        n_steps = 1
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(len(x_train), input_shape = (x_train.shape[1],), activation='relu'))
        opt = keras.optimizers.AdamW(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mse')

        model.fit(x_train, y_train, epochs=15, verbose=0)

        y_pred = model.predict(x_test, verbose=0)
        y_pred = y_pred.astype("float64")
        pd.options.display.float_format = '{:.0f}'.format
        df_pred = pd.DataFrame(y_pred)
        df_pred = df_pred.iloc[:, 2]
        df_pred = df_pred.to_frame()
        df_pred = df_pred.rename(columns={2:"Weekly_Sales"})

        y_pres_test.reset_index(inplace = True)
        df_pred["Date"] = y_pres_test["Date"]
        df_pred_sum = df_pred.groupby("Date")[["Weekly_Sales"]].sum()
        y_pres_test_sum = y_pres_test.groupby("Date")[["Weekly_Sales"]].sum()
        y_pres_train_sum = y_pres_train.groupby("Date")[["Weekly_Sales"]].sum()

        # make variables for melting
        train_new = y_pres_train_sum['Weekly_Sales']
        actual_new = y_pres_test_sum['Weekly_Sales']
        pred_new = df_pred_sum['Weekly_Sales']

        train_new_index = train_new.index
        test_new_index = actual_new.index
        train_new_index_series = train_new_index.to_series()
        test_new_index_series = test_new_index.to_series()
        new_index = pd.concat([train_new_index_series, test_new_index_series])

        return train_new, actual_new, pred_new, new_index
    
    train_new, actual_new, pred_new, new_index = predictions(df_lstm)
    plot_data = pd.DataFrame({
        'training': train_new,
        'actual': actual_new,
        'forecast': pred_new,
        'index': new_index
        })
    plot_data_melted = plot_data.melt(id_vars='index', var_name='Type', value_name='Value')
    fig = px.line(plot_data_melted, x='index', y='Value', color='Type', title='Forecast vs Actuals',
                    color_discrete_map={
                        'training': 'blue',
                        'actual': 'green',
                        'forecast': 'red',
                        })
    st.plotly_chart(fig)
    
    # def temp_change(change, df):
    #     if change == "increase":
    #         df_temp_inc = df.copy(deep=True)
    #         df_temp_inc_work = df_temp_inc.sample(frac = 0.5)
    #         temp_inc = df_temp_inc_work["Temperature"] + 10
    #         df_temp_inc_work["Temperature"] = temp_inc
    #         df_temp_inc.update(df_temp_inc_work)
    #         return df_temp_inc
    #     elif change == "decrease":
    #         df_temp_dec = df.copy(deep=True)
    #         df_temp_dec_work = df_temp_dec.sample(frac = 0.5)
    #         temp_dec = df_temp_dec_work["Temperature"] - 10
    #         df_temp_dec_work["Temperature"] = temp_dec
    #         df_temp_dec.update(df_temp_dec_work)
    #         return df_temp_dec
        
    # def fuel_change(change, df):
    #     if change == "increase":
    #         df_fuel_inc = df.copy(deep=True)
    #         df_fuel_inc_work = df_fuel_inc.sample(frac = 0.5)
    #         fuel_inc = df_fuel_inc_work["Fuel_Price"] + 1
    #         df_fuel_inc_work["Fuel_Price"] = fuel_inc
    #         df_fuel_inc.update(df_fuel_inc_work)
    #         return df_fuel_inc
    #     elif change == "decrease":
    #         df_fuel_dec = df.copy(deep=True)
    #         df_fuel_dec_work = df_fuel_dec.sample(frac = 0.5)
    #         fuel_dec = df_fuel_dec_work["Fuel_Price"] - 1
    #         df_fuel_dec_work["Fuel_Price"] = fuel_dec
    #         df_fuel_dec.update(df_fuel_dec_work)
    #         return df_fuel_dec

    # def cpi_change(change, df):
    #     if change == "increase":
    #         df_cpi_inc = df.copy(deep=True)
    #         df_cpi_inc_work = df_cpi_inc.sample(frac = 0.5)
    #         cpi_inc = df_cpi_inc_work["CPI"] + 25
    #         df_cpi_inc_work["CPI"] = cpi_inc
    #         df_cpi_inc.update(df_cpi_inc_work)
    #         return df_cpi_inc
    #     elif change == "decrease":
    #         df_cpi_dec = df.copy(deep=True)
    #         df_cpi_dec_work = df_cpi_dec.sample(frac = 0.5)
    #         cpi_dec = df_cpi_dec_work["CPI"] - 25
    #         df_cpi_dec_work["CPI"] = cpi_dec
    #         df_cpi_dec.update(df_cpi_dec_work)
    #         return df_cpi_dec
    
    # def unemployment_change(change, df):
    #     if change == "increase":
    #         df_unemp_inc = df.copy(deep=True)
    #         df_unemp_inc_work = df_unemp_inc.sample(frac = 0.5)
    #         unemp_inc = df_unemp_inc_work["Unemployment"] + 1
    #         df_unemp_inc_work["Unemployment"] = unemp_inc
    #         df_unemp_inc.update(df_unemp_inc_work)
    #         return df_unemp_inc
    #     elif change == "decrease":
    #         df_unemp_dec = df.copy(deep=True)
    #         df_unemp_dec_work = df_unemp_dec.sample(frac = 0.5)
    #         unemp_dec = df_unemp_dec_work["Unemployment"] - 1
    #         df_unemp_dec_work["Unemployment"] = unemp_dec
    #         df_unemp_dec.update(df_unemp_dec_work)
    #         return df_unemp_dec
        
    # def change_predictions(df, change_type, change):
    #     if change_type == "Temperature":
    #         if change == "increase":
    #             change_df = temp_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             temp_pred = pred_new_change
    #             return temp_pred
    #         elif change == "decrease":
    #             change_df = temp_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             temp_pred = pred_new_change
    #             return temp_pred
            
    #     elif change_type == "Fuel Price":
    #         if change == "increase":
    #             change_df = fuel_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             fuel_pred = pred_new_change
    #             return fuel_pred
    #         elif change == "decrease":
    #             change_df = fuel_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             fuel_pred = pred_new_change
    #             return fuel_pred
            
    #     elif change_type == "CPI":
    #         if change == "increase":
    #             change_df = cpi_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             cpi_pred = pred_new_change
    #             return cpi_pred
    #         elif change == "decrease":
    #             change_df = cpi_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             cpi_pred = pred_new_change
    #             return cpi_pred
            
    #     elif change_type == "Unemployment":
    #         if change == "increase":
    #             change_df = unemployment_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             unemp_pred = pred_new_change
    #             return unemp_pred
    #         elif change == "decrease":
    #             change_df = unemployment_change(change, df)
    #             train_new_change, actual_new_change, pred_new_change, new_index_change = predictions(change_df)
    #             unemp_pred = pred_new_change
    #             return unemp_pred
            
    # def changes_plot(df, change, direction):
    #     train_new, actual_new, pred_new, new_index = predictions(df)
    #     change_pred = change_predictions(df, change, direction)
    #     plot_data = pd.DataFrame({
    #         'training': train_new,
    #         'actual': actual_new,
    #         'forecast': pred_new,
    #         'changed forecast': change_pred,
    #         'index': new_index
    #         })
    #     plot_data_melted = plot_data.melt(id_vars='index', var_name='Type', value_name='Value')
    #     fig = px.line(plot_data_melted, x='index', y='Value', color='Type', title='Forecast vs Actuals',
    #                   color_discrete_map={
    #                       'training': 'blue',
    #                       'actual': 'green',
    #                       'forecast': 'red',
    #                       'changed forecast': 'orange'
    #                       })
    #     st.plotly_chart(fig)
            
    # if selected_lstm_change == "None":
    #     train_new, actual_new, pred_new, new_index = predictions(df_lstm)
    #     plot_data = pd.DataFrame({
    #         'training': train_new,
    #         'actual': actual_new,
    #         'forecast': pred_new,
    #         'index': new_index
    #         })
    #     plot_data_melted = plot_data.melt(id_vars='index', var_name='Type', value_name='Value')
    #     fig = px.line(plot_data_melted, x='index', y='Value', color='Type', title='Forecast vs Actuals',
    #                   color_discrete_map={
    #                       'training': 'blue',
    #                       'actual': 'green',
    #                       'forecast': 'red',
    #                       })
    #     st.plotly_chart(fig)

    # elif selected_lstm_change == "Increase Temperature":
    #     changes_plot(df_lstm, "Temperature", "increase")

    # elif selected_lstm_change == "Decrease Temperature":
    #     changes_plot(df_lstm, "Temperature", "decrease")

    # elif selected_lstm_change == "Increase Fuel Price":
    #     changes_plot(df_lstm, "Fuel Price", "increase")

    # elif selected_lstm_change == "Decrease Fuel Price":
    #     changes_plot(df_lstm, "Fuel Price", "decrease")

    # elif selected_lstm_change == "Increase CPI":
    #     changes_plot(df_lstm, "CPI", "increase")

    # elif selected_lstm_change == "Decrease CPI":
    #     changes_plot(df_lstm, "CPI", "decrease")

    # elif selected_lstm_change == "Increase Unemployment":
    #     changes_plot(df_lstm, "Unemployment", "increase")

    # elif selected_lstm_change == "Decrease Unemployment":
    #     changes_plot(df_lstm, "Unemployment", "decrease")
