# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:37:32 2023

@author: HP
"""

import base64
import os
import numpy as np 
import pandas as pd
import streamlit as st
import netCDF4 as nc
import math
from streamlit_folium import st_folium
import folium
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Bidirectional
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


#"""
#*******************TRAINING*******************
#"""

st.markdown(f"""<h1 style='text-align: center; font-weight:bold;font-family:times new roman;color:black;background-color:powderblue;font-size:20pt;'>Know the CO2 level at your area⚠️</h1>""",unsafe_allow_html=True)
m = folium.Map(location=None, width='100%', height='100%', left='0%', top='0%', position='relative', tiles='OpenStreetMap', attr=None, min_zoom=0, max_zoom=18, zoom_start=10, min_lat=- 90, max_lat=90, min_lon=- 180, max_lon=180, max_bounds=True, crs='EPSG3857', control_scale=False, prefer_canvas=False, no_touch=False, disable_3d=False, png_enabled=False, zoom_control=True)
m.add_child(folium.LatLngPopup())
map = st_folium(m)
try:
    user_lat=map['last_clicked']['lat']
    user_lon=map['last_clicked']['lng'] 

except:
    st.warning("No location choosen")



#user_lat=float(user_lat)
#user_lon=float(user_lon)




if st.button("Predict"):
           
    df_all1=pd.DataFrame(columns=['DATE','TRAINING CO2'])
    j=0
    for root, dirs, files in os.walk(r"C:\Users\HP\Downloads\datas\input_data"):
        for file in files:
            #st.write(file)
            if os.path.splitext(file)[1] == '.nc4':
                filePath = os.path.join(root, file)
            ds = nc.Dataset(filePath)
            df=pd.DataFrame(columns=["Latitude","Longitude","xco2"])
    
            df["Longitude"] = ds['longitude'][:]
            df["Latitude"] = ds['latitude'][:]
            df["xco2"]=ds['xco2'][:]
    
            #Repalce inplace 
            df.fillna(0,inplace=True)
    
            #df_first=df[(60>df['Latitude']> 59)]
            df_first=df.loc[(df['Latitude'] >user_lat) &(df['Latitude'] < user_lat+20) & (df['Longitude']> user_lon)&(df['Longitude']< user_lon+20 ),'xco2']
            res=df_first.mean()  ##PARTICULAR DAY MEAN for areas combined
            #st.write(res)
    
    
            df_all1.loc[j,"DATE"] = file[15:17]+"/"+file[13:15]+"/20"+file[11:13]
            df_all1.loc[j,"TRAINING CO2"] = res
    
            j+=1
    st.header("RAW TRAINING DATA")
    st.write(df_all1)
    st.header("PRE-PROCESSED TRAINING DATA")
    df_all1.fillna(df_all1['TRAINING CO2'].mean(),inplace=True) ##COMBINED DAY MEAN FOR NULL VALUES
    
    st.write(df_all1)
    df_all1.to_csv(r"day_combined.csv")
    data_frame=pd.read_csv(r"day_combined.csv") 
    #print(data_frame)
    df1=data_frame.reset_index()['TRAINING CO2']
    #print(df1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df1 =scaler.fit_transform(np.array(df1).reshape(-1,1)) 
    training_size=int(len(df1)*0.50)
    test_size=len(df1)-training_size
    print(training_size,test_size)
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step=1):
    	dataX, dataY = [], []
    	for i in range(len(dataset)-time_step-1):
    		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
    		dataX.append(a)
    		dataY.append(dataset[i + time_step, 0])
    	return np.array(dataX), np.array(dataY)
    
     # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 5
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    
     # print(X_train.shape), print(y_train.shape)
     # print(X_test.shape), print(ytest.shape)
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    
    model = Sequential()
    model.add(Bidirectional(LSTM(100, input_shape=(time_step,1))))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    #model.summary()
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=25,batch_size=2,verbose=1)
    
    x_input=df1[len(df1)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    # demonstrate prediction for next days
    predict_days=10
    lst_output=[]
    n_steps=5
    i=0
    while(i<predict_days):
    
        if(len(temp_input)>n_steps):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    
    print(lst_output)    
    
    predicted_values=list(scaler.inverse_transform(lst_output).reshape(1,-1))
    predicted_values=predicted_values[0].tolist()
    print(predicted_values)
    
    df_all1=pd.DataFrame(columns=['DATE','original CO2'])
    j=0
    for root, dirs, files in os.walk(r"C:\Users\HP\Downloads\datas\data"):
        for file in files:
            #st.write(file)
            if os.path.splitext(file)[1] == '.nc4':
                filePath = os.path.join(root, file)
            ds = nc.Dataset(filePath)
            df=pd.DataFrame(columns=["Latitude","Longitude","xco2"])
            #st.write(df)
    
            df["Longitude"] = ds['longitude'][:]
            df["Latitude"] = ds['latitude'][:]
            df["xco2"]=ds['xco2'][:]
    
            #Repalce inplace 
            df.fillna(0,inplace=True)
    
            #df_first=df[(60>df['Latitude']> 59)]
            df_first=df.loc[(df['Latitude'] >user_lat) &(df['Latitude'] < user_lat+20) & (df['Longitude']> user_lon)&(df['Longitude']< user_lon+20 ),'xco2']
            res=df_first.mean()  ##PARTICULAR DAY MEAN for areas combined
            #st.write(res)
    
    
            df_all1.loc[j,"DATE"] = file[15:17]+"/"+file[13:15]+"/20"+file[11:13]
            df_all1.loc[j,"original CO2"] = res
    
            j+=1
    df_all1.fillna(df_all1['original CO2'].mean(),inplace=True) ##COMBINED DAY MEAN FOR NULL VALUES
    st.header("ORIGINAL TEST DATA")
    st.write(df_all1)
    
    df_all1['predicted CO2']=predicted_values
    st.header("ORIGINAL VS PREDICTED TEST DATA")
    st.write(df_all1)  

    o=df_all1['original CO2'].mean()
    p=df_all1['predicted CO2'].mean()
    print(o,p)
    error_rate=(abs(o-p)/o)*100
    error_rate_str="ERROR RATE: "+str(error_rate)
    st.error(error_rate_str)
    acc_rate=100-error_rate
    acc_rate_str="ACCURACY RATE: "+str(acc_rate)
    st.success(acc_rate_str)