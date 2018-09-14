# Classroom Activity Profiling 


### Methodology

- For implementation, we are using an array of air quality sensors for measuring different
environmental parameters inside the classroom along with a sound sensor for measuring sound
intensity levels. 

- We are grouping dataset by mean per minute for all given data. Thereafter we are calculating statndard deviation for all single point of grouped data set with respect to a certain frame or time. 

- That standard deviation will be our feature set to train the model and predict the same.

- we are using Appropriated supervised technique identify underline patterns in acquired dataset. Afterwards the same will be used for modeling classifier for automated event identification.

- We are using statistical classification model rather than advanced machine learning algorithm to classify events.

### Process
#####  1. Collection of Raw Data 
*Data is collected per second and stored in comma seperated values (csv)*.  

|          |           |     |    |    |    |        |          |        |       |       | 
|----------|-----------|-----|----|----|----|--------|----------|--------|-------|-------| 
| 4/4/2018 |  13:44:34 | 359 | 24 | 42 | 44 | 0.0329 | 20915.16 | 263.44 | 29.25 | 84.06 | 
| 4/4/2018 |  13:44:35 | 322 | 27 | 51 | 53 | 0.0278 | 20915.16 | 184.41 | 29.22 | 83.87 | 
| 4/4/2018 |  13:44:36 | 289 | 27 | 52 | 54 | 0.0244 | 20915.16 | 213.75 | 29.22 | 84    | 
| 4/4/2018 |  13:44:37 | 288 | 33 | 52 | 57 | 0.0227 | 20915.16 | 322.71 | 29.28 | 83.94 | 
| 4/4/2018 |  13:44:38 | 318 | 33 | 52 | 59 | 0.0219 | 18670.4  | 299.26 | 29.25 | 83.81 | 
| 4/4/2018 |  13:44:39 | 326 | 33 | 64 | 75 | 0.021  | 18670.4  | 194.66 | 29.25 | 83.94 | 
| 4/4/2018 |  13:44:40 | 327 | 33 | 79 | 90 | 0.021  | 16830.43 | 192.05 | 29.25 | 83.75 | 
| 4/4/2018 |  13:44:41 | 312 | 33 | 79 | 90 | 0.021  | 15296.92 | 154.14 | 29.25 | 83.75 | 
| 4/4/2018 |  13:44:43 | 285 | 33 | 79 | 90 | 0.021  | 14000.74 | 192.05 | 29.25 | 83.87 | 
Table 1.1

 ##### 2. Data Cleaning 
*All the nan and inf values were removed and there was also problem in date and time formatting in few datasets*

 ##### 3. Grouping data per minute by mean
*Therafter data was grouped per minute by mean. We used followig algorithm for grouping of data*

```python
#Loading data into dataframe
df = pd.read_csv(path)

#Adding column name to dataframe
df.columns = ["time","pm1","pm2","pm10","temp","hum"]

#Removing Seconds from timestamp, such that we will have hour and miute only
df['time'] = df['time'].str[11:-3]

#Calculating mean for all dataset with same timestamp
dfMean = df.groupby("time").mean()
```
*After applying algorithm mentioned above, data will look like* 


|          |           |     |    |    |    |        |          |        |       |       | 
|----------|-----------|-----|----|----|----|--------|----------|--------|-------|-------| 
| 4/4/2018 |  13:44 | 359 | 24 | 42 | 44 | 0.0329 | 20915.16 | 263.44 | 29.25 | 84.06 | 
| 4/4/2018 |  13:44 | 322 | 27 | 51 | 53 | 0.0278 | 20915.16 | 184.41 | 29.22 | 83.87 | 
| 4/4/2018 |  13:44 | 289 | 27 | 52 | 54 | 0.0244 | 20915.16 | 213.75 | 29.22 | 84    | 
| 4/4/2018 |  13:44 | 288 | 33 | 52 | 57 | 0.0227 | 20915.16 | 322.71 | 29.28 | 83.94 | 
| 4/4/2018 |  13:44 | 318 | 33 | 52 | 59 | 0.0219 | 18670.4  | 299.26 | 29.25 | 83.81 | 

*And final dataset for that particular minute will become*


|          |           |     |    |    |    |        |          |        |       |       | 
|----------|-----------|-----|----|----|----|--------|----------|--------|-------|-------| 
| 4/4/2018 |  13:44 | 312 | 27 | 49 | 54 | 0.0289 | 20910.16 | 299.44 | 29.25 | 83.96 | 


*This is nothing but mean of all data point of a minute grouped together*

##### 4. Extracting feature set 
*We are using PM10 and Sound for classification model, but later we will add two more data (PM2.5 and PM1)* 

|          |           |     |    |    |    |
|----------|-----------|-----|----|----|----| 
| 4/4/2018 |  13:44 | 24 | 42 | 44 | 65 |
| 4/4/2018 |  13:44 | 27 | 51 | 53 | 71 |

_Graph of Given dataSet_

![PM10 and Sound vs Time](https://image.ibb.co/fWkTc9/download.png)

##### 5. Preprocessing 
- *We are using Standard scalar Standardization here.*
- *The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1. Given the distribution of the data, each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.*
- *Standardize features by removing the mean and scaling to unit variance
Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.*
- *This is useful when you want to compare data that correspond to different units. In that case, you want to remove the units. To do that in a consistent way of all the data, you transform the data in a way that the variance is unitary and that the mean of the series is 0.*
```python
from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(dfNew[['pm10', 'sound']])
df_std = std_scale.transform(dfNew[['pm10', 'sound']])
```
*After Standardization*
![Sound vs Time](https://image.ibb.co/jntdAU/download_1.png)
![PM10 vs Time](https://image.ibb.co/nrHS4p/download_2.png)

##### 6. Classify Training dataSet 

*We divided standardized dataset of given date into two part, first part was used to train model and 2nd part was used as test dataset*
- *We are using 3X3 matrix to classify events and used the algorithm below to add class*
``` python
def labelData(npTrain,):
   
    classi = [["idle","class","Board Work"],
             ["Lecture","Teacher's Movement","Board Work"],
             ["Discussion","Gossip + BW","Gossip + mov"],
             ["Group noise","Mass Entry Exit","Mass Entry Exit"]]

    yTrain = []
    #classi[2]

    for i in range (len(npTrain)-1):
        #print(npTrain[i][2],end=" : ")
        #print(npTrain[i][5], end=" and ")
        if npTrain[i][1] < -1:
            if npTrain[i+1][0] <-1:
                    yTrain.append(classi[0][0])
            elif npTrain[i+1][0] < 1:
                    yTrain.append(classi[0][1])
            else:
                    yTrain.append(classi[0][2])
        elif npTrain[i][1] < 0:
            if npTrain[i+1][0] <-1:
                yTrain.append(classi[1][0])
            elif npTrain[i+1][0] < 1:
                yTrain.append(classi[1][1])
            else:
                yTrain.append(classi[1][2])

        elif npTrain[i][1] < 1:
            if npTrain[i+1][0] <-1:
                yTrain.append(classi[2][0])
            elif npTrain[i+1][0] < 1:
                yTrain.append(classi[2][1])
            else:
                yTrain.append(classi[2][2])

        elif npTrain[i][1] > 1:
            if npTrain[i+1][0] <-1:
                yTrain.append(classi[3][0])
            elif npTrain[i+1][0] < 1:
                yTrain.append(classi[3][1])            
            else:
                yTrain.append(classi[3][2])


        #print(yTrain[i])
    yTrain.append('x')
    return yTrain
```

*We Now have train dataset ready with feauture data and respective class*

|           |     |    |    |    |
|-----------|-----|----|----|----| 
Time | x1(pm10) | x2(sound) | x3(pm2) | Class |
13:44 | -1.02149641 |-0.01892986 |-1.033 | idle | 
13:44 | 3.50402158 | 1.54883038 | 3.0448733 | mass entry/exit |

##### 7. Building Model
*We used Naive Baye's classifier first but since it is not pure Machine learning(computation based) classification algorithm, So it needed lot's of mathematical works to be done in preprocessing.
So Later, We switched to conventinal machine learning algorithm like SVM and Decision Tree classfier*
```python
model = svm.SVC(kernel='linear', C=1, gamma=1) 
model.fit(x, y)
model.score(x, y)
#Predict Output
predicted= model.predict(npTest)

a = []
t = 0
for i in range(len(npTrain)):
    
    predicted= model.predict([npTrain[i]])
    a.append([npTime[i],predicted[0]])

t = i+1

for i in range(len(npTest)):
    
    
    predicted= model.predict([npTest[i]])
    a.append([npTime[i+t],predicted[0]])
    
dfoutsvm = pd.DataFrame(a)
dfoutsvm.columns = ['time','eventSVMl']
dfout['svml'] = dfoutsvm['eventSVMl']
a = dfout.values
```
*We also tried tuning parameters (like type of kernel of svm model)* 
- The kernel parameter can be tuned to take “Linear”,”Poly”,”rbf” etc.
- The gamma value can be tuned by setting the “Gamma” parameter.

### Conclusion
We are able to classify certain classroom events using three data points mainly
- PM10
- Sound
- PM2.5

The Events that can be classified using the dataset we have are:
- Idle
- mass Entry/Exit
- Discussion
- Lecture (Single Point)
- Board Work 
- Gossip
- Little Movement and sound

### To Do

- Validation of model by collection more data 
- testing model 
- Finding co-relation between different datapoints 

### Features
- Since, We standardize data before fitting into model, so no need to declare a thresold value
- Standardization also helps in making dataset unitless. So it becomes easy to work with two different type of datapoints having different unit.
 
