from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import io
import base64

global uname, dataset, X, Y, labels, unique, label_encoder, sc, pca, rf_cls

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    return a, p, r, f

def RunML(request):
    if request.method == 'GET':
        global X, Y, rf_cls
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
        svm_cls = svm.SVC(kernel='poly', C=3.0, gamma=0.25, tol=0.1, degree=3)
        svm_cls.fit(X_train, y_train)
        predict = svm_cls.predict(X_test)
        a, p, r, f = calculateMetrics("Propose SVM Algorithm", predict, y_test)
        
        rf_cls = RandomForestClassifier()
        rf_cls.fit(X_train, y_train)
        predict = rf_cls.predict(X_test)
        a1, p1, r1, f1 = calculateMetrics("Extenssion Random Forest Algorithm", predict, y_test)
        
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th><th><font size="" color="black">Precision</th>'
        output+='<th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th></tr>'
        algorithms = ['Propose SVM', 'Extension Random Forest']
        output+='<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(a)+'</td><td><font size="" color="black">'+str(p)+'</td><td><font size="" color="black">'+str(r)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
        output+='<td><font size="" color="black">'+algorithms[1]+'</td><td><font size="" color="black">'+str(a1)+'</td><td><font size="" color="black">'+str(p1)+'</td><td><font size="" color="black">'+str(r1)+'</td><td><font size="" color="black">'+str(f)+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['Propose SVM','Precision',p],['Propose SVM','Recall',r],['Propose SVM','F1 Score',f],['Propose SVM','Accuracy',a],
                           ['Extension Random Forest','Precision',p1],['Extension Random Forest','Recall',r1],['Extension Random Forest','F1 Score',f1],['Extension Random Forest','Accuracy',a1],
                          ],columns=['Algorithms','Metrics','Value'])
        df.pivot_table(index="Algorithms", columns="Metrics", values="Value").plot(kind='bar', figsize=(6, 4))
        plt.title("All Algorithms Performance Graph")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def ProcessMining(request):
    if request.method == 'GET':
        global dataset, X, Y, labels, unique, label_encoder, sc, pca
        dataset.fillna(0, inplace = True)
        Y = dataset['isFraud'].values.ravel()
        dataset.drop(['isFraud'], axis = 1,inplace=True)
        label_encoder = []
        columns = dataset.columns
        types = dataset.dtypes.values
        for i in range(len(types)):
            name = types[i]
            if name == 'object': #finding column with object type
                print(columns[i]+"===========")
                le = LabelEncoder()
                dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
                label_encoder.append([columns[i], le])
        dataset.fillna(0, inplace = True)            
        X = dataset.values
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        unique, count = np.unique(Y, return_counts = True)
        pca = PCA(2) 
        XX = pca.fit_transform(X)
        plt.figure(figsize=(7, 7))
        for cls in unique:
            plt.scatter(XX[Y == cls, 0], XX[Y == cls, 1], label=cls) 
        plt.legend()
        plt.title("Process Mining User Behaviour Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data':"User Behaviour Process Mining Graph", 'img': img_b64}
        return render(request, 'UserScreen.html', context)           

def DetectFraud(request):
    if request.method == 'GET':
       return render(request, 'DetectFraud.html', {})

def DetectFraudAction(request):
    if request.method == 'POST':
        global dataset, X, Y, sc, rf_cls, label_encoder
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("FraudApp/static/Data.csv"):
            os.remove("FraudApp/static/Data.csv")
        fs = FileSystemStorage()
        filename = fs.save('FraudApp/static/Data.csv', myfile)
        dataset = pd.read_csv('FraudApp/static/Data.csv')
        dataset.fillna(0, inplace = True)
        temp = dataset.values
        for i in range(len(label_encoder)):
            le = label_encoder[i]
            if le[0] == 'R_emaildomain':
                dataset[le[0]] = pd.Series(le[1].fit_transform(dataset[le[0]].astype(str)))#encode all str columns to numeric
            else:
                dataset[le[0]] = pd.Series(le[1].transform(dataset[le[0]].astype(str)))#encode all str columns to numeric
        dataset.fillna(0, inplace = True)
        X = dataset.values
        X = sc.transform(X)
        predict = rf_cls.predict(X)
        output = ''
        output+='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test Data</th><th><font size="" color="black">Detection Result</th></tr>'
        for i in range(len(predict)):
            pred = "Normal"
            if predict[i] == 1:
                pred = "Fraud"
            output+='<td><font size="" color="black">'+str(temp[i])+'</td><td><font size="" color="black">'+str(pred)+'</td></tr>'
        context= {'data': output}
        return render(request, 'UserScreen.html', context)           


def LoadDatasetAction(request):
    if request.method == 'POST':
        global dataset, X, Y
        myfile = request.FILES['t1']
        name = request.FILES['t1'].name
        if os.path.exists("FraudApp/static/Data.csv"):
            os.remove("FraudApp/static/Data.csv")
        fs = FileSystemStorage()
        filename = fs.save('FraudApp/static/Data.csv', myfile)
        dataset = pd.read_csv('FraudApp/static/Data.csv')
        context= {'data': "Dataset Loaded. Below are some values from Dataset<br/><br/>"+str(dataset)}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):
    if request.method == 'GET':
       return render(request, 'LoadDataset.html', {})   
    
def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)

 
