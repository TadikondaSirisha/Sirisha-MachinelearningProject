from tkinter import *
from tkinter import filedialog, ttk
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  # Using MLPClassifier for DNN
import torch
import torch.nn as nn

main = Tk()
main.title("Detection of Ransomware Attacks Using Processor and Disk Usage Data")
main.geometry("1300x1200")

# Global variables
global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler
global accuracy, precision, recall, fscore
precision = []
recall = []
fscore = []
accuracy = []

def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END, str(dataset))
    labels, count = np.unique(dataset['label'], return_counts=True)
    height = count
    bars = ['Benign', 'Ransomware']
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.show()

def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, scaler
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)
    data = dataset.values
    X = data[:, 1:data.shape[1]-1]
    Y = data[:, data.shape[1]-1].astype(int)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END, "Normalized Features\n")
    text.insert(END, X)
    text.insert(END, "\n\nDataset Train & Test Split Details\n")
    text.insert(END, "80% dataset for training : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "20% dataset for testing  : " + str(X_test.shape[0]) + "\n")

def calculateMetrics(algorithm, predict, testY):
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    text.insert(END, algorithm + ' Accuracy  : ' + str(a) + "\n")
    text.insert(END, algorithm + ' Precision   : ' + str(p) + "\n")
    text.insert(END, algorithm + ' Recall      : ' + str(r) + "\n")
    text.insert(END, algorithm + ' FMeasure    : ' + str(f) + "\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, annot=True, cmap="viridis", fmt="g")
    plt.title(algorithm + " Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def runsvm():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    svm_cls = svm.SVC(kernel="poly", gamma="scale", C=0.004)
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)

def runknn():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    knn_cls = KNeighborsClassifier(n_neighbors=5)
    knn_cls.fit(X_train, y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN", predict, y_test)

def runDT():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    dt_cls = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=2)
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree", predict, y_test)

def runRF():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    rf = RandomForestClassifier(n_estimators=40, criterion='gini', max_features="log2")
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    calculateMetrics("Random Forest", predict, y_test)

def runXGBoost():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    xgb_cls = XGBClassifier(n_estimators=10, learning_rate=0.09, max_depth=2)
    xgb_cls.fit(X_train, y_train)
    predict = xgb_cls.predict(X_test)
    calculateMetrics("XGBoost", predict, y_test)

def autoencoders():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)

    # Define the Autoencoder model
    class Autoencoder(nn.Module):
        def __init__(self, input_size):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_size),
                nn.Sigmoid()  # Use Sigmoid for output layer
            )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # Convert training data to PyTorch tensors
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)

    # Initialize the Autoencoder model
    input_size = X_train.shape[1]
    model = Autoencoder(input_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the Autoencoder
    for epoch in range(100):  # Number of epochs
        model.train()
        optimizer.zero_grad()
        outputs = model(train_tensor)
        loss = criterion(outputs, train_tensor)  # Compare output with input
        loss.backward()
        optimizer.step()

    # Perform reconstruction on the test data
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_tensor)

    # Calculate reconstruction error
    reconstruction_error = torch.mean((reconstructed - test_tensor) ** 2, dim=1).numpy()
    threshold = np.percentile(reconstruction_error, 95)  # Set a threshold for anomaly detection
    predictions = (reconstruction_error > threshold).astype(int)  # 1 for anomaly, 0 for normal

    # Calculate metrics
    calculateMetrics("Autoencoders", predictions, y_test)

def runDNN():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    calculateMetrics("DNN", predict, y_test)

def runLSTM():
    global X_train, y_train, X_test, y_test
    text.delete('1.0', END)

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    class LSTMModel(nn.Module):
        def __init__(self):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
            self.fc = nn.Linear(50, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    train_tensor = torch.FloatTensor(X_train_reshaped)
    train_labels = torch.FloatTensor(y_train).view(-1, 1)
    test_tensor = torch.FloatTensor(X_test_reshaped)

    model = LSTMModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_tensor)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_tensor)
        predictions = (torch.sigmoid(test_outputs) > 0.5).int().numpy()

    calculateMetrics("LSTM", predictions, y_test)

def display_comparison_table():
    # Check if all algorithms have been run
    if len(accuracy) < 7 or len(precision) < 7 or len(recall) < 7 or len(fscore) < 7:
        text.insert(END, "Not all algorithms have been run. Please run all algorithms before generating the comparison table.\n")
        return

    # Clear the Treeview
    for row in tree.get_children():
        tree.delete(row)

    # Insert data into the Treeview
    for i in range(len(accuracy)):
        tree.insert("", "end", values=(['SVM', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'Autoencoders', 'DNN'][i], accuracy[i], precision[i], recall[i], fscore[i]))

# GUI Layout
font = ('times', 16, 'bold')
title = Label(main, text='Detection of Ransomware Attacks Using Processor and Disk Usage Data')
title.config(bg='honeydew2', fg='DodgerBlue2', font=font, height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=27, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=200)
text.config(font=font1)

# Treeview for comparison table
tree = ttk.Treeview(main, columns=('Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score'), show='headings')
tree.heading('Algorithm', text='Algorithm')
tree.heading('Accuracy', text ='Accuracy')
tree.heading('Precision', text='Precision')
tree.heading('Recall', text='Recall')
tree.heading('F1 Score', text='F1 Score')
tree.place(x=10, y=500, width=1260, height=200)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Attack Database", command=uploadDataset)
uploadButton.place(x=10, y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=250, y=100)
processButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runsvm)
svmButton.place(x=490, y=100)
svmButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runknn)
knnButton.place(x=730, y=100)
knnButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree", command=runDT)
dtButton.place(x=970, y=100)
dtButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRF)
rfButton.place(x=1200, y=100)
rfButton.config(font=font1)

xgButton = Button(main, text="Run XGBoost Algorithm", command=runXGBoost)
xgButton.place(x=10, y=150)
xgButton.config(font=font1)

autoencButton = Button(main, text="Run Autoencoders", command=autoencoders)
autoencButton.place(x=250, y=150)
autoencButton.config(font=font1)

dnnButton = Button(main, text="Run DNN Algorithm", command=runDNN)
dnnButton.place(x=490, y=150)
dnnButton.config(font=font1)

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=730, y=150)
lstmButton.config(font=font1)

# Button to show comparison table
tableButton = Button(main, text="Show Comparison Table", command=display_comparison_table)
tableButton.place(x=970, y=150)
tableButton.config(font=font1)

main.config(bg='honeydew2')
main.mainloop()