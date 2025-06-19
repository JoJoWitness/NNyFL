import numpy as np
import pandas as pd
import sys, getopt 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.acc = None

    def load_weights(self, X):
        self.weights = np.zeros(1 + X.shape[1])

    def weighted_sum(self, X):
        weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
        return weighted_sum

    def predict(self, X):
        predict = np.where(self.weighted_sum(X) >= 0.0, 1,0)
        return predict

    def update(self, y_true, y_pred):
        update = self.learning_rate * (y_true - y_pred)
        return update
        
    def fit(self, X, y):
        for _ in range(self.n_iterations):
            error = 0
            for ix, y_true in zip(X, y):
                y_pred = self.predict(ix)
                update = self.update(y_true, y_pred)
                self.weights[1:] =  self.weights[1:] + update * ix
                self.weights[0] = self.weights[0] + update
        return self
    
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        self.acc = np.mean(y_pred == y)


def main():

    input_flag = False
    
    df = pd.read_csv('glass.data', header=None)
    df = shuffle(df)

    X = df.iloc[:, 1:10].values
    y = df.iloc[:, 10].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.5)

    perceptrons = []
    for glass_type in range(0, 7):
        train_tag = np.where(train_labels == glass_type + 1, 1, 0)
        test_tag = np.where(test_labels == glass_type + 1, 1, 0)
        p = Perceptron(learning_rate=0.001, n_iterations=1000)
        p.load_weights(train_data)
        p.fit(train_data, train_tag)
        p.accuracy(test_data, test_tag)
        perceptrons.append(p)

    while not input_flag:
        user_input = input("Ingrese 9 valores separados por comas: ")
        parse_input = np.array([float(x) for x in user_input.strip().split(",")])
        parse_input_scaled = scaler.transform(parse_input.reshape(1, -1))
        if parse_input.shape[0] != 9:
            print("Debe ingresar 9 valores")
            continue
        input_flag = True

    activations = [p.weighted_sum(parse_input_scaled)[0] for p in perceptrons]
    predicted_type = np.argmax(activations)
    print(f"Vidrio de tipo: {predicted_type+1}")
    print(f"La precision es del: {perceptrons[predicted_type].acc}")
    print(f"El error es del: {1 - perceptrons[predicted_type].acc}")
    

if __name__ == "__main__":
    main()

