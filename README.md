# Intro to AI - Final Project  
### Naive Bayes & Perceptron for Digit Recognition and Facial Classification  

This was the final project for my **Intro to AI** class, where I implemented **Naive Bayes** and **Perceptron** classifiers for **handwritten digit classification (0-9)** and **facial recognition**. I trained both models, tested them, and compared their accuracy and training time using different amounts of training data.  

## What’s in this project  
- Implemented Naive Bayes & Perceptron to classify digits and faces  
- Tested accuracy & training time using different training set sizes  
- Analyzed results through graphs  
- Faced issues like data loading errors and division by zero in Naive Bayes and fixed them  

## Results  
- **Perceptron is better for digits** since it learns iteratively and adapts to complex patterns  
- **Naive Bayes is just as good for faces** and is much faster, making it the better choice for binary classification  
- **See the graphs below for the comparison**  

### Accuracy vs. Training Size  
![Accuracy vs Training Size](Accuracy%20vs%20Training%20Size.png)  

### Training Time vs. Training Size  
![Training Time vs Training Size](Training%20Time%20vs%20Training%20Size.png)  

## Files included  
- `Implement.py` → The full implementation of both classifiers  
- `data/` → Contains all training and testing images/labels  
- `Outputs.PNG` → Screenshot of accuracy results from the terminal  
- `Accuracy vs Training Size.png` and `Training Time vs Training Size.png` → Performance graphs  
- `AI_Final_Dayyan_Hamid.pdf` → My final report explaining everything  

## Running the Code  
Make sure you have **Python 3**, **NumPy**, and **Matplotlib** installed. Then just run: "python3 Implement.py" in the terminal. This will train both models, evaluate them, and output accuracy results.

