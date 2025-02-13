import numpy as np
import matplotlib.pyplot as plt
import random 
from statistics import mean, stdev
import time 

#the first thing we need to do is load all of hte data 
def load_images(image_file, label_file, image_height, image_width):
    #this loads the images so all lines form the file are read into a list of strings 
    with open(image_file, 'r') as f:
        lines = f.readlines()

    num_images = len(lines) // image_height #this is the number of images in the file MIGHT NEED TO CHANGE THIS
    images = []
    #now we need to process each image to label 
    for i in range(num_images):
        #we need to eaxtract the image lines here 
        image_lines = lines[i * image_height: (i + 1) * image_height] #this is a list of lines for each image, as in image one is from 0 (0 * image_height = 0) to (1 * image_height)
        binary_image = []
        for line in image_lines: 
            binary_line = []
            for char in line:  
                if char in ['#','+']:#nothing is 0 and anything is 1 in the image 
                    binary_line.append(1)
                else:
                    binary_line.append(0) 
            binary_line = binary_line[:image_width] + [0] * max(0, image_width - len(binary_line))  # Ensure the correct width
            binary_image.append(binary_line) #line by line added 
        images.append(binary_image) #this numpy array contains the binary for images
    images = np.array(images) #this is the numpy array of images

    #now we load labeled data 
    with open(label_file, 'r') as f:
        labels = f.readlines()
    
    return images, labels #returns labeled images 

#digit training set 
digit_train_images, digit_train_labels = load_images(
    image_file='data/digitdata/trainingimages',
    label_file='data/digitdata/traininglabels',
    image_height=28,  
    image_width=28    
)
#]faces training set 
face_train_images, face_train_labels = load_images(
    image_file='data/facedata/facedatatrain',
    label_file='data/facedata/facedatatrainlabels',
    image_height=70,  
    image_width=60    
)

# Digit test set
digit_test_images, digit_test_labels = load_images(
    image_file='data/digitdata/testimages',  # Path to test images
    label_file='data/digitdata/testlabels',  # Path to test labels
    image_height=28,
    image_width=28
)

# Face test set 
face_test_images, face_test_labels = load_images(
    image_file='data/facedata/facedatatest',
    label_file='data/facedata/facedatatestlabels',
    image_height=70,
    image_width=60
)

# # print to verify
# print(f"Digit Training Images Shape: {digit_train_images.shape}")
# print(f"Face Training Images Shape: {face_train_images.shape}")


#now we need to implement the naive bayes classifier
class NaiveBayesClassifier:
    def __init__(self, num_class):
        self.class_priors = None
        self.feature_probs = None
        self.num_classes = num_class  #this is the number of classifications (ex 10 options for digits and 2 for faces)

    def train(self, x, y):
        num_samples, num_features = x.shape #gets number of samples and features
        self.class_priors = np.zeros(self.num_classes) #initalize and array to hold P(C) for each class
        self.feature_probs = np.zeros((self.num_classes, num_features)) #initalize an array to hold P(X|C) for each class and feature
        alpha = 1  # Laplace smoothing factor, found it online in naive bayes and it makes sure that porbabilities are not 0 or exactly 1
        for c in range(self.num_classes):
            x_c = x[y == c]  # select samples belonging to specific class (c)
            self.class_priors[c] = len(x_c) / num_samples  # calculate P(C)
            self.feature_probs[c] = (np.sum(x_c, axis=0)+alpha) / (len(x_c)+ 2*alpha) #calculate P(X|C) for each feature, sum of feature values over num of samples
        pass

    def predict(self, x): #make C predicictions based on the highest log P(c|X)
        num_samples, num_features = x.shape #gets number of samples and features
            # log probabilities to avoid underflow
        log_priors = np.log(self.class_priors)  # log(P(C))
        log_likelihoods = np.log(self.feature_probs)  # log(P(X|C))
        log_one_minus_likelihoods = np.log(1 - self.feature_probs)  # log(1 - P(X|C))
        predictions = []
        for sample in x:
            log_post = log_priors + np.sum(sample * log_likelihoods + (1 - sample) * log_one_minus_likelihoods, axis=1) #calc log P (C|X)
            predictions.append(np.argmax(log_post)) #find the max log P(C|X) to give the prediction (aka most likely)

        return np.array(predictions)
#now we implement the perceptron classifier
class PerceptronClassifier:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = np.zeros((num_classes, num_features))  # weight matrix (num_classes x num_features) initialized to zeros
        self.bias = np.zeros(num_classes)  # bias vector inittialized to zeros

    def train(self, x, y, epochs=10): #x is  the training data, y is the labels
        for epoch in range(epochs):  #epcoh is any repetition of the training data aka iteratiosn 
            for i, sample in enumerate(x): # each training sample 
                true_label = y[i] 
                for c in range(self.num_classes):
                    # prediction score for class c
                    score = np.dot(self.weights[c], sample) + self.bias[c]
                    # Binary classification: Correct if c == true_label, otherwise incorrect
                    if (c == true_label and score <= 0) or (c != true_label and score > 0):
                        # Update weights and bias
                        adjustment = 1 if c == true_label else -1
                        self.weights[c] += adjustment * sample
                        self.bias[c] += adjustment

    def predict(self, x):
        #output is the predicted class label for each sample in x
        predictions = []
        for sample in x:
            # Compute scores for all classes
            scores = np.dot(self.weights, sample) + self.bias
            # Choose the class with the highest score
            predictions.append(np.argmax(scores))
        return np.array(predictions)


# to randomly sample training data   
def sample_training_data(x, y, percentage):
    #x is training data (features) y is training labels and percentage is percent of data to sample 
    num_samples = int(len(x) * percentage / 100)
    indices = random.sample(range(len(x)), num_samples)
    x_sampled = x[indices]
    y_sampled = y[indices]
    return x_sampled, y_sampled

# Evaluate accuracy with different training percentages
def evaluate_classifier(nb, x_train, y_train, x_test, y_test, percentages, iterations=5):
    #x is data (features) y is labels 
    results = {}
    for percentage in percentages:
        accuracies = []
        for _ in range(iterations):
            # subset of the training data
            x_sampled, y_sampled = sample_training_data(x_train, y_train, percentage)
            # train
            nb.train(x_sampled, y_sampled)
            # test on the full test set
            y_pred = nb.predict(x_test)
            acc = np.mean(y_test == y_pred)
            accuracies.append(acc)
        # save mean and standard deviation
        results[percentage] = {
            'mean': mean(accuracies),
            'std': stdev(accuracies)
        }
    return results

percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #in the description of project need to train x percetn at a time 
iterations = 5 #he said cna be more 

# flatten the image data
digit_train_data = digit_train_images.reshape(digit_train_images.shape[0], -1)
digit_test_data = digit_test_images.reshape(digit_test_images.shape[0], -1)
face_train_data = face_train_images.reshape(face_train_images.shape[0], -1)  # Flatten to (num_samples, 4200)
face_test_data = face_test_images.reshape(face_test_images.shape[0], -1)

digit_train_labels = np.array([int(label.strip()) for label in digit_train_labels])
digit_test_labels = np.array([int(label.strip()) for label in digit_test_labels])
face_train_labels = np.array([int(label.strip()) for label in face_train_labels])
face_test_labels = np.array([int(label.strip()) for label in face_test_labels])

# initialize and train DIGITS
nb_digits = NaiveBayesClassifier(num_class=10) 
perceptron_digits = PerceptronClassifier(num_features=784 , num_classes=10) #784 is the number of features in the data 28x28
# initialize and train FACES
nb_faces = NaiveBayesClassifier(num_class=2)
perceptron_faces = PerceptronClassifier(num_features=4200, num_classes=2)  # 4200 = 70x60 features


# evalutaion DIGITS
results_perceptron = evaluate_classifier(perceptron_digits, digit_train_data, digit_train_labels, digit_test_data, digit_test_labels, percentages, iterations)
results = evaluate_classifier(nb_digits, digit_train_data, digit_train_labels, digit_test_data, digit_test_labels, percentages, iterations)
# evalutaion FACES
results_faces_nb = evaluate_classifier(nb_faces, face_train_data, face_train_labels, face_test_data, face_test_labels, percentages, iterations)
results_faces_perceptron = evaluate_classifier(perceptron_faces, face_train_data, face_train_labels, face_test_data, face_test_labels, percentages, iterations)

#  print results
for percentage, stats in results.items():
    print(f"DIGITS - Bayes - Training size: {percentage}% -> Mean Accuracy: {stats['mean']:.2f}, Std Dev: {stats['std']:.2f}")
for percentage, stats in results_perceptron.items():
    print(f"DIGITS - Perceptron - Training size: {percentage}% -> Mean Accuracy: {stats['mean']:.2f}, Std Dev: {stats['std']:.2f}")
for percentage, stats in results_faces_nb.items():
    print(f"Face - Naive Bayes - Training size: {percentage}% -> Mean Accuracy: {stats['mean']:.2f}, Std Dev: {stats['std']:.2f}")
for percentage, stats in results_faces_perceptron.items():
    print(f"Face - Perceptron - Training size: {percentage}% -> Mean Accuracy: {stats['mean']:.2f}, Std Dev: {stats['std']:.2f}")



















##below this is for generating the graphs 
# To store timing information
training_times_nb = []
training_times_perceptron = []

# Evaluate classifiers and track training times
for percentage in percentages:
    times_nb = []
    times_perceptron = []

    for _ in range(iterations):
        # Sample data for this percentage
        x_sampled, y_sampled = sample_training_data(digit_train_data, digit_train_labels, percentage)
        
        # Measure training time for Naive Bayes
        nb_start = time.time()
        nb_digits.train(x_sampled, y_sampled)
        nb_end = time.time()
        times_nb.append(nb_end - nb_start)

        # Measure training time for Perceptron
        perceptron_start = time.time()
        perceptron_digits.train(x_sampled, y_sampled)
        perceptron_end = time.time()
        times_perceptron.append(perceptron_end - perceptron_start)

    # Store average training time for this percentage
    training_times_nb.append(np.mean(times_nb))
    training_times_perceptron.append(np.mean(times_perceptron))

# Extract mean of accuracies for plotting
mean_accuracy_nb = [results[percentage]['mean'] for percentage in percentages]
mean_accuracy_perceptron = [results_perceptron[percentage]['mean'] for percentage in percentages]

# Plot 1: Training Time vs Training Size
plt.figure(figsize=(10, 6))
plt.plot(percentages, training_times_nb, label="Naive Bayes", marker='o', color='blue')
plt.plot(percentages, training_times_perceptron, label="Perceptron", marker='o', color='orange')
plt.xlabel("Training Size (%)")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time vs Training Size")
plt.legend()
plt.grid()
plt.show()

# Plot 2: Accuracy vs Training Size
plt.figure(figsize=(10, 6))
plt.plot(percentages, mean_accuracy_nb, label="Naive Bayes", marker='o', color='blue')
plt.plot(percentages, mean_accuracy_perceptron, label="Perceptron", marker='o', color='orange')
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Training Size")
plt.legend()
plt.grid()
plt.show()