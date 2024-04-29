import numpy as np
import random
import math
import os

print_epochs = True
random.seed(0)


def main():
	# np.set_printoptions(threshold=np.inf)
	# trainScratch()
	faceModels()
	# digitModels()

	# print(face_weights)
	print("Done.")

def faceModels():
	facedatatest = getNumpyArray('./facedata/facedatatest', 'face')
	facedatatestlabels = getArrFromLabels('./facedata/facedatatestlabels')
	facedatatrain = getNumpyArray('./facedata/facedatatrain', 'face')
	facedatatrainlabels = getArrFromLabels('./facedata/facedatatrainlabels')
	facedatavalidation = getNumpyArray('./facedata/facedatavalidation', 'face')
	facedatavalidationlabels = getArrFromLabels('./facedata/facedatavalidationlabels')
	
	# perceptron_weights = trainPerceptron(facedatatrain, facedatatrainlabels, "Epochs for face perceptron: ")
	# perceptron_results = testPerceptron(perceptron_weights, facedatatest, facedatatestlabels)
	# print("Accuracy for face perceptron: " + str(perceptron_results))
	neural_weights = trainNeuralNetwork(facedatatrain, facedatatrainlabels, facedatatest, facedatatestlabels)
	# neural_results = testNeuralNetwork(neural_weights, facedatatrain, facedatatrainlabels)
	# print("Accuracy for face neural network: " + str(neural_results))

def digitModels():
	digitdatatest = getNumpyArray('./digitdata/testimages', 'digit')
	digitdatatestlabels = getArrFromLabels('./digitdata/testlabels')
	digitdatatrain = getNumpyArray('./digitdata/trainingimages', 'digit')
	digitdatatrainlabels = getArrFromLabels('./digitdata/traininglabels')
	digitdatavalidation = getNumpyArray('./digitdata/validationimages', 'digit')
	digitdatavalidationlabels = getArrFromLabels('./digitdata/validationlabels')

	trainLabelsArray = transformLabelsToBinary(digitdatatrainlabels)
	perceptron_weightsArray = trainMultiplePerceptrons(digitdatatrain, trainLabelsArray)
	perceptron_results = testMultiplePerceptrons(perceptron_weightsArray, digitdatatest, digitdatatestlabels)
	print("Accuracy for digit perceptron: " + str(perceptron_results))

def testNeuralNetwork(weights, data, labels):
	input_size = data.shape[1]
	theta1 = weights[:input_size * (input_size+1)].reshape(input_size, input_size+1)
	theta2 = weights[input_size * (input_size+1):].reshape(1, input_size+1)
	bias1, bias2, threshold, correct = 1, 1, 0.5, 0
	for i in range(data.shape[0]):
		current_input = data[i]
		true_out = labels[i]
		current_input = np.insert(current_input, 0, bias1)
		(_, expected_out) = forward_prop(current_input, theta1, theta2, bias2)
		print(expected_out, true_out)
		binary_prediction = (expected_out > threshold).astype(int)
		if true_out == binary_prediction:
			correct+=1
	return correct/data.shape[0]

def trainScratch():
	x1 = np.array([1, 0, 1])
	a1 = np.insert(x1, 0, 1)
	theta1 = np.array([[1,0,0,0],[1,1,1,0],[0,1,1,1]]).astype(float)
	z2 = theta1 @ a1
	gz2 = sigmoidFunc(z2)
	a2 = np.insert(gz2, 0, 1)
	theta2 = np.array([1,0,1,0]).astype(float)
	print(theta1.shape)
	print(theta2.shape)
	z3 = theta2 @ a2
	a3 = sigmoidFunc(z3)
	y1 = 1
	d3 = a3 - y1
	# not including bias
	gpz2 = a2[1:] * (1 - a2[1:])
	# print(gpz2.shape)
	# print(theta2[1:].shape)
	# print(d3.shape)
	d2 = (theta2[1:].T * d3) * gpz2
	delta1 = np.zeros_like(theta1).astype(float)
	delta2 = np.zeros_like(theta2).astype(float)
	delta1 += d2.reshape(-1, 1) @ a1.reshape(1, -1)
	delta2 += (d3 * a2)
	theta1 -= delta1
	theta2 -= delta2



	
# input: 4200 for face
# 4200 x 4201
# 1 x 4201
# choose lambda=0
# choose alpha=1
def trainNeuralNetwork(data, labels, data2, labels2):
	# initialize
	n, input_size = data.shape
	totalWeightSize = input_size*(input_size+1)+(input_size+1)
	bound = math.sqrt(6/(input_size + 1))
	weights = np.random.uniform(-bound, bound, totalWeightSize)
	theta1 = weights[:input_size * (input_size+1)].reshape(input_size, input_size+1)
	theta2 = weights[input_size * (input_size+1):]
	bias1, bias2 = 1, 1
	for j in range(1, 21):
		print("Epoch " + str(j))
		delta1 = np.zeros_like(theta1)
		delta2 = np.zeros_like(theta2)
		correct = 0
		total_loss = 0
		for i in range(n):
			# forward prop
			yi = labels[i]
			a1 = np.insert(data[i], 0, bias1)
			z2 = theta1 @ a1
			gz2 = sigmoidFunc(z2)
			a2 = np.insert(gz2, 0, bias2)
			z3 = theta2 @ a2
			a3 = sigmoidFunc(z3)
			predicted = 0
			if a3 > 0.5:
				predicted = 1
			if predicted == yi:
				correct+=1

			loss = -(yi * np.log(a3) + (1 - yi) * np.log(1 - a3))
			total_loss += loss
			# Backprop
			d3 = a3 - yi
			gpz2 = a2[1:] * (1 - a2[1:])	
			d2 = (theta2[1:].T * d3) * gpz2

			delta1 += d2[:, None] @ a1[None, :]
			delta2 += (d3 * a2)
		AvgRegGrad1 = (1/n)*delta1
		AvgRegGrad2 = (1/n)*delta2
		theta1 -= .005*AvgRegGrad1
		theta2 -= .005*AvgRegGrad2
		print("accuracy: " + str(correct/n))
		print("loss: " + str(total_loss/n))
	threshold, correct = 0.5, 0
	for i in range(data2.shape[0]):
		yi = labels2[i]
		a1 = np.insert(data2[i], 0, bias1)
		z2 = theta1 @ a1
		gz2 = sigmoidFunc(z2)
		a2 = np.insert(gz2, 0, bias2)
		z3 = theta2 @ a2
		a3 = sigmoidFunc(z3)
		print(a3, yi)
		prediction = 0
		if a3 > threshold:
			prediction = 1
		if prediction == yi:
			correct+=1
	print(correct/data2.shape[0])
	return (correct/data2.shape[0])
	# flat_weights1 = theta1.flatten()
	# flat_weights2 = theta2.flatten()
	# weights_final = np.concatenate((flat_weights1, flat_weights2))
	# return weights_final

def forward_prop(input, theta1, theta2, bias2):
	z2 = theta1 @ input
	a2 = sigmoidFunc(z2)
	a2 = np.insert(a2, 0, bias2)
	z3 = theta2 @ a2
	expected_out = sigmoidFunc(z3)
	return (a2, expected_out)

def sigmoidFunc(x):
	return 1 / (1 + np.exp(-x))

def compute_numerical_gradient(theta, data, labels, lambd, input_size, hidden_size, output_size, epsilon=1e-4):
    num_params = len(theta)
    grad_approx = np.zeros(num_params)
    
    for i in range(num_params):
        print(i)
        theta_plus = theta.copy()
        theta_plus[i] += epsilon
        theta1_plus, theta2_plus = unpack_theta(theta_plus, input_size, hidden_size, output_size)
        cost_plus = compute_cost(theta1_plus, theta2_plus, data, labels, lambd)
        
        theta_minus = theta.copy()
        theta_minus[i] -= epsilon
        theta1_minus, theta2_minus = unpack_theta(theta_minus, input_size, hidden_size, output_size)
        cost_minus = compute_cost(theta1_minus, theta2_minus, data, labels, lambd)
        
        grad_approx[i] = (cost_plus - cost_minus) / (2 * epsilon)
    
    return grad_approx

def unpack_theta(theta, input_size, hidden_size, output_size):
    # Unpack theta into individual weight matrices
    theta1_size = (input_size + 1) * hidden_size
    theta1 = theta[:theta1_size].reshape(hidden_size, input_size + 1)
    theta2 = theta[theta1_size:].reshape(output_size, hidden_size + 1)
    return theta1, theta2

def compute_cost(theta1, theta2, data, labels, lambd):
    n = data.shape[0]
    _, K = theta2.shape  # Assuming K is the number of output classes
    
    # Forward propagation to get predicted probabilities
    predictions = np.zeros((n, K))
    for i in range(n):
        current_input = np.insert(data[i], 0, 1)  # Insert bias term
        _, expected_out = forward_prop(current_input, theta1, theta2, 1)
        predictions[i] = expected_out
    
    # Compute cross-entropy loss term
    cross_entropy_loss = 0
    for i in range(n):
        for k in range(K):
            if labels[i] == k:  # True class
                cross_entropy_loss -= np.log(predictions[i, k])
            else:  # Other classes
                cross_entropy_loss -= np.log(1 - predictions[i, k])
    
    cross_entropy_loss /= n    
    # Compute regularization term
    reg_term = lambd / (2 * n) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))
    
    # Total cost
    cost = cross_entropy_loss + reg_term
    
    return cost

# def cost(weights, labels):
# 	return (true_y*np.log(expected_y) + (1 - true_y)*np.log(1 - expected_y))

def testMultiplePerceptrons(weights, data, labels):
	numCorrect = 0
	testSize = labels.shape[0]
	for i in range(testSize):
		input = data[i]
		expected_out = labels[i]
		maxOut = -np.inf
		digit = None
		for j in range(10):
			output = weights[j][0] + input @ weights[j][1:]
			if output > maxOut:
				maxOut = output
				digit = j
		if digit == expected_out:
			numCorrect+=1
	return numCorrect/testSize

def trainMultiplePerceptrons(data, labels):
	weights = []
	for i in range(10):
		w_i = trainPerceptron(data, labels[i], f"Epochs for digit {i}: ")
		weights.append(w_i)
	weights = np.vstack(weights)
	return weights


# transforms vector like [0,1,2,3,1,2] to [[1,0,0,0,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [0,0,0,1,0,0]]
def transformLabelsToBinary(labels):
	unique_elements = np.unique(labels)
	result_arrays = []
	for ind, ele in enumerate(unique_elements):
		mask = (labels == ele)
		result_arrays.append(mask.astype(int))
	result_arrays = np.vstack(result_arrays)
	return result_arrays

# trains binary perceptron
def trainPerceptron(data, labels, epochsMessage):
	numWeights = data.shape[1] + 1
	weights = np.zeros(numWeights)
	epochs = 0
	while epochs < 5000:
		notUpdated = True
		for i in range(labels.shape[0]):
			input = data[i]
			output = weights[0] + input @ weights[1:]
			expected_out = labels[i]
			if output < 0 and expected_out:
				weights[0] += 1
				weights[1:] += input
				notUpdated = False
			elif output >= 0 and not expected_out:
				weights[0] -= 1
				weights[1:] -= input
				notUpdated = False
		epochs+=1
		if notUpdated:
			break
	if print_epochs:
		print(epochsMessage + str(epochs))
	return weights

# test binary perceptron
def testPerceptron(weights, data, labels):
	numCorrect = 0
	testSize = labels.shape[0]
	for i in range(testSize):
		input = data[i]
		output = weights[0] + input @ weights[1:]
		expected_out = labels[i]
		if output >= 0 and expected_out or output < 0 and not expected_out:
			numCorrect+=1
	return numCorrect/testSize

def getNumpyArray(input_file, type):
	if type == 'face':
		size = 4200
		preprocess_face_file(input_file, "temp_file.txt")
	elif type == 'digit':
		size = 784
		preprocess_digit_file(input_file, "temp_file.txt")        
	with open('temp_file.txt', 'r') as file:
		lines = file.readlines()
	toReturn = np.array([[int(char) for char in line.strip()] for line in lines]).reshape(-1, size)
	os.remove('temp_file.txt')
	return toReturn

def preprocess_face_file(input_file, output_file):
	with open(input_file, 'r') as file:
		lines = [line.rstrip('\n') for line in file.readlines()]

	# Replace spaces with 1s and '#' characters with 0s
	modified_lines = [line.replace(' ', '0').replace('#', '1') for line in lines]

	with open(output_file, 'w') as file:
		file.writelines(modified_lines)
		
def preprocess_digit_file(input_file, output_file):
	with open(input_file, 'r') as file:
		lines = [line.rstrip('\n') for line in file.readlines()]

	# Replace spaces with 1s and '#' characters with 0s
	modified_lines = [line.replace(' ', '0').replace('#', '1').replace('+', '1') for line in lines]

	with open(output_file, 'w') as file:
		file.writelines(modified_lines)

def getArrFromLabels(input_file):
	with open(input_file, 'r') as file:
		data = file.readlines()
	data = [int(line.strip()) for line in data]
	arr = np.array(data)
	return arr
	

if __name__ == '__main__':
	main()