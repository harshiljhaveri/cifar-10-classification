from matplotlib import pyplot as plt
import numpy as np
import pickle


def get_data(inputs_file_path):
    # TODO: Load inputs and labels
    # TODO: Normalize inputs
    with open(inputs_file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)


def grad_sigma(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying CIFAR10 with 
    batched learning. Please implement the TODOs for the entire 
    model but do not change the method and constructor arguments. 
    Make sure that your Model class works with multiple batch 
    layers. Additionally, please exclusively use NumPy and 
    Python built-in functions for your implementation.
    """

    def __init__(self, layers):
        # TODO: Initialize all hyperparametrs
        self.input_size = 3072  # Size of image vectors
        self.num_classes = 10  # Number of classes/possible labels
        self.batch_size = 32
        self.learning_rate = 0.002

        # TODO: Initialize weights and biases
        self.b = [np.random.randn(y, 1) for y in layers[1:]]
        self.W = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def divideinbacthes(self, inputs, y):
        for i in range(0, inputs.shape[0], self.batch_size):
            batch = zip(inputs[i:i + self.batch_size],
                        y[i:i + self.batch_size])
            yield batch

    def forward(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 3072) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        a = inputs
        A = [inputs]  # list to store A for every layer
        Z = []  # list to store z vectors for every layer
        for b, w in zip(self.b, self.W):
            z = np.dot(w, a) + b
            Z.append(z)
            a = sigmoid(z)
            A.append(a)
        A[-1] = softmax(z)
        return Z, A

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be generally decreasing with every training loop (step). 
        :param probabilities: matrix that contains the probabilities 
        of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        return np.sum(np.nan_to_num(-labels * np.log(probabilities)))

    def compute_gradients(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases 
        after one forward pass and loss calculation. You should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each 
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        gradB = [np.zeros(b.shape) for b in self.b]
        gradW = [np.zeros(w.shape) for w in self.W]

        Z, A = self.forward(inputs)
        loss = self.loss(A[-1], labels)
        costdiff = A[-1] - labels

        diff = costdiff
        gradB[-1] = diff
        gradW[-1] = np.dot(diff, A[-2].T)

        for h in range(2, 4):
            z = Z[-h]
            diff_a = grad_sigma(z)
            diff = np.dot(self.W[-h + 1].T, diff) * diff_a
            gradW[-h] = np.dot(diff, A[-h - 1].T)
            gradB[-h] = diff

        return loss, gradB, gradW

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number 
        of correct predictions with the correct answers.
        :param probabilities: result of running model.forward() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        for i in range(len(labels)):
            if np.argmax(probabilities[i]) == labels[i]:
                count += 1
        return float(count) / len(labels)

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient 
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        # TODO: change the weights and biases of the model to descent the gradient
        # update weight and biases by multiplying ratio learning rate and batch_size
        # multiplied with the accumulated gradients(partial derivatives)
        # calculate change in weight(diff) and biases and update weight
        # with the changes
        self.W = [w - (self.learning_rate / self.batch_size) * gw for w, gw in zip(self.W, gradW)]
        self.b = [b - (self.learning_rate / self.batch_size) * gb for b, gb in zip(self.b, gradB)]


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward 
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''
    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    # Optional TODO: Call visualize_loss and observe the loss per batch as the model trains.
    bs = 32
    n = train_inputs.shape[0] // bs
    batch_iter = model.divideinbacthes(train_inputs, train_labels)
    for _ in range(bs):
        for i in range(n):
            for j in batch_iter:
                gradB = [np.zeros(b.shape) for b in model.b]
                gradW = [np.zeros(w.shape) for w in model.W]
                for batch_inputs, batch_labels in j:
                    Z, A = model.forward(batch_inputs)
                    loss, differential_B, diff_gradW = model.compute_gradients(batch_inputs, A[-1], batch_labels)
                    gradB = [gradb + delb for gradb, delb in zip(gradB, differential_B)]
                    gradW = [gradw + delw for gradw, delw in zip(gradW, differential_W)]
    model.gradient_descent(gradW, gradB)


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. 
    :param test_inputs: CIFAR10 test data (all images to be tested)
    :param test_labels: CIFAR10 test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    count = 0
    for x, j in zip(test_inputs, test_labels):
        Z, A = model.forward(x)
        if np.argmax(A[-1]) == j:
            count += 1
    acc = (float(count) / test_inputs.shape[0])
    return acc


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize loss per batch. You can call this in train() to observe.
    param losses: an array of loss value from each batch of train

    NOTE: DO NOT EDIT
    
    :return: doesn't return anything, a plot should pop-up
    """

    plt.ion()
    plt.show()

    x = np.arange(1, len(losses) + 1)
    plt.xlabel('i\'th Batch')
    plt.ylabel('Loss Value')
    plt.title('Loss per Batch')
    plt.plot(x, losses, color='r')
    plt.draw()
    plt.pause(0.001)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the zput of model.forward()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    plt.ioff()

    images = np.reshape(image_inputs, (-1, 3, 32, 32))
    images = np.moveaxis(images, 1, -1)
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()


def main():
    '''
    Read in CIFAR10 data, initialize your model, and train and test your model 
    for one epoch. The number of training steps should be your the number of 
    batches you run through in a single epoch. 
    :return: None
    '''

    # TODO: load CIFAR10 train and test examples into train_inputs, train_labels, test_inputs, test_labels
    for i in range(5):
        fileName = "data_batch_" + str(i + 1)
        data = get_data(fileName)
        try:
            if i == 0:
                features = data[b"data"]
                labels = np.array(data[b"labels"])
            else:
                features = np.append(features, data[b"data"], axis=0)
                labels = np.append(labels, data[b"labels"], axis=0)
        except KeyError:
            if i == 0:
                features = data["data"]
                labels = np.array(data["labels"])
            else:
                features = np.append(features, data["data"], axis=0)
                labels = np.append(labels, data["labels"], axis=0)

    features = features / 255.0
    features = features.reshape(-1, 3072, 1)
    one_hot = np.zeros((labels.shape[0], 10))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    one_hot = one_hot.reshape(-1, 10, 1)
    # TODO: Create Model
    model = Model([3072, 50, 30, 10])

    # TODO: Train model by calling train() ONCE on all data
    train(model, features, one_hot)

    # TODO: Test the accuracy by calling test() after running train()
    try:
        test_features = get_data("test_batch")[b"data"] / 255.0
        test_features = test_features.reshape(-1, 3072, 1)
        test_label = get_data("test_batch")[b"labels"]
    except KeyError:
        test_features = get_data("test_batch")["data"] / 255.0
        test_features = test_features.reshape(-1, 3072, 1)
        test_label = get_data("test_batch")["labels"]
    # make predictions of test dataset
    print(f"{test(model, test_features, test_label):.4f}")

    # TODO: Visualize the data by using visualize_results() on a set of 10 examples
    comp = np.zeros((10,10))
    for x in range(10):
        Z, A = model.forward(features[x])
        comp[x][np.argmax(A[-1])] = 1

    visualize_results(features[:10], comp, test_label[:10])


if __name__ == '__main__':
    main()
