import numpy as np
import matplotlib.pyplot as plt
import data

RAND_SEED = 100
param_niter = 10000
param_delta = 0.05
loss_over_time = np.zeros(param_niter)


def softmax(x):
    exp = np.exp(x)
    sumexp = np.sum(np.exp(x), axis=1)
    return exp / sumexp[:, None]


def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    return exp_x_shifted / np.sum(exp_x_shifted)


def binlogreg_train(input_data, labels):

    C = int(max(labels) + 1)
    N = len(labels)
    D = 2

    # inicjalizacija parametara w i b
    weights = np.random.randn(C, D)
    bias = np.zeros((C, ), dtype=int)

    # true_probs_hot_encoded sluzi za gradijente
    # slicno je true_probs, ali ovo je za efikasnije racunanje
    # parcijalnih gubitaka po klasifikacijskim mjerama
    true_probs_hot_encoded = np.zeros((N, C), dtype=int)
    # fill true_probs_hot_encoded
    for ind in range(N):
        true_probs_hot_encoded[ind][labels[ind]] = 1

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    for epoch in range(param_niter):

        # klasifikacijske mjere
        scores = np.dot(input_data, weights.T) + bias  # N x 1

        # vjerojatnosti razreda c_1
        probs = softmax(scores)  # N x 1

        # vjerojatnosti tocnog razreda: c0 ili c1
        # true_probs sluzi za racunanje loss-a
        '''
        true_probs = np.empty(N)
        for ind in range(len(true_probs)):
        true_probs[ind] = probs[ind][true_labels[ind]]
        '''
        if epoch % 1000 == 0:
            # gubitak
            loss = (-1/N) * np.sum(np.log(probs))
            print('iteration: ', epoch)
            print('loss: ', loss)

        # racunanje matrice aposteriornih vrijednosti po razredima
        # MOZDA MOZE POBOLJSANJE, A NE probs.T !!!
        aposterior_matrix = probs - true_probs_hot_encoded

        # gradijenti parametara
        grad_weights = (1/N) * np.dot(aposterior_matrix.T, input_data)
        grad_bias = np.sum(aposterior_matrix.T, axis=1)

        # poboljšani parametri
        weights += -param_delta * grad_weights
        # bias += -param_delta * grad_bias
        np.add(bias, -param_delta*grad_bias, out=bias, casting="unsafe")

    return weights, bias


def logreg_classify(input_data, weights, bias):

    # klasifikacijske mjere
    scores = np.dot(input_data, weights.T) + bias

    # vjerojatnosti razreda
    return np.array(softmax(scores))  # N x 1


def logreg_decfun(w, b):
    def classify(X):
        return logreg_classify(X, w, b)
    return classify


if __name__ == "__main__":

    np.random.seed(RAND_SEED)

    # get the training dataset
    input_data, true_labels = data.sample_gauss_2d(4, 100)
    true_labels = np.int_(true_labels)
    plt.figure(figsize=(10, 7))
    plt.scatter(input_data[:, 0], input_data[:, 1],
                c=true_labels, cmap='plasma', s=20, alpha=.7)
    # plt.show()

    # train the model
    weights, bias = binlogreg_train(input_data, true_labels)

    # graph loss over time
    '''
    plt.clf()
    plt.scatter(np.arange(0, param_niter), loss_over_time, s=1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
    '''

    # evaluate the model on the training dataset
    probs_by_class = logreg_classify(input_data, weights, bias)

    predicted_labels = np.copy(probs_by_class)
    predicted_labels = np.argmax(predicted_labels, axis=1)

    ''' DIJAGNOSTICKI ISPIS
    print('true_labels:\n', true_labels)
    print('predicted_labels:\n', predicted_labels)
    '''

    # report performance
    accuracy, recall, precision, confusion_matrix = data.eval_perf_multi(
        predicted_labels, true_labels)
    print('\nAccuracy: ', accuracy, '\nRecall: ', recall,
          '\nPrecision: ', precision)
    # AP = data.eval_AP(true_labels[probs_c1.argsort()])
    # print('nAP: ', AP)

    # graph the decision surface
    decfun = logreg_decfun(weights, bias)
    bbox = (np.min(input_data, axis=0), np.max(input_data, axis=0))
    data.graph_surface_2(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(input_data, true_labels, predicted_labels)

    # shot the plot
    plt.show()
