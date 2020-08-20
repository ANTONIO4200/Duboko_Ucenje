import numpy as np
import matplotlib.pyplot as plt
import data

RAND_SEED = 93214560
param_niter = 100000
param_delta = 0.1
loss_over_time = np.empty(param_niter)

'''
X: matrica podataka dimenzija NxD;
Y_: vektor točnih razreda podataka dimenzija Nx1 (koristimo ga tijekom učenja);
Y: vektor predviđenih razreda podataka dimenzija Nx1
    (koristimo ga tijekom ispitivanja performanse).
'''


def binlogreg_train(input_data, labels):
    '''
    Argumenti
      input_data:  podatci, np.array NxD
      labels: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      weights, bias: parametri logističke regresije
  '''

    # inicjalizacija parametara w i b
    weights = np.random.randn(2)
    bias = 0
    N = len(labels)

    '''
    print('\ninput_data: \n', input_data)
    print('\nlabels: \n', labels)
    print('\nweights: \n', weights)
    print('\nbias: \n', bias)
    print('\n-------------')
    '''
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    for epoch in range(param_niter):

        # klasifikacijske mjere
        scores = np.dot(input_data, weights) + bias  # N x 1

        # vjerojatnosti razreda c_1
        probs = sigmoid(scores)  # N x 1

        dL_dscores = probs - true_labels  # N x 1

        '''
        # vjerojatnosti tocnog razreda: c0 ili c1
        for ind in range(len(probs)):
            if true_labels[ind] == 0:
                probs[ind] = 1 - probs[ind]
        '''

        # gradijenti parametara
        # grad_weights = (1/N) * np.sum(np.dot(dL_dscores.T, input_data))     # D x 1
        grad_weights = (1/N) * (np.dot(dL_dscores.T, input_data))
        grad_bias = (1/N) * np.sum(dL_dscores)     # 1 x 1

        # poboljšani parametri
        weights += -param_delta * grad_weights
        bias += -param_delta * grad_bias

        if epoch % 1000 == 0:
            # gubitak
            loss = (-1/N) * np.sum(np.log(probs))
            print('iteration: ', epoch)
            print('loss: ', loss)

            ''' DIJAGNOSTICKI ISPIS
            print('scores:\n', scores)
            print('probs:\n', probs)
            print('true_probs:\n', true_probs)
            print('dL_dscores:\n', dL_dscores)
            print('grad_weights:\n', grad_weights)
            print('grad_bias:\n', grad_bias)
            print('\n\n')
            '''

    return weights, bias


def binlogreg_classify(input_data, weights, bias):
    '''
    Argumenti
      input_data:    podatci, np.array NxD
      weights, bias: parametri logističke regresije

    Povratne vrijednosti
      probs: vjerojatnosti razreda c1
    '''
    # klasifikacijske mjere
    scores = np.dot(input_data, weights) + bias     # N x 1

    # vjerojatnosti razreda c_1
    def sigmoid(x): return np.exp(x) / (1 + np.exp(x))
    return np.array(sigmoid(scores))  # N x 1


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify


if __name__ == "__main__":
    np.random.seed(RAND_SEED)

    # get the training dataset
    input_data, true_labels = data.sample_gauss_2d(2, 200)
    plt.figure(figsize=(10, 7))
    plt.scatter(input_data[:, 0], input_data[:, 1],
                c=true_labels, cmap='plasma', s=100, alpha=0)
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
    probs_c1 = binlogreg_classify(input_data, weights, bias)
    # print('\nprobs_c1:\n', probs_c1)

    predicted_labels = np.copy(probs_c1)
    # predicted_labels = predicted_labels > 0.5
    for ind in range(len(predicted_labels)):
        if predicted_labels[ind] > 0.5:
            predicted_labels[ind] = 1
        else:
            predicted_labels[ind] = 0

    ''' DIJAGNOSTICKI ISPIS
    print('true_labels:\n', true_labels)
    print('predicted_labels:\n', predicted_labels)
    '''

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(
        predicted_labels, true_labels)
    AP = data.eval_AP(true_labels[probs_c1.argsort()])
    print('\nAccuracy: ', accuracy, '\nRecall: ', recall,
          '\nPrecision: ', precision, '\nAP: ', AP)

    # graph the decision surface
    decfun = binlogreg_decfun(weights, bias)
    bbox = (np.min(input_data, axis=0), np.max(input_data, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(input_data, true_labels, predicted_labels)

    # shot the plot
    plt.show()
