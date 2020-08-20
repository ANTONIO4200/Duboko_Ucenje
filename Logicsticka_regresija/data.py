import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    def __init__(self):
        minx = 0
        maxx = 10
        miny = 0
        maxy = 10
        self.mux = np.random.uniform(minx, maxx)
        self.muy = np.random.uniform(miny, maxy)

        eigvalx = (np.random.random_sample() * (maxx - minx) / 5) ** 2
        eigvaly = (np.random.random_sample() * (maxy - miny) / 5) ** 2

        D = [[eigvalx, 0],
             [0, eigvaly]]

        theta = np.random.random_sample() * 2 * np.pi

        R = [[np.cos(theta), -np.sin(theta)],
             [np.sin(theta), np.cos(theta)]]

        self.Sigma = np.matmul(np.matmul(np.transpose(R), D), R)

    def get_sample(self, n):
        return np.random.multivariate_normal(
            [self.mux, self.muy], self.Sigma, n)


def sample_gauss_2d(C, N):
    '''
    Argumenti
        C: broj stvorenih razdioba
        N: broj uzrokovanih podataka iz razdiobe

    Povratne vrijednosti
        X: dimenzija (N*C)x2
        Y: (N*C)x1
    '''
    X = np.empty((0, 2))
    Y = np.empty(N*C)
    for i in range(C):
        Y[i*N: (i+1)*N] = i
        G = Random2DGaussian()
        X_temp = G.get_sample(N)
        X = np.append(X, X_temp, axis=0)
    return X, Y


def eval_perf_multi(predicted_labels, true_labels):
    # iteriraj po tocnim i predvidenim razredima i povecavaj brojace
    noClasses = max(true_labels) + 1
    conf_matrix = np.zeros((noClasses, noClasses), dtype=int)

    for pred, true in zip(predicted_labels, true_labels):
        conf_matrix[true][pred] += 1

    Accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
    Precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    Recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    return Accuracy, Recall, Precision, conf_matrix


def eval_AP(labels):

    n = len(labels)
    pos = sum(labels)
    neg = n - pos

    tp = pos
    tn = 0
    fn = 0
    fp = neg

    sumprec = 0
    # IPython.embed()
    for x in labels:
        precision = tp / (tp + fp)

        if x:
            sumprec += precision

        # print (x, tp,tn,fp,fn, precision, recall, sumprec)
        # IPython.embed()

        tp -= x
        fn += x
        fp -= not x
        tn += not x

    return sumprec/pos


def graph_data(input_data, true_labels, predicted_labels):
    '''
    input_data - podatci (np.array dimenzija Nx2)
    true_labels - tocni indeksi razreda podataka (Nx1)
    predicted_labels - predvideni indeksi razreda podataka (Nx1)
    '''
    no_class = max(true_labels) + 1
    plt.figtext(.5, .95, "tocno klasificiran = krug")
    plt.figtext(.5, .90, "netocno klasificiran = kvadrat")
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, no_class)]

    for ind in range(len(input_data)):
        true = true_labels[ind]
        pred = predicted_labels[ind]
        color = colors[true]

        if true == pred:
            plt.scatter(input_data[ind, 0], input_data[ind, 1],
                        c=np.array([color]), marker='o', edgecolors='black')
        else:
            plt.scatter(input_data[ind, 0], input_data[ind, 1],
                        c=np.array([color]), marker='s', edgecolors='black')


def graph_surface_1(function, rect, offset=0.5, width=256, height=256):
    #width = int(round(rect[1][0] - rect[0][0]))
    #height = int(round(rect[1][1] - rect[0][1]))
    lsWidth = np.linspace(rect[0][0], rect[1][0], width)
    lsHeight = np.linspace(rect[0][1], rect[1][1], height)
    xx, yy = np.meshgrid(lsWidth, lsHeight)
    meshgrid = np.stack((xx.flatten(), yy.flatten()), axis=1)
    logreg_return = function(meshgrid)
    predicted_class = np.argmax(logreg_return, axis=1).reshape(width, height)

    plt.pcolormesh(predicted_class)


def graph_surface_2(function, rect, offset=0.5, width=256, height=256):
    # CHANGE THIS FOR DIFFERENT CLASS
    Class_No = 0
    lsWidth = np.linspace(rect[0][0], rect[1][0], width)
    lsHeight = np.linspace(rect[0][1], rect[1][1], height)
    xx, yy = np.meshgrid(lsWidth, lsHeight)
    meshgrid = np.stack((xx.flatten(), yy.flatten()), axis=1)
    logreg_return = function(meshgrid)
    values = logreg_return[:, Class_No].reshape((width, height))

    delta = offset if offset else 0
    maxval = max(np.max(values)-delta, -(np.min(values)-delta))

    plt.pcolormesh(xx, yy, values,
                   # vmin=delta-maxval, vmax=delta+maxval)
                   vmin=delta-maxval, vmax=delta+maxval)

    if offset is not None:
        plt.contour(xx, yy, values,
                    colors='black', levels=[offset])
