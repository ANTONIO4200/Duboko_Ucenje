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


def eval_perf_binary(predicted_labels, true_labels):
    TP = 0  # True Positive
    TN = 0  # True Negative
    FP = 0  # False Positive
    FN = 0  # False Negative
    # iteriraj po tocnim i predvidenim razredima i povecavaj brojace
    for predvideno, tocno in zip(predicted_labels, true_labels):
        if tocno == 1:
            if predvideno == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predvideno == 1:
                FP += 1
            else:
                TN += 1

    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    return Accuracy, Recall, Precision


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
    plt.figtext(.2, .95, "razred 0 = sivo")
    plt.figtext(.2, .90, "razred 1 = bijelo")
    plt.figtext(.5, .95, "tocno klasificiran = krug")
    plt.figtext(.5, .90, "netocno klasificiran = kvadrat")
    for ind in range(len(input_data)):
        if true_labels[ind] == 0:
            # obojaj sivo
            if predicted_labels[ind] == 0:
                # oznaci krugom
                plt.scatter(input_data[ind, 0], input_data[ind, 1],
                            c='gray', marker='o', edgecolors='black')
            else:
                # oznaci kvadratom
                plt.scatter(input_data[ind, 0], input_data[ind, 1],
                            c='gray', marker='s', edgecolors='black')
        else:
            # obojaj bijelo
            if predicted_labels[ind] == 1:
                # oznaci krugom
                plt.scatter(input_data[ind, 0], input_data[ind, 1],
                            c='white', marker='o', edgecolors='black')
            else:
                # oznaci kvadratom
                plt.scatter(input_data[ind, 0], input_data[ind, 1],
                            c='white', marker='s', edgecolors='black')


def myDummyDecision(input_data):
    scores = input_data[:, 0] + input_data[:, 1] - 10
    return scores


'''
  fun    ... decizijska funkcija (Nx2)->(Nx1)
  rect   ... željena domena prikaza zadana kao:
             ([x_min,y_min], [x_max,y_max])
  offset ... "nulta" vrijednost decizijske funkcije na koju 
             je potrebno poravnati središte palete boja;
             tipično imamo:
             offset = 0.5 za probabilističke modele 
                (npr. logistička regresija)
             offset = 0 za modele koji ne spljošćuju
                klasifikacijske mjere (npr. SVM)
  width,height ... rezolucija koordinatne mreže
'''


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    lsWidth = np.linspace(rect[0][0], rect[1][0], width)
    lsHeight = np.linspace(rect[0][1], rect[1][1], height)
    xx, yy = np.meshgrid(lsWidth, lsHeight)
    meshgrid = np.stack((xx.flatten(), yy.flatten()), axis=1)
    values = function(meshgrid).reshape((width, height))

    delta = offset if offset else 0
    maxval = max(np.max(values)-delta, -(np.min(values)-delta))

    plt.pcolormesh(xx, yy, values,
                   # vmin=delta-maxval, vmax=delta+maxval)
                   vmin=delta-maxval, vmax=delta+maxval)

    if offset is not None:
        plt.contour(xx, yy, values,
                    colors='black', levels=[offset])


if __name__ == "__main__":

    np.random.seed(156215)

    # get the training dataset
    input_data, true_labels = sample_gauss_2d(2, 150)

    # get the class predictions
    predicted_labels = myDummyDecision(input_data) > 0.5

    '''
    plt.scatter(input_data[:, 0], input_data[:, 1])
    plt.show()
    '''

    bbox = (np.min(input_data, axis=0), np.max(input_data, axis=0))
    graph_surface(myDummyDecision, bbox)

    # graph the data points
    graph_data(input_data, true_labels, predicted_labels)

    plt.show()
