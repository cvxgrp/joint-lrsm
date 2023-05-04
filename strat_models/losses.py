import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics, preprocessing
from scipy.stats import bernoulli
import cvxpy as cp


class Loss:
    """
    Inputs:
        N/A

    All losses have an attribute of isDistribution, which is a Boolean
    that denotes whether or not a Loss is a distribution estimate
    (i.e., isDistribution==True -> accepts Y,Z, and
            isDistribution==False -> accepts X,Y,Z.)

    All losses implement the following functions:

    1. evaluate(theta, data). Evaluates the regularizer at theta with data.
    2. prox(t, nu, data, warm_start, pool): Evaluates the proximal operator of the regularizer at theta
    """

    def __init__(self):
        pass

    def evaluate(self, theta):
        raise NotImplementedError(
            'This method is not implemented for the parent class.')

    def setup(self, data, K):
        """This function has any important setup required for the problem."""
        raise NotImplementedError(
            'This method is not implemented for the parent class.')

    def prox(self, t, nu, data, warm_start, pool):
        raise NotImplementedError(
            'This method is not implemented for the parent class.')

    def grad(self, theta, pool, cache):
        raise NotImplementedError(
            'This method is not implemented for the parent class.')

    def anll(self, data, G):
        return -np.mean(self.logprob(data, G))

    def report(self, data, G):
        raise NotImplementedError(
            'This method is not implemented for the parent class.')


def turn_into_iterable(x):
    try:
        iter(x)
    except TypeError:
        return [x]
    else:
        return x


def find_solution(x):
    """Finds the real solution to ax^3 + bx^2 + cx + d = 0."""
    roots = np.roots(x)
    for root in roots:
        if np.isreal(root) and root >= 1e-4 and root <= 1 - 1e-4:
            return np.real(root)
    return 0.5


def log_reg_prox(XY, nu, theta, t):
    if XY is None:
        return nu

    X, Y = XY

    nu_tch = torch.from_numpy(nu)
    theta_i = torch.from_numpy(theta).requires_grad_(True)
    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    optim = torch.optim.LBFGS([theta_i], lr=1, max_iter=50)

    def closure():
        optim.zero_grad()
        ls = t * loss(X @ theta_i, Y) + 0.5 * torch.sum((theta_i - nu_tch)**2)
        ls.backward()
        return ls

    optim.step(closure)
    return theta_i.data.numpy()


def quant_prox(XY, tau, nu, theta, t):
    if XY is None:
        return nu

    X, Y = XY
    Y = Y.flatten()
    theta_i = cp.Variable(nu.shape)
    theta_i.value = theta

    diff = Y - X @ theta_i
    loss1 = cp.sum(cp.pos(diff)) - (1-tau) * cp.sum(diff)
    loss2 = 0.5 * cp.norm(theta_i-nu)**2 / t
    loss = loss1 + loss2
    problem = cp.Problem(cp.Minimize(loss))
    problem.solve(solver=cp.MOSEK, warm_start=True)

    return np.array(theta_i.value)


def bce_grad(Y, N, theta):
    if Y is None:
        return np.zeros_like(theta)
    theta = np.clip(theta, 1e-5, 1 - 1e-5)

    g = -(Y / theta - (N - Y) / (1 - theta))
    return np.array(g)


def log_reg_grad(XY, theta):
    if XY is None:
        return np.zeros_like(theta)

    X, Y = XY
    scores = (X.unsqueeze(-1) * theta).sum(1)
    prob = F.softmax(scores, dim=1)
    Y_ = torch.zeros_like(prob)
    for i in range(len(Y)):
        Y_[i, Y[i]] = 1.

    g = X.T @ (prob - Y_)
    return np.array(g)


def quant_grad(XY, tau, theta, delta=1e-4):
    if XY is None:
        return np.zeros_like(theta)

    X, Y = XY

    Z = Y - X @ theta
    gZ = torch.zeros_like(Z)
    gZ[(Z > 0) & (Z <= delta)] /= delta
    gZ[Z > delta] = 1.
    gZ -= 1 - tau

    g = -X.T @ gZ
    return np.array(g)


# Losses
class sum_squares_loss(Loss):
    """
    f(theta) = ||X @ theta - Y||_2^2
    """

    def __init__(self, intercept=False):
        super().__init__()
        self.isDistribution = False
        self.intercept = intercept

    def evaluate(self, theta, data, prob=True):
        assert 'X' in data and 'Y' in data and 'Cls' in data

        X = torch.from_numpy(data['X'])

        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

        pred = (X * theta[data['Cls']]).sum(1)

        return sum((torch.from_numpy(data['Y']) - pred)**2)

    def setup(self, data, G):
        X = data['X']
        Y = data['Y']
        Z = data['Z']

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        _, n = X.shape
        # _, m = Y.shape

        if self.intercept:
            n = n + 1

        K = len(G.nodes())

        shape = (n, )
        theta_shape = (K, ) + shape

        for x, y, z in zip(X, Y, Z):
            vertex = G._node[z]
            if 'X' in vertex:
                vertex['X'] += [x]
                vertex['Y'] += [y]
            else:
                vertex['X'] = [x]
                vertex['Y'] = [y]

        XtX = torch.zeros(K, n, n).double()
        XtY = torch.zeros(K, n).double()
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                X = torch.tensor(np.array(vertex['X'])).double()
                Y = torch.tensor(np.array(vertex['Y'])).double()

                if self.intercept:
                    X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)],
                                  1)

                XtX[i] = X.t() @ X
                XtY[i] = X.t() @ Y
                del vertex['X']
                del vertex['Y']

        cache = {
            'XtX': XtX,
            'XtY': XtY,
            'K': K,
            'n': n,
            'theta_shape': theta_shape,
            'shape': shape
        }
        return cache

    def grad(self, theta, pool, cache):
        XtX = cache['XtX']
        XtY = cache['XtY']
        theta = torch.tensor(theta).unsqueeze(-1)
        g = torch.matmul(XtX, theta).squeeze(-1) - XtY
        return np.array(g)

    def prox(self, t, nu, warm_start, pool, cache):
        # raise NotImplementedError('This method is not yet done!!!')

        XtX = cache['XtX']
        XtY = cache['XtY']
        n = cache['n']

        A_LU = torch.lu(XtX + 1. /
                        (2 * t) * torch.eye(n).unsqueeze(0).double())
        b = XtY + 1. / (2 * t) * torch.from_numpy(nu)
        x = torch.lu_solve(b, *A_LU)

        return x.numpy()

    def predict(self, data, G):
        X = torch.from_numpy(data['X'])
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        pred = (X * theta).sum(1).numpy()
        return pred

    def log_prob(self, data, G):
        X = torch.from_numpy(data['X'])
        Y = data['Y']
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        pred = (X.unsqueeze * theta).sum(1).numpy()

        log_prob = -metrics.mean_squared_error(Y, pred)
        return log_prob

    def report(self, data, G):
        X = torch.from_numpy(data['X'])
        Y = data['Y']
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        pred = (X * theta).sum(1).numpy()

        rmse = np.sqrt(metrics.mean_squared_error(Y, pred))
        mae = metrics.mean_absolute_error(Y, pred)

        report = {
            'RMSE': rmse,
            'MAE': mae,
        }
        return report


class logistic_loss(Loss):
    """
    f(theta) = sum[ log(1 + exp{-Y * theta @ X} )  ]
    """

    def __init__(self, intercept=False):
        super().__init__()
        self.isDistribution = False
        self.intercept = intercept

    def evaluate(self, theta, data, prob=True):
        assert 'X' in data and 'Y' in data and 'Cls' in data

        X = torch.from_numpy(data['X'])
        Y = torch.from_numpy(self.le.transform(data['Y'].flatten()))
        # Y = Y.flatten()

        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

        scores = (X.unsqueeze(-1) * theta[data['Cls']]).sum(1)

        if prob:
            loss = F.cross_entropy(scores, Y, reduction='sum')
            return loss.numpy()
        else:
            y_pred = torch.argmax(scores, 1).numpy()
            acc = metrics.accuracy_score(Y, y_pred)
            return -acc

    def cvx_loss(self, theta, data, prob=True):
        assert 'X' in data and 'Y' in data and 'Cls' in data

        X = torch.from_numpy(data['X'])
        Y = torch.from_numpy(self.le.transform(data['Y'].flatten()))

        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

        scores = cp.sum(X.unsqueeze(-1) * theta[data['Cls']], axis=1)
        loss = F.cross_entropy(scores, Y, reduction='sum')
        return loss.numpy()

    def setup(self, data, G):
        X = data['X']
        Y = data['Y']
        Z = data["Z"]

        self.le = preprocessing.LabelEncoder()
        Y = self.le.fit_transform(Y.flatten()).copy()
        num_classes = len(self.le.classes_)

        K = len(G.nodes())
        n = X.shape[1]

        if self.intercept:
            n = n + 1

        shape = (n, num_classes)
        theta_shape = (K, ) + shape

        for x, y, z in zip(X, Y, Z):
            vertex = G._node[z]
            if 'X' in vertex:
                vertex['X'] += [x]
                vertex['Y'] += [y]
            else:
                vertex['X'] = [x]
                vertex['Y'] = [y]

        XY_data = []
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                X, Y = torch.tensor(np.array(vertex['X'])), torch.tensor(
                    np.array(vertex['Y']))
                if self.intercept:
                    X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)],
                                  1)
                XY_data += [(X, Y)]
                del vertex['X']
                del vertex['Y']
            else:
                XY_data += [None]

        cache = {
            "XY": XY_data,
            'n': n,
            'theta_shape': theta_shape,
            'shape': shape,
            'K': K
        }
        return cache

    def grad(self, theta, pool, cache):
        XY = cache['XY']
        gs = pool.starmap(log_reg_grad, zip(XY, theta))

        return np.array(gs)

    def prox(self, t, nu, warm_start, pool, cache):
        res = pool.starmap(
            log_reg_prox,
            zip(cache['XY'], nu, warm_start, t * np.ones(cache['K'])))
        return np.array(res)

    def logprob(self, data, G):
        Y = torch.from_numpy(self.le.transform(data['Y'].flatten()))
        s = self.scores(data, G)
        log_prob = -F.cross_entropy(s, Y, reduction='none')
        return log_prob.numpy()

    def scores(self, data, G):
        X = torch.from_numpy(data['X'])
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        scores = (X.unsqueeze(-1) * theta).sum(1)
        return scores

    def predict(self, data, G):
        s = self.scores(data, G)
        y_pred = self.le.inverse_transform(torch.argmax(s, 1).numpy())
        return y_pred

    def report(self, data, G):
        X = torch.from_numpy(data['X'])
        Y = torch.from_numpy(self.le.transform(data['Y'].flatten()))
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        s = (X.unsqueeze(-1) * theta).sum(1)
        prob_pred = F.softmax(s, dim=0)
        y_pred = torch.argmax(s, 1).numpy()
        Y_ = torch.zeros_like(prob_pred)
        for i in range(len(Y)):
            Y_[i, Y[i]] = 1.

        acc = metrics.accuracy_score(Y, y_pred)
        log_prob = -F.cross_entropy(s, Y, reduction='none')
        auc = metrics.roc_auc_score(Y_, prob_pred, average='micro')
        f1 = metrics.f1_score(Y, y_pred, average='macro')

        report = {
            'Accuracy': acc,
            'avg_ll': log_prob.numpy().mean(),
            'AUC': auc,
            'F_1': f1
        }
        return report


class bernoulli_loss(Loss):
    """
    f(theta) = -sum(y)log(theta) - (n - sum(y))log(1-theta),
    where y in reals^n and theta in [0,1].
    """

    def __init__(self, min_theta=1e-5, max_theta=1 - 1e-5):
        super().__init__()
        self.isDistribution = True
        self.min_theta = min_theta
        self.max_theta = max_theta

    def evaluate(self, theta, data, prob=True):
        assert 'Y' in data and 'Cls' in data

        Y = torch.tensor(np.array(data['Y']))
        Theta = torch.tensor(np.array(theta[data['Cls']]).flatten())

        if prob:
            loss = F.binary_cross_entropy(Theta, Y, reduction='sum')
            return loss.numpy()
        else:
            y_pred = np.round(Theta)
            acc = metrics.accuracy_score(Y, y_pred)
            return -acc

    def setup(self, data, G):
        Y = data["Y"]
        Z = data["Z"]

        K = len(G.nodes())

        shape = (1, )
        theta_shape = (K, ) + shape

        for y, z in zip(Y, Z):
            vertex = G._node[z]
            if "Y" in vertex:
                vertex["Y"] += [y]
            else:
                vertex["Y"] = [y]

        S = np.zeros((K, 1))
        N = np.zeros((K, 1))

        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                S[i] = np.sum(vertex['Y'])
                N[i] = len(vertex['Y'])
                del vertex['Y']

        cache = {
            "S": S,
            "N": N,
            "theta_shape": theta_shape,
            "shape": shape,
            "K": K
        }
        return cache

    def grad(self, theta, pool, cache):
        S = cache["S"]
        N = cache["N"]

        gs = pool.starmap(bce_grad, zip(S, N, theta))

        return np.array(gs)

    def prox(self, t, nu, warm_start, pool, cache):
        S = cache["S"]
        N = cache["N"]

        a = -1 * np.ones(nu.shape)
        b = (1 + nu)
        c = t * N - nu
        d = -t * S

        coefs = np.hstack([a, b, c, d])
        theta = np.array(pool.map(find_solution, coefs))[:, np.newaxis]

        return np.clip(theta, self.min_theta, self.max_theta)

    def logprob(self, data, G):
        Y = turn_into_iterable(data["Y"])
        Z = turn_into_iterable(data["Z"])
        parameter = [G._node[z]["theta"][0] for z in Z]
        return bernoulli.logpmf(Y, p=parameter)

    def sample(self, data, G):
        Z = turn_into_iterable(data["Z"])
        parameter = [G._node[z]["theta"][0] for z in Z]
        return bernoulli.rvs(p=parameter)

    def report(self, data, G):
        Y = turn_into_iterable(data["Y"])
        Z = turn_into_iterable(data["Z"])
        parameter = [G._node[z]["theta"][0] for z in Z]
        y_pred = np.round(parameter)

        log_prob = bernoulli.logpmf(Y, p=parameter)
        acc = metrics.accuracy_score(Y, y_pred)
        auc = metrics.roc_auc_score(Y, y_pred)
        f1 = metrics.f1_score(Y, y_pred)

        report = {
            'Accuracy': acc,
            'avg_ll': log_prob.mean(),
            'AUC': auc,
            'F_1': f1
        }
        return report


class quantile_loss(Loss):
    """
    f(theta) = sum[ (1 - tau) * (Y - X @ theta)^- + tau * (Y - X @ theta)^+ ]
    """

    def __init__(self, quantile, intercept=False):
        super().__init__()
        self.isDistribution = False
        self.tau = quantile
        self.intercept = intercept

    def evaluate(self, theta, data, prob=True):
        assert 'X' in data and 'Y' in data and 'Cls' in data

        X = torch.from_numpy(data['X'])

        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

        pred = (X * theta[data['Cls']]).sum(1)
        diff = torch.from_numpy(data['Y']) - pred

        diff[diff > 0] *= self.tau
        diff[diff < 0] *= (self.tau - 1)
        loss = sum(diff)
        return loss

    def cvx_loss(self, theta, data, prob=True):
        assert 'X' in data and 'Y' in data and 'Cls' in data

        X = torch.from_numpy(data['X'])

        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)

        Xtheta = cp.multiply(X, theta[data['Cls']])
        pred = cp.sum(Xtheta, axis=1)
        diff = data['Y'] - pred

        loss = cp.sum(self.tau * cp.pos(diff) + (1-self.tau)*cp.neg(diff))
        return loss

    def setup(self, data, G):
        X = data['X']
        Y = data['Y']
        Z = data["Z"]

        K = len(G.nodes())
        n = X.shape[1]

        if self.intercept:
            n = n + 1

        shape = (n, )
        theta_shape = (K, ) + shape

        for x, y, z in zip(X, Y, Z):
            vertex = G._node[z]
            if 'X' in vertex:
                vertex['X'] += [x]
                vertex['Y'] += [y]
            else:
                vertex['X'] = [x]
                vertex['Y'] = [y]

        XY_data = []
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                X, Y = torch.tensor(np.array(vertex['X'])), torch.tensor(
                    np.array(vertex['Y']))
                if self.intercept:
                    X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)],
                                  1)
                XY_data += [(X, Y)]
                del vertex['X']
                del vertex['Y']
            else:
                XY_data += [None]

        cache = {
            "XY": XY_data,
            'n': n,
            'theta_shape': theta_shape,
            'shape': shape,
            'K': K
        }
        return cache

    def grad(self, theta, pool, cache, delta=1e-4):
        # grad of huber loss
        XY = cache['XY']
        gs = pool.starmap(
            quant_grad,
            zip(XY, [self.tau] * cache['K'], theta, [delta] * cache['K']))

        return np.array(gs)

    def prox(self, t, nu, warm_start, pool, cache):
        res = pool.starmap(
            quant_prox,
            zip(cache['XY'], [self.tau] * cache['K'], nu, warm_start,
                t * np.ones(cache['K'])))
        return np.array(res)

    def predict(self, data, G):
        X = torch.from_numpy(data['X'])
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        pred = (X * theta).sum(1).numpy()
        return pred

    def log_prob(self, data, G):
        raise NotImplementedError(
            'This method is not implemented for the quantile loss class.')

    def report(self, data, G):
        X = torch.from_numpy(data['X'])
        Y = data['Y']
        if self.intercept:
            X = torch.cat([X, torch.ones_like(X[:, 0]).unsqueeze(1)], 1)
        theta = torch.tensor(np.array([G._node[z]['theta']
                                       for z in data['Z']]))
        y_pred = (X * theta).sum(1).numpy()

        pinball = metrics.mean_pinball_loss(Y, y_pred, alpha=self.tau)

        report = {
            'pinball': pinball,
        }
        return report
