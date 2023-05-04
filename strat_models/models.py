import numpy as np
import networkx as nx
from strat_models.fit import *


def G_to_data(G, theta_shape):
    theta_init = np.zeros(theta_shape)
    W0 = nx.to_scipy_sparse_matrix(G).copy()

    W_init = W0

    data = {'theta_init': theta_init, 'W_init': W_init, 'W0': W0.copy()}
    return data


def transfer_result_to_G(result, G):
    """Puts solution vectors into a graph G"""
    theta = result['theta']
    for i, node in enumerate(G.nodes()):
        vertex = G._node[node]
        vertex['theta'] = theta[i]

    return None


class BaseModel:

    def __init__(self, loss, reg):
        self.loss = loss
        self.local_reg = reg


class StratifiedModel:

    def __init__(self, BaseModel: BaseModel, graph, config):
        self.change_base_model(BaseModel, config)
        self.G = graph

    def change_base_model(self, base_model, config):
        """
        Alters/edits the Basemodel inside the StratifiedModel
        and updates all relevant attributes.
        """
        self.base_model = base_model

        self.loss = base_model.loss
        self.isDistribution = base_model.loss.isDistribution

        self.local_reg = base_model.local_reg
        self.lambd = base_model.local_reg.lambd

        self.config = config

    def fit(self, data_train, data_val, alg, **kwargs):
        cache = self.loss.setup(data_train, self.G)

        # loss_fn
        def l_fn(theta):
            return self.loss.evaluate(theta, data_train)

        r_fn = self.local_reg.evaluate

        def l_grad(theta, pool):
            return self.loss.grad(theta, pool, cache)

        def val_scores(theta):
            return self.loss.evaluate(theta, data_val, prob=True)

        def l_prox(t, nu, warm_start, pool):
            return self.loss.prox(t, nu, warm_start, pool, cache)

        r_prox = self.local_reg.prox

        G_data = G_to_data(self.G, cache['theta_shape'])

        if alg == 'MAPG':
            mapg = MAPG(l_fn, r_fn, val_scores, cache['K'], cache['shape'],
                        G_data, self.config)
            result, info = mapg.fit(l_grad, r_prox, **kwargs)
        elif alg == 'LogDiag':
            ld = LogDiag(l_fn, r_fn, val_scores, cache['K'], cache['shape'],
                         G_data, self.config)
            result, info = ld.fit(l_grad, r_prox, **kwargs)
        elif alg == 'TrConstraint':
            trcnstr = TrConstraint(l_fn, r_fn, val_scores, cache['K'],
                                   cache['shape'], G_data, self.config)
            result, info = trcnstr.fit(l_grad, r_prox, **kwargs)
        elif alg == 'TrConstraintBi':
            trcnstrbi = TrConstraintBi(l_fn, r_fn, val_scores, cache['K'],
                                       cache['shape'], G_data, self.config)
            result, info = trcnstrbi.fit(l_prox, r_prox, **kwargs)
        elif alg == 'LogDiagBi':
            ldbi = LogDiagBi(l_fn, r_fn, val_scores, cache['K'],
                             cache['shape'], G_data, self.config)
            result, info = ldbi.fit(l_prox, r_prox, **kwargs)
        elif alg == 'Org':
            L = nx.laplacian_matrix(self.G)
            org = Alg(l_fn, r_fn, val_scores, cache['K'], cache['shape'],
                      G_data, self.config)
            result, info = org.fit(L, l_prox, r_prox, **kwargs)

        transfer_result_to_G(result, self.G)
        return info

    def scores(self, data):
        return self.base_model.loss.scores(data, self.G)

    def anll(self, data):
        return self.base_model.loss.anll(data, self.G)

    def predict(self, data):
        return self.base_model.loss.predict(data, self.G)

    def report(self, data):
        return self.base_model.loss.report(data, self.G)

    def sample(self, data):
        if not self.isDistribution:
            raise NotImplementedError("This model is not a distribution.")
            return None
        else:
            return self.base_model.loss.sample(data, self.G)
