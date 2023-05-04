import numpy as np
import scipy
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import multiprocessing as mp
import time
import matplotlib.pylab as plt
import logging
import os
import cvxpy as cp


def Lap(weight_matrix):
    """
    weight_matrix should be a sparse matrix in scipy.sparse
    """

    W = sparse.lil_matrix(weight_matrix)
    dias = sparse.dia_matrix((np.sum(W, axis=1).flatten(), 0),
                             shape=weight_matrix.shape)
    L = dias - W
    return L


def Lap_diff(L):
    W = L.diagonal() - L
    return W


class Alg:

    def __init__(self,
                 l_fn,
                 r_fn,
                 val_scores,
                 K,
                 shape,
                 G_data=dict(),
                 config=dict()):
        """
        Parameters:
            - l_fn: local loss function
            - r_fn: local regularizer
            - val_scores: validation metric
            - K: number of strata
            - shape: shape of theta
            - G_data (optional): dictionary of warm starting values (default=dict())
            - config: some algorithm-dependent hyperparameters (default=dict())
        """
        self.lambd_1 = config['lambd_1']
        self.lambd_2 = config['lambd_2']
        self.eta = config['eta']
        self.mu = config['mu']
        self.l_fn, self.r_fn = l_fn, r_fn
        self.val_scores = val_scores
        self.K, self.shape = K, shape
        self.n = np.prod(self.shape)
        self.G_data = G_data

    def loss(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = np.sum(np.array(W.todense()) * pairwise_dist) / 4
        _, logdet = np.linalg.slogdet(self.mu * np.eye(self.K) + Lap(W))
        R_1 = -logdet
        R_2 = 0.5 * (1 - self.eta) * sparse.linalg.norm(
            W - self.G_data['W0'], ord='fro')**2 + self.eta * np.sum(np.abs(W))

        total_loss = self.l_fn(theta) + np.sum(self.r_fn(
            theta)) + lap_reg + self.lambd_1 * R_1 + self.lambd_2 * R_2
        return total_loss

    def W_grad(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        grad_1 = squareform(sq_dist)**2 / 4.

        grad_2 = self.lambd_2 * (1 - self.eta) * (W - self.G_data['W0'])

        L1 = self.mu * np.eye(self.K) + Lap(W)
        inv_L1 = scipy.linalg.inv(L1)
        grad_3 = -self.lambd_1 * Lap_diff(inv_L1)

        return grad_1 + grad_2 + grad_3

    def fit(self,
            L,
            l_prox,
            r_prox,
            abs_tol=1e-3,
            rel_tol=1e-3,
            rho=1,
            mu=10,
            tau_incr=2,
            tau_decr=2,
            max_rho=1e2,
            min_rho=1e-1,
            maxiter=500,
            verbose=False,
            n_jobs=1,
            max_cg_iterations=10,
            figpath=None):
        """
        ADMM implementation, copied from the paper `A Distributed Method for Fitting Laplacian Regularized Stratified Models`.
        """

        # Initialization
        if 'theta_init' in self.G_data:
            theta = self.G_data['theta_init'].copy()
        else:
            theta = np.zeros((self.K, ) + self.shape)
        if 'theta_tilde' in self.G_data:
            theta_tilde = self.G_data['theta_tilde'].copy()
        else:
            theta_tilde = theta.copy()
        if 'theta_hat' in self.G_data:
            theta_hat = self.G_data['theta_hat'].copy()
        else:
            theta_hat = theta.copy()
        if 'u' in self.G_data:
            u = self.G_data['u'].copy()
        else:
            u = np.zeros(theta.shape)
        if 'u_tilde' in self.G_data:
            u_tilde = self.G_data['u_tilde'].copy()
        else:
            u_tilde = np.zeros(theta.shape)

        res_pri = np.zeros(theta.shape)
        res_pri_tilde = np.zeros(theta.shape)
        res_dual = np.zeros(theta.shape)
        res_dual_tilde = np.zeros(theta.shape)

        optimal = False
        n_jobs = n_jobs if self.K > n_jobs else self.K
        prox_pool = mp.Pool(n_jobs)
        LOSS = []

        if verbose:
            logging.info("%3s | %10s %10s %10s %10s %6s %6s %6s %6s" %
                         ("it", "s_norm", "r_norm", "eps_pri", "eps_dual",
                          "rho", "time1", "time2", "time3"))

        # Main ADMM loop
        start_time = time.perf_counter()
        for t in range(1, maxiter + 1):
            # theta update
            start_time_1 = time.perf_counter()
            theta = l_prox(1. / rho, theta_hat - u, theta, prox_pool)
            time_1 = time.perf_counter() - start_time_1

            # theta_tilde update
            start_time_2 = time.perf_counter()
            theta_tilde = r_prox(1. / rho, theta_hat - u_tilde, theta_tilde,
                                 prox_pool)
            time_2 = time.perf_counter() - start_time_2

            # theta_hat update
            start_time_3 = time.perf_counter()
            sys = L + 2 * rho * sparse.eye(self.K)
            M = sparse.diags(1. / sys.diagonal())
            indices = np.ndindex(self.shape)
            rhs = rho * (theta.T + u.T + theta_tilde.T + u_tilde.T)
            for i, ind in enumerate(indices):
                index = ind[::-1]
                sol = sparse.linalg.cg(sys,
                                       rhs[index],
                                       M=M,
                                       x0=theta_hat.T[index],
                                       maxiter=max_cg_iterations)[0]
                res_dual.T[index] = -rho * (sol - theta_hat.T[index])
                res_dual_tilde.T[index] = res_dual.T[index]
                theta_hat.T[index] = sol
            time_3 = time.perf_counter() - start_time_3

            # u and u_tilde update
            res_pri = theta - theta_hat
            res_pri_tilde = theta_tilde - theta_hat
            u += theta - theta_hat
            u_tilde += theta_tilde - theta_hat

            loss = self.loss(theta, self.G_data['W0'])
            LOSS.append(loss)
            if t % 5 == 0:
                logging.info(f'{t}th iteration: loss = {loss: .4e}')

            # calculate residual norms
            res_pri_norm = np.linalg.norm(np.append(res_pri, res_pri_tilde))
            res_dual_norm = np.linalg.norm(np.append(res_dual, res_dual_tilde))

            eps_pri = np.sqrt(2 * self.K * np.prod(self.shape)) * abs_tol + \
                rel_tol * max(res_pri_norm, res_dual_norm)
            eps_dual = np.sqrt(2 * self.K * np.prod(self.shape)) * abs_tol + \
                rel_tol * np.linalg.norm(rho * np.append(u, u_tilde))

            if verbose:
                logging.info(
                    "%3d | %8.4e %8.4e %8.4e %8.4e %4.3f %4.3f %4.3f %4.3f" %
                    (t, res_pri_norm, res_dual_norm, eps_pri, eps_dual, rho,
                     time_1 * 1000, time_2 * 1000, time_3 * 1000))

            # check stopping condition
            if res_pri_norm <= eps_pri and res_dual_norm <= eps_dual:
                optimal = True
                break

            # penalty parameter update
            new_rho = rho
            if res_pri_norm > mu * res_dual_norm:
                new_rho = tau_incr * rho
            elif res_dual_norm > mu * res_pri_norm:
                new_rho = rho / tau_decr
            new_rho = np.clip(new_rho, min_rho, max_rho)
            u *= rho / new_rho
            u_tilde *= rho / new_rho
            rho = new_rho

        main_loop_time = time.perf_counter() - start_time

        # clean up the multiprocessing pool
        prox_pool.close()
        prox_pool.join()

        if verbose:
            if optimal:
                logging.info(f"Terminated (optimal) in {t} iterations.")
            else:
                logging.info("Terminated (reached max iterations).")
            logging.info("run time: %8.4e seconds" % main_loop_time)

        # construct result
        result = {
            'theta': theta,
            'theta_tilde': theta_tilde,
            'theta_hat': theta_hat,
            'u': u,
            'u_tilde': u_tilde,
            "W": self.G_data["W0"]
        }

        info = {'time': main_loop_time, 'iterations': t, 'optimal': optimal}

        plt.plot(range(t), LOSS)
        plt.yscale('symlog')
        plt.savefig(os.path.join(figpath, 'loss.jpg'), dpi=600)
        plt.close()

        return result, info


class MAPG(Alg):
    """
    MAPG implementation of joint learning method as described in the paper `Joint Graph Learning and Model Fitting in Laplacian Regularized Stratified Models`
    """

    def __init__(self,
                 l_fn,
                 r_fn,
                 val_scores,
                 K,
                 shape,
                 G_data=dict(),
                 config=dict()):
        super().__init__(l_fn, r_fn, val_scores, K, shape, G_data, config)

    def fit(self,
            l_grad,
            r_prox,
            alpha_x=1e-3,
            alpha_y=1e-3,
            res_tol=1e-4,
            max_iter=1000,
            patience=10,
            n_jobs=1,
            figpath=None):
        """
        Parameters:
            - l_grad: gradient oracle of local loss function
            - r_prox: proximal oracle of local regularizer
            - alpha_x: stepsize of proximal gradient descent (default=1e-3)
            - alpha_y: stepsize of accelerated proximal gradient descent (default=1e-3)
            - res_tol: threshold of stopping (default=1e-4)
            - max_iter: maximum number of iterations (default=1000)
            - patience: the maximum number of iterations to tolerate non-improvement (default=200)
            - n_jobs: number of jobs to spawn. (default=1)
            - figpath: path to save plots
        Returns:
            - result: Dictionary with the solution vectors
            - info: Information about the algorithm's performance
        """

        # Initialization
        if 'theta_init' in self.G_data:
            theta_x = self.G_data['theta_init'].copy()
        else:
            theta_x = np.zeros((self.K, ) + self.shape)
        old_theta_x = theta_x.copy()
        theta_z = theta_x.copy()
        if 'W_init' in self.G_data:
            W_x = self.G_data['W_init'].copy()
        else:
            W_x = sparse.eye(self.K)
        old_W_x = W_x.copy()
        W_z = W_x.copy()

        t_0 = 0.
        t_1 = 1.
        F_1 = self.loss(theta_x, W_x)

        n_jobs = n_jobs if n_jobs < self.K else self.K
        pool = mp.Pool(n_jobs)
        optimal = False
        LOSS = []
        best_k = 1
        best_val = np.inf
        best_theta, best_W = theta_x, W_x
        m_cnt = 0

        # MAPG loop
        start_time = time.perf_counter()
        for k in range(1, max_iter + 1):
            # compute y
            theta_y = theta_x + t_0 / t_1 * (theta_z - theta_x) + (
                t_0 - 1) / t_1 * (theta_x - old_theta_x)
            W_y = W_x + t_0 / t_1 * (W_z - W_x) + (t_0 - 1) / t_1 * (W_x -
                                                                     old_W_x)

            # compute z
            theta_z, W_z = self.prox_grad(theta_y, W_y, l_grad, r_prox,
                                          alpha_y, pool)
            F_z = self.loss(theta_z, W_z)

            # compute v
            theta_v, W_v = self.prox_grad(theta_x, W_x, l_grad, r_prox,
                                          alpha_x, pool)
            F_v = self.loss(theta_v, W_v)

            # update t
            t_0 = t_1
            t_1 = (np.sqrt(4 * t_0**2 + 1) + 1) / 2.

            # update x
            old_theta_x, old_W_x = theta_x, W_x

            if F_z < F_v:
                F_1 = F_z
                theta_x, W_x = theta_z, W_z
                m_cnt += 1
            else:
                F_1 = F_v
                theta_x, W_x = theta_v, W_v

            val_loss = self.val_scores(theta_x)
            if best_val > val_loss:
                best_val = val_loss
                best_theta, best_W = theta_x, W_x
                best_k = k

            LOSS.append(F_1)
            if k % 20 == 0:
                logging.info(
                    f'{k}th iteration: loss = {F_1: .4e}, val = {val_loss: .4e}, best iter = {best_k}'
                )

            # stopping criterion
            res = np.sum((theta_x - old_theta_x)**2) + np.sum(
                (W_x - old_W_x)**2)

            if res < res_tol:
                optimal = True
                break
            if k - best_k > patience:
                optimal = True
                break

        main_loop_time = time.perf_counter() - start_time

        # clean up the multiprocessing pool
        pool.close()
        pool.join()

        # construct the result
        result = {'theta': theta_x, 'W': W_x}
        # result = {'theta': best_theta, 'W': best_W}

        info = {'time': main_loop_time, 'iterations': k, 'optimal': optimal}

        plt.plot(range(k), LOSS)
        plt.yscale('symlog')
        plt.savefig(os.path.join(figpath, 'loss.jpg'), dpi=600)
        plt.close()

        return result, info

    def theta_grad(self, theta, W, l_grad, pool):
        grad_1 = l_grad(theta, pool)
        grad_2 = (Lap(W) @ theta.reshape(self.K, self.n)).reshape((self.K, ) +
                                                                  self.shape)
        return grad_1 + grad_2

    def prox_grad(self, theta, W, l_grad, r_prox, alpha, pool):
        grad_theta = self.theta_grad(theta, W, l_grad, pool)
        new_theta = r_prox(alpha, theta - alpha * grad_theta, theta, pool)

        grad_W = self.W_grad(theta, W)
        M = W - alpha * grad_W
        M = (M + M.T) / 2. - alpha * self.lambd_2 * self.eta
        M[M < 0] = 0
        new_W = sparse.lil_matrix(M)
        return new_theta, new_W


class LogDiag(MAPG):
    """
    MAPG implementation of Log-Diagonal method
    """

    def __init__(self,
                 l_fn,
                 r_fn,
                 val_scores,
                 K,
                 shape,
                 G_data=dict(),
                 config=dict()):
        super().__init__(l_fn, r_fn, val_scores, K, shape, G_data, config)

    def loss(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = np.sum(np.array(W.todense()) * pairwise_dist) / 4
        logdiag = np.log(self.mu + np.sum(W, axis=1)).sum()
        R_1 = -logdiag
        R_2 = 0.5 * (1 - self.eta) * sparse.linalg.norm(
            W - self.G_data['W0'], ord='fro')**2 + self.eta * np.sum(np.abs(W))

        total_loss = self.l_fn(theta) + np.sum(self.r_fn(
            theta)) + lap_reg + self.lambd_1 * R_1 + self.lambd_2 * R_2
        return total_loss

    def W_grad(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        grad_1 = squareform(sq_dist)**2 / 4.

        grad_2 = self.lambd_2 * (1 - self.eta) * (W - self.G_data['W0'])

        W1 = np.sum(W, axis=1)
        inv = np.repeat(1 / (W1 + self.mu), len(W1), axis=1)
        inv[np.diag_indices_from(inv)] = 0
        grad_3 = -self.lambd_1 * inv

        return grad_1 + grad_2 + grad_3


class LogDiagBi(Alg):
    """
    Alternating implementation of Log-Diagonal method
    """

    def __init__(self,
                 l_fn,
                 r_fn,
                 val_scores,
                 K,
                 shape,
                 G_data=dict(),
                 config=dict()):
        super().__init__(l_fn, r_fn, val_scores, K, shape, G_data, config)

    def loss(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = np.sum(np.array(W.todense()) * pairwise_dist) / 4
        logdiag = np.log(np.sum(W, axis=1)).sum()
        R_1 = -logdiag
        R_2 = 0.5 * sparse.linalg.norm(W, ord='fro')**2

        total_loss = self.l_fn(theta) + np.sum(self.r_fn(
            theta)) + lap_reg + self.lambd_1 * R_1 + self.lambd_2 * R_2
        return total_loss

    def fit(self,
            l_prox,
            r_prox,
            abs_tol=1e-3,
            rel_tol=1e-3,
            res_tol=1e-4,
            max_iter=1000,
            n_jobs=1,
            figpath=None):

        # Initialization
        if 'theta_init' in self.G_data:
            theta = self.G_data['theta_init'].copy()
        else:
            theta = np.zeros((self.K, ) + self.shape)
        old_theta = theta.copy()

        if 'W_init' in self.G_data:
            W = self.G_data['W_init'].copy()
        else:
            W = sparse.eye(self.K)
        old_W = W.copy()

        n_jobs = n_jobs if n_jobs < self.K else self.K
        optimal = False
        LOSS = []
        best_k = 1
        best_val = np.inf

        # BCD loop
        start_time = time.perf_counter()
        for k in range(1, max_iter + 1):

            # update theta
            old_theta = theta
            L = Lap(W)
            start_time1 = time.perf_counter()
            theta, state = self.update_theta(L,
                                             l_prox,
                                             r_prox,
                                             theta,
                                             abs_tol,
                                             rel_tol,
                                             n_jobs=n_jobs)
            time_1 = time.perf_counter() - start_time1

            # update W
            old_W = W
            start_time2 = time.perf_counter()
            W = self.update_W(theta)
            time_2 = time.perf_counter() - start_time2

            val_loss = self.val_scores(theta)
            if best_val > val_loss:
                best_val = val_loss
                best_k = k

            logging.info(
                f'time 1 = {time_1}, time 2 = {time_2}, optimal = {state}')

            F_val = self.loss(theta, W)
            LOSS.append(F_val)
            logging.info(
                f'{k}th iteration: loss = {F_val: .4e}, val = {val_loss: .4e}, best iter = {best_k}'
            )

            # stopping criterion
            res = np.sum((theta - old_theta)**2) + np.sum((W - old_W)**2)

            if res < res_tol:
                optimal = True
                break

        main_loop_time = time.perf_counter() - start_time

        # construct the result
        result = {'theta': theta, 'W': W}

        info = {'time': main_loop_time, 'iterations': k, 'optimal': optimal}

        plt.plot(range(k), LOSS)
        plt.yscale('symlog')
        plt.savefig(os.path.join(figpath, 'loss.jpg'), dpi=600)
        plt.close()

        return result, info

    def update_theta(self,
                     L,
                     l_prox,
                     r_prox,
                     theta_init,
                     abs_tol=1e-3,
                     rel_tol=1e-3,
                     rho=1,
                     mu=10,
                     tau_incr=2,
                     tau_decr=2,
                     max_rho=1e2,
                     min_rho=1e-1,
                     maxiter=500,
                     n_jobs=1,
                     max_cg_iterations=10):

        # Initialization
        theta = theta_init.copy()
        theta_tilde = theta.copy()
        theta_hat = theta.copy()
        u = np.zeros(theta.shape)
        u_tilde = np.zeros(theta.shape)

        res_pri = np.zeros(theta.shape)
        res_pri_tilde = np.zeros(theta.shape)
        res_dual = np.zeros(theta.shape)
        res_dual_tilde = np.zeros(theta.shape)

        prox_pool = mp.Pool(n_jobs)
        optimal = False

        # Main ADMM loop
        for t in range(1, maxiter + 1):
            # theta update
            theta = l_prox(1. / rho, theta_hat - u, theta, prox_pool)

            # theta_tilde update
            theta_tilde = r_prox(1. / rho, theta_hat - u_tilde, theta_tilde,
                                 prox_pool)

            # theta_hat update
            sys = L + 2 * rho * sparse.eye(self.K)
            M = sparse.diags(1. / sys.diagonal())
            indices = np.ndindex(self.shape)
            rhs = rho * (theta.T + u.T + theta_tilde.T + u_tilde.T)
            for i, ind in enumerate(indices):
                index = ind[::-1]
                sol = sparse.linalg.cg(sys,
                                       rhs[index],
                                       M=M,
                                       x0=theta_hat.T[index],
                                       maxiter=max_cg_iterations)[0]
                res_dual.T[index] = -rho * (sol - theta_hat.T[index])
                res_dual_tilde.T[index] = res_dual.T[index]
                theta_hat.T[index] = sol

            # u and u_tilde update
            res_pri = theta - theta_hat
            res_pri_tilde = theta_tilde - theta_hat
            u += theta - theta_hat
            u_tilde += theta_tilde - theta_hat

            # calculate residual norms
            res_pri_norm = np.linalg.norm(np.append(res_pri, res_pri_tilde))
            res_dual_norm = np.linalg.norm(np.append(res_dual, res_dual_tilde))

            eps_pri = np.sqrt(2 * self.K * np.prod(self.shape)) * abs_tol + \
                rel_tol * max(res_pri_norm, res_dual_norm)
            eps_dual = np.sqrt(2 * self.K * np.prod(self.shape)) * abs_tol + \
                rel_tol * np.linalg.norm(rho * np.append(u, u_tilde))

            # check stopping condition
            if res_pri_norm <= eps_pri and res_dual_norm <= eps_dual:
                optimal = True
                break

            # penalty parameter update
            new_rho = rho
            if res_pri_norm > mu * res_dual_norm:
                new_rho = tau_incr * rho
            elif res_dual_norm > mu * res_pri_norm:
                new_rho = rho / tau_decr
            new_rho = np.clip(new_rho, min_rho, max_rho)
            u *= rho / new_rho
            u_tilde *= rho / new_rho
            rho = new_rho

        # clean up the multiprocessing pool
        prox_pool.close()
        prox_pool.join()

        return theta, optimal

    def update_W(self, theta):
        cpW = cp.Variable((self.K, self.K), symmetric=True)

        # laplacian term
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = cp.sum(cp.multiply(cpW, pairwise_dist)) / 4

        # logdiag term
        logdiag = self.lambd_1 * cp.sum(cp.log(cp.sum(cpW, axis=1)))

        # frobenius term
        sq = self.lambd_2 * cp.sum_squares(cpW) / 2

        constraints = [cpW >= 0] + [cpW[i, i] == 0 for i in range(self.K)]

        problem = cp.Problem(cp.Minimize(lap_reg - logdiag + sq),
                             constraints=constraints)
        problem.solve(warm_start=True)

        W = cpW.value
        W[abs(W) <= 1e-5] = 0
        W = sparse.lil_matrix(W)
        return W


class TrConstraint(MAPG):
    """
    MAPG implementation of Tr-Constraint method
    """

    def __init__(self,
                 l_fn,
                 r_fn,
                 val_scores,
                 K,
                 shape,
                 G_data=dict(),
                 config=dict()):
        self.lambd_1 = config['lambd_1']
        self.lambd_2 = config['lambd_2']
        self.l_fn, self.r_fn = l_fn, r_fn
        self.val_scores = val_scores
        self.K, self.shape = K, shape
        self.n = np.prod(self.shape)
        self.G_data = G_data

    def loss(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = np.sum(np.array(W.todense()) * pairwise_dist) / 4
        R_2 = 0.5 * sparse.linalg.norm(Lap(W), ord='fro')**2

        total_loss = self.l_fn(theta) + np.sum(
            self.r_fn(theta)) + lap_reg + self.lambd_2 * R_2
        return total_loss

    def W_grad(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        grad_1 = squareform(sq_dist)**2 / 4.

        L = Lap(W)
        grad_2 = self.lambd_2 * Lap_diff(L)

        return grad_1 + grad_2

    def prox_grad(self, theta, W, l_grad, r_prox, alpha, pool):
        grad_theta = self.theta_grad(theta, W, l_grad, pool)
        new_theta = r_prox(alpha, theta - alpha * grad_theta, theta, pool)

        grad_W = self.W_grad(theta, W)
        M = W - alpha * grad_W
        cpW = cp.Variable(M.shape, symmetric=True)
        constraints = [cpW >= 0, cp.sum(cpW) == self.lambd_1
                       ] + [cpW[i, i] == 0 for i in range(self.K)]
        problem = cp.Problem(cp.Minimize(cp.sum_squares(cpW - M)),
                             constraints=constraints)
        problem.solve(warm_start=True)
        new_W = cpW.value
        new_W[abs(new_W) < 1e-5] = 0
        new_W = sparse.lil_matrix(new_W)
        return new_theta, new_W


class TrConstraintBi(Alg):
    """
    Alternating implementation of Tr-Constraint method
    """

    def __init__(self,
                 l_fn,
                 r_fn,
                 val_scores,
                 K,
                 shape,
                 G_data=dict(),
                 config=dict()):
        self.lambd_1 = config['lambd_1']
        self.lambd_2 = config['lambd_2']
        self.l_fn, self.r_fn = l_fn, r_fn
        self.val_scores = val_scores
        self.K, self.shape = K, shape
        self.n = np.prod(self.shape)
        self.G_data = G_data

    def loss(self, theta, W):
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = np.sum(np.array(W.todense()) * pairwise_dist) / 4
        R_2 = 0.5 * sparse.linalg.norm(Lap(W), ord='fro')**2

        total_loss = self.l_fn(theta) + np.sum(
            self.r_fn(theta)) + lap_reg + self.lambd_2 * R_2
        return total_loss

    def fit(self,
            l_prox,
            r_prox,
            abs_tol=1e-3,
            rel_tol=1e-3,
            res_tol=1e-4,
            max_iter=1000,
            n_jobs=1,
            figpath=None):

        # Initialization
        if 'theta_init' in self.G_data:
            theta = self.G_data['theta_init'].copy()
        else:
            theta = np.zeros((self.K, ) + self.shape)
        old_theta = theta.copy()

        if 'W_init' in self.G_data:
            W = self.G_data['W_init'].copy()
        else:
            W = sparse.eye(self.K)
        old_W = W.copy()

        n_jobs = n_jobs if n_jobs < self.K else self.K
        optimal = False
        LOSS = []
        best_k = 1
        best_val = np.inf

        # BCD loop
        start_time = time.perf_counter()
        for k in range(1, max_iter + 1):

            # update theta
            old_theta = theta
            L = Lap(W)
            start_time1 = time.perf_counter()
            theta, state = self.update_theta(L,
                                             l_prox,
                                             r_prox,
                                             theta,
                                             abs_tol,
                                             rel_tol,
                                             n_jobs=n_jobs)
            time_1 = time.perf_counter() - start_time1

            # update W
            old_W = W
            start_time2 = time.perf_counter()
            W = self.update_W(theta)
            time_2 = time.perf_counter() - start_time2

            val_loss = self.val_scores(theta)
            if best_val > val_loss:
                best_val = val_loss
                best_k = k

            logging.info(
                f'time 1 = {time_1}, time 2 = {time_2}, optimal = {state}')

            F_val = self.loss(theta, W)
            LOSS.append(F_val)
            logging.info(
                f'{k}th iteration: loss = {F_val: .4e}, val = {val_loss: .4e}, best iter = {best_k}'
            )

            # stopping criterion
            res = np.sum((theta - old_theta)**2) + np.sum((W - old_W)**2)

            print(self.l_fn(theta),
                  sparse.linalg.norm(W - self.G_data['W0'], ord='fro'),
                  np.sum((theta - old_theta)**2), res)

            if res < res_tol:
                optimal = True
                break

        main_loop_time = time.perf_counter() - start_time

        # construct the result
        result = {'theta': theta, 'W': W}

        info = {'time': main_loop_time, 'iterations': k, 'optimal': optimal}

        plt.plot(range(k), LOSS)
        plt.yscale('symlog')
        plt.savefig(os.path.join(figpath, 'loss.jpg'), dpi=600)
        plt.close()

        return result, info

    def update_theta(self,
                     L,
                     l_prox,
                     r_prox,
                     theta_init,
                     abs_tol=1e-3,
                     rel_tol=1e-3,
                     rho=1,
                     mu=10,
                     tau_incr=2,
                     tau_decr=2,
                     max_rho=1e2,
                     min_rho=1e-1,
                     maxiter=500,
                     n_jobs=1,
                     max_cg_iterations=10):

        # Initialization
        theta = theta_init.copy()
        theta_tilde = theta.copy()
        theta_hat = theta.copy()
        u = np.zeros(theta.shape)
        u_tilde = np.zeros(theta.shape)

        res_pri = np.zeros(theta.shape)
        res_pri_tilde = np.zeros(theta.shape)
        res_dual = np.zeros(theta.shape)
        res_dual_tilde = np.zeros(theta.shape)

        prox_pool = mp.Pool(n_jobs)
        optimal = False

        # Main ADMM loop
        for t in range(1, maxiter + 1):
            # theta update
            theta = l_prox(1. / rho, theta_hat - u, theta, prox_pool)

            # theta_tilde update
            theta_tilde = r_prox(1. / rho, theta_hat - u_tilde, theta_tilde,
                                 prox_pool)

            # theta_hat update
            sys = L + 2 * rho * sparse.eye(self.K)
            M = sparse.diags(1. / sys.diagonal())
            indices = np.ndindex(self.shape)
            rhs = rho * (theta.T + u.T + theta_tilde.T + u_tilde.T)
            for i, ind in enumerate(indices):
                index = ind[::-1]
                sol = sparse.linalg.cg(sys,
                                       rhs[index],
                                       M=M,
                                       x0=theta_hat.T[index],
                                       maxiter=max_cg_iterations)[0]
                res_dual.T[index] = -rho * (sol - theta_hat.T[index])
                res_dual_tilde.T[index] = res_dual.T[index]
                theta_hat.T[index] = sol

            # u and u_tilde update
            res_pri = theta - theta_hat
            res_pri_tilde = theta_tilde - theta_hat
            u += theta - theta_hat
            u_tilde += theta_tilde - theta_hat

            # calculate residual norms
            res_pri_norm = np.linalg.norm(np.append(res_pri, res_pri_tilde))
            res_dual_norm = np.linalg.norm(np.append(res_dual, res_dual_tilde))

            eps_pri = np.sqrt(2 * self.K * np.prod(self.shape)) * abs_tol + \
                rel_tol * max(res_pri_norm, res_dual_norm)
            eps_dual = np.sqrt(2 * self.K * np.prod(self.shape)) * abs_tol + \
                rel_tol * np.linalg.norm(rho * np.append(u, u_tilde))

            # check stopping condition
            if res_pri_norm <= eps_pri and res_dual_norm <= eps_dual:
                optimal = True
                break

            # penalty parameter update
            new_rho = rho
            if res_pri_norm > mu * res_dual_norm:
                new_rho = tau_incr * rho
            elif res_dual_norm > mu * res_pri_norm:
                new_rho = rho / tau_decr
            new_rho = np.clip(new_rho, min_rho, max_rho)
            u *= rho / new_rho
            u_tilde *= rho / new_rho
            rho = new_rho

        # clean up the multiprocessing pool
        prox_pool.close()
        prox_pool.join()

        return theta, optimal

    def update_W(self, theta):
        cpW = cp.Variable((self.K, self.K), symmetric=True)

        # laplacian term
        sq_dist = pdist(theta.reshape(self.K, self.n))
        pairwise_dist = squareform(sq_dist)**2
        lap_reg = cp.sum(cp.multiply(cpW, pairwise_dist)) / 4

        # frobenius term
        sq = self.lambd_2 * (cp.sum_squares(cp.sum(cpW, 1)) +
                             cp.sum_squares(cpW)) / 2

        constraints = [cpW >= 0, cp.sum(cpW) == self.lambd_1
                       ] + [cpW[i, i] == 0 for i in range(self.K)]

        problem = cp.Problem(cp.Minimize(lap_reg + sq),
                             constraints=constraints)
        problem.solve(warm_start=True)
        W = cpW.value
        W[abs(W) <= 1e-5] = 0
        W = sparse.lil_matrix(W)
        return W
