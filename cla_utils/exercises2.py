import numpy as np


def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """
    u = Q.conj().T @ v
    r = v
    for i in range(Q.shape[1]):
        r = r - u[i] * Q[:, i]
    return r, u


def solve_Q(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    x = Q.conj().T @ b

    return x


def time_solve_Q(m):
    import timeit
    setup = f'''
from numpy import random
import numpy as np
import cla_utils
m = {m}
random.seed(1431*m)
A = random.randn(m, m) + 1j*random.randn(m, m)
v = random.randn(m) + 1j*random.randn(m)
Q, R = np.linalg.qr(A)
    '''
    mycode = '''
x = cla_utils.solve_Q(Q, v)
    '''
    print(min(timeit.Timer(mycode, setup=setup).repeat(7, 1000)))
    mycode2 = '''
x = np.linalg.solve(Q, v)
    '''
    print(min(timeit.Timer(mycode2, setup=setup).repeat(7, 1000)))


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    P = Q @ Q.conj().T

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an mxl-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U, for appropriate l.
    """
    m, n = V.shape
    Q_, _ = np.linalg.qr(V, 'complete')

    Q = Q_[:, n:]

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm, transforming A to Q in place and returning R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """
    n = A.shape[1]  # Number of columns in A
    R = np.zeros((n, n), dtype='complex')  # Initialize R matrix
    for j in range(n):
        R[:j, j] = np.dot(A[:, :j].conj().T, A[:, j])
        A[:, j] = A[:, j] - np.dot(A[:, :j], R[:j, j])
        R[j, j] = np.linalg.norm(A[:, j])
        A[:, j] = A[:, j] / R[j, j]
    return R


def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, transforming A to Q in place and returning
    R.

    :param A: mxn numpy array

    :return R: nxn numpy array
    """

    n = A.shape[1]  # Number of columns in A
    R = np.zeros((n, n), dtype='complex')  # Initialize R matrix
    for i in range(n):
        R[i, i] = np.linalg.norm(A[:, i])
        A[:, i] = A[:, i]/R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.dot(A[:, i].conj().T, A[:, j])
            A[:, j] = A[:, j] - np.dot(A[:, i], R[i, j])
    return R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is normalised and orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """
    n = A.shape[1]
    R = np.eye(n, n, dtype="complex")
    R[k, k+1:] = A[:, k+1:].T @ A[:, k]
    R[k, :] = R[k, :] / np.linalg.norm(A[:, k])

    return R


def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:, :] = np.dot(A, Rk)
        R[:, :] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
