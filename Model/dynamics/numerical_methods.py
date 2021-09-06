"""All functions in this file solve:
du/dt + df/dx + dg/dy = Q
u is a vector of all the conserved quantities [ n_conserved x nx x ny ]
f is a function for determining the flux (in x direction) from the conserved quantities
g is a function for determining the flux (in y direction) from the conserved quantities
Q is a function for determining the source term from the conserved quantities
no_source_ind: indices of conserved quantities for which the source is everywhere zero"""
import numpy as np


def lax_friedrichs(u, f, g, Q, dt, dx, dy, no_source_ind=None):
    if no_source_ind is None:
        no_source_ind = []
    un = u.copy()  # copy the existing values of u into un
    sigma_x = dt / dx
    sigma_y = dt / dy
    u_new_no_source = 0.25 * (un[:, 2:, 1:-1] + un[:, 0:-2, 1:-1] + un[:, 1:-1, 2:] + un[:, 1:-1, 0:-2]) - \
                       0.5 * sigma_x * (f(un[:, 2:, 1:-1]) - f(un[:, 0:-2, 1:-1])) - \
                       0.5 * sigma_y * (g(un[:, 1:-1, 2:]) - g(un[:, 1:-1, 0:-2]))
    u = include_source(u, un, u_new_no_source, Q, no_source_ind, dt)
    return u


def jacobian_mult(J, f, nx, ny):
    """
    Does matrix multiplication of f by Jacobian to return vector of dim [n_conserved x nx x ny]

    :param J: numpy array [nx x ny x n_conserved x n_conserved]
    :param f: numpy array [n_conserved x nx x ny]
    :param nx:
    :param ny:
    :return:
    """
    f = np.reshape(f.transpose(1, 2, 0), [nx, ny, -1, 1])
    return np.matmul(J, f).transpose(2, 0, 1, 3)[:, :, :, 0]


def lax_wendroff(u, f, g, Q, dt, dx, dy, no_source_ind, nx, ny, A, B):
    """A is the jacobian function of f, A = df/du, returns matrix: [nx x ny x n_conserved x n_conserved]
    B is the jacobian of g, B = dg/du"""
    un = u.copy()  # copy the existing values of u into un
    sigma_x = dt / dx
    sigma_y = dt / dy
    A_plus_half = A(0.5 * (un[:, 2:, 1:-1] + un[:, 1:-1, 1:-1]))
    A_plus_half_term = jacobian_mult(A_plus_half, f(un[:, 2:, 1:-1]) - f(un[:, 1:-1, 1:-1]), nx - 2, ny - 2)
    A_minus_half = A(0.5 * (un[:, 1:-1, 1:-1] + un[:, 0:-2, 1:-1]))
    A_minus_half_term = jacobian_mult(A_minus_half, f(un[:, 1:-1, 1:-1]) - f(un[:, 0:-2, 1:-1]), nx - 2, ny - 2)

    B_plus_half = B(0.5 * (un[:, 1:-1, 2:] + un[:, 1:-1, 1:-1]))
    B_plus_half_term = jacobian_mult(B_plus_half, g(un[:, 1:-1, 2:]) - g(un[:, 1:-1, 1:-1]), nx - 2, ny - 2)
    B_minus_half = B(0.5 * (un[:, 1:-1, 1:-1] + un[:, 1:-1, 0:-2]))
    B_minus_half_term = jacobian_mult(B_minus_half, g(un[:, 1:-1, 1:-1]) - g(un[:, 1:-1, 0:-2]), nx - 2, ny - 2)

    u_new_no_source = un[:, 1:-1, 1:-1] - 0.5 * sigma_x * (f(un[:, 2:, 1:-1]) - f(un[:, 0:-2, 1:-1])) + \
                       0.5 * sigma_x ** 2 * (A_plus_half_term - A_minus_half_term) - \
                       0.5 * sigma_y * (g(un[:, 1:-1, 2:]) - g(un[:, 1:-1, 0:-2])) + \
                       0.5 * sigma_y ** 2 * (B_plus_half_term - B_minus_half_term)
    u = include_source(u, un, u_new_no_source, Q, no_source_ind, dt)
    return u


def richtmyer(u, f, g, Q, dt, dx, dy, no_source_ind=None):
    if no_source_ind is None:
        no_source_ind = []
    un = u.copy()  # copy the existing values of u into un
    sigma_x = dt / dx
    sigma_y = dt / dy
    u_minus_half_x = 0.5 * (un[:, 1:, 1:-1] + un[:, 0:-1, 1:-1]) - 0.5 * sigma_x * \
                    (f(un[:, 1:, 1:-1]) - f(un[:, 0:-1, 1:-1]))
    # u_minus_half_x = 0.5 * (un[:, 1:-1, 1:-1] + un[:, 0:-2, 1:-1]) - 0.5 * sigma_x * \
    #                 (f(un[:, 1:-1, 1:-1]) - f(un[:, 0:-2, 1:-1]))
    u_minus_half_y = 0.5 * (un[:, 1:-1, 1:] + un[:, 1:-1, 0:-1]) - 0.5 * sigma_y * \
                    (g(un[:, 1:-1, 1:]) - g(un[:, 1:-1, 0:-1]))
    # u_minus_half_y = 0.5 * (un[:, 1:-1, 1:-1] + un[:, 1:-1, 0:-2]) - 0.5 * sigma_y * \
    #                 (g(un[:, 1:-1, 1:-1]) - g(un[:, 1:-1, 0:-2]))
    u_new_no_source = un[:, 1:-1, 1:-1] - \
                      sigma_x * (f(u_minus_half_x[:, 1:, :]) - f(u_minus_half_x[:, 0:-1, :])) - \
                      sigma_y * (g(u_minus_half_y[:, :, 1:]) - g(u_minus_half_y[:, :, 0:-1]))
    u = include_source(u, un, u_new_no_source, Q, no_source_ind, dt)
    return u


def maccormack(u, f, g, Q, dt, dx, dy, no_source_ind=None):
    if no_source_ind is None:
        no_source_ind = []
    un = u.copy()  # copy the existing values of u into un
    sigma_x = dt / dx
    sigma_y = dt / dy
    u_pred = un[:, 1:-1, 1:-1] - sigma_x * (f(un[:, 2:, 1:-1]) - f(un[:, 1:-1, 1:-1])) - \
             sigma_y * (g(un[:, 1:-1, 2:]) - g(un[:, 1:-1, 1:-1]))
    u_pred_minus_x = un[:, 0:-2, 1:-1] - sigma_x * (f(un[:, 1:-1, 1:-1]) - f(un[:, 0:-2, 1:-1])) - \
                     sigma_y * (g(un[:, 0:-2, 2:]) - g(un[:, 0:-2, 1:-1]))
    u_pred_minus_y = un[:, 1:-1, 0:-2] - sigma_x * (f(un[:, 2:, 0:-2]) - f(un[:, 1:-1, 0:-2])) - \
                     sigma_y * (g(un[:, 1:-1, 1:-1]) - g(un[:, 1:-1, 0:-2]))
    u_new_no_source = 0.5 * (un[:, 1:-1, 1:-1] + u_pred) - \
                       0.5 * sigma_x * (f(u_pred) - f(u_pred_minus_x)) - \
                       0.5 * sigma_x * (g(u_pred) - g(u_pred_minus_y))
    u = include_source(u, un, u_new_no_source, Q, no_source_ind, dt)
    return u


def include_source(u, un, u_new_no_source, Q, no_source_ind, dt):
    for i in no_source_ind:
        # If can update quantity without source then do this first and use updated value to compute
        # source
        u[i, 1:-1, 1:-1] = u_new_no_source[i]
    u_input_for_source = 0.5 * (u + un)
    u[:, 1:-1, 1:-1] = u_new_no_source + Q(u_input_for_source) * dt
    return u


def centered_diff_x(u, dx):
    return (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dx)


def centered_diff_y(u, dy):
    return (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dy)
