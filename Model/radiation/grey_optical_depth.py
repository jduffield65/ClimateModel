""" All optical depth functions return:
1 - mass concentration distribution, q
2 - optical depth, tau
3 - sympy_function as function of pressure
4 - sympy_function arguments excluding pressure"""
import numpy as np
from ..constants import p_surface, g
from sympy import symbols, lambdify, diff, exp, simplify, sympify, integrate, cancel, Function
import inspect
from sympy.solvers import solve


def get_scale_height_alpha(p_width):
    """

    :param p_width: difference between pressure value at surface and where
        mass concentration falls, q, falls to 1/e of q(p_max)
    :return:
    alpha: the larger alpha, the more peaked q and tau are about p_surface.
    """
    p_fall_value = p_surface - p_width
    if p_fall_value > p_surface:
        raise ValueError('p_fall_value is above p_max')
    if p_fall_value == 0:
        return 0.0
    else:
        return -1 / np.log(p_fall_value / p_surface)


def scale_height(p, p_width=0.22 * p_surface, tau_surface=4, k=1):
    """
    method used in textbook, Atmospheric Circulation Dynamics and General Circulation Models.
    scale height of absorbing constituent is H/alpha, where H is scale height of pressure

    :param p: numpy array.
        pressure levels to find optical depth.
    :param p_width: float, optional.
        Difference between pressure at surface
        and where mass concentration, q, falls to 1/e of q(p_surface)
        default: 0.22*p_surface
    :param tau_surface: float, optional.
        The value of optical depth at the surface.
        default: 4
    :param k: float, optional.
        Absorption coefficient for gas.
        default: 1
    """
    alpha = get_scale_height_alpha(p_width)

    def tau_func_sympy(p, tau_surface, alpha):
        return tau_surface * (p / p_surface) ** (alpha + 1)

    tau_func, tau_diff_func = differentiate(tau_func_sympy)
    tau = tau_func(p, tau_surface, alpha)
    q = g / k * tau_diff_func(p, tau_surface, alpha)  # constituent density distribution
    return q, tau, tau_func_sympy, [tau_surface, alpha]


def get_exponential_p_width(alpha):
    """
    This finds the p_width for the exponential optical depth if you know alpha.
    Useful for finding an analytic solution as this requires alpha_lw/alpha_sw to be an integer
    and small. Hence you know alpha and want to find p_fall_value.
    :param alpha: float.
        the larger alpha, the more peaked tau is about p_max.
    :param p_max: float, optional.
        Will differ from p_surface for peak_in_atmopsphere.
        default: p_surface
    :return:
    p_width: Difference between p_max and pressure
        where mass concentration, q, falls to 1/e of q(p_surface)
    """
    return 1 / alpha


def get_exponential_alpha(p_width, p_max=p_surface):
    """

    :param p_width: Difference between pressure at surface
        and where mass concentration, q, falls to 1/e of q(p_surface)
    :param p_max: pressure level where mass concentration, q is peaked.
        default: p_surface
    :return:
    alpha: the larger alpha, the more peaked q is about p_max.
    """
    p_fall_value = p_max - p_width
    if p_fall_value > p_max:
        raise ValueError('p_fall_value is larger than p_max')
    return 1 / (p_max - p_fall_value)


def exponential(p, p_width=0.22 * p_surface, tau_surface=4, k=1):
    """
    Optical depth falls off exponentially as pressure decreases.
    Can use this method to get an analytic solution with a short wave contribution.

    :param p: numpy array.
        pressure levels to find optical depth.
    :param p_width: float, optional.
        Difference between pressure at surface
        and where mass concentration, q, falls to 1/e of q(p_surface)
        default: 0.22*p_surface
    :param tau_surface: float, optional.
        The value of optical depth at the surface.
        default: 4
    :param k: float, optional.
        Absorption coefficient for gas.
        default: 1
    """
    alpha = get_exponential_alpha(p_width)
    coef = tau_surface / (np.exp(alpha * p_surface) - 1)

    def tau_func_sympy(p, coef, alpha):
        return coef * (exp(alpha * p) - 1)

    tau_func, tau_diff_func = differentiate(tau_func_sympy)
    tau = tau_func(p, coef, alpha)
    q = g / k * tau_diff_func(p, coef, alpha)  # constituent density distribution
    # q = coef * g / k * alpha * np.exp(alpha * p)
    # tau = coef * (np.exp(alpha * p) - 1)
    return q, tau, tau_func_sympy, [coef, alpha]


def peak_in_atmosphere(p, p_width=10000, p_max=50000, tau_surface=4, k=1):
    """
    Mass concentration, q, is peaked at p_max and falls off away from this as exp(-alpha|p-p_max|)
    either side.

    :param p: numpy array.
        pressure levels to find optical depth.
    :param p_width: float, optional.
        Difference between p_max and pressure
        where mass concentration, q, falls to 1/e of q(p_max)
        default: 10000
    :param p_max: float, optional.
        pressure level where mass concentration is peaked.
        default: 50000
    :param tau_surface: float, optional.
        The value of optical depth at the surface.
        default: 4
    :param k: float, optional.
        Absorption coefficient for gas.
        default: 1
    """
    alpha = get_exponential_alpha(p_width, p_max)
    coef = tau_surface / (2 - np.exp(-alpha * p_max) - np.exp(alpha * (p_max - p_surface)))

    class tau_func_sympy(Function):
        # ensure last argument is the threshold which determines the function to use
        @staticmethod
        def below_thresh(p, coef, alpha, p_max):
            return coef * (exp(alpha * (p - p_max)) - exp(-alpha * p_max))

        @staticmethod
        def above_thresh(p, coef, alpha, p_max):
            return coef * (2 - exp(-alpha * p_max) - exp(alpha * (p_max - p)))

        @classmethod
        def eval(cls, p, coef, alpha, p_max):
            if p <= p_max:
                return cls.below_thresh(p, coef, alpha, p_max)
            else:
                return cls.above_thresh(p, coef, alpha, p_max)

    tau_func_less, tau_diff_func_less = differentiate(tau_func_sympy.below_thresh)
    tau_func_more, tau_diff_func_more = differentiate(tau_func_sympy.above_thresh)

    def tau_func(p, coef, alpha, p_max):
        p = np.array(p)
        tau = p.copy()
        tau[p <= p_max] = tau_func_less(p[p <= p_max], coef, alpha, p_max)
        tau[p > p_max] = tau_func_more(p[p > p_max], coef, alpha, p_max)
        return tau

    def tau_diff_func(p, coef, alpha, p_max):
        p = np.array(p)
        tau_diff = p.copy()
        tau_diff[p <= p_max] = tau_diff_func_less(p[p <= p_max], coef, alpha, p_max)
        tau_diff[p > p_max] = tau_diff_func_more(p[p > p_max], coef, alpha, p_max)
        return tau_diff

    tau = tau_func(p, coef, alpha, p_max)
    q = g / k * tau_diff_func(p, coef, alpha, p_max)  # constituent density distribution
    return q, tau, tau_func_sympy, [coef, alpha, p_max]


def scale_height_and_peak_in_atmosphere(p, p_width1=0.7788 * p_surface, tau_surface1=4,
                                        p_width2=10000, p_max2=50000, tau_surface2=4, k=1):
    """
    Combination of scale_height and peak_in_atmosphere functions.

    :param p: numpy array.
        pressure levels to find optical depth.
    :param p_width1: float, optional.
        Difference between pressure at surface
        and where mass concentration, q1, falls to 1/e of q1(p_surface)
        default: 0.22*p_surface
    :param tau_surface1: float, optional, scale_height arg.
        The value of optical depth at the surface due to scale_height.
        default: 4
    :param p_width2: float, optional.
        Difference between p_max2 and pressure
        where mass concentration, q2, falls to 1/e of q2(p_max2)
        default: 10000
    :param p_max2: float, optional, peak_in_atmosphere arg.
        pressure level where mass concentration is peaked.
        default: 50000
    :param tau_surface2: float, optional, peak_in_atmosphere arg.
        The value of optical depth at the surface due to peak_in_atmosphere.
        default: 4
    :param k: float, optional.
        Absorption coefficient for gas.
        default: 1
    """
    alpha1 = get_scale_height_alpha(p_width1)
    alpha2 = get_exponential_alpha(p_width2, p_max2)
    coef2 = tau_surface2 / (2 - np.exp(-alpha2 * p_max2) - np.exp(alpha2 * (p_max2 - p_surface)))

    class tau_func_sympy(Function):
        @staticmethod
        def below_thresh(p, tau_surface1, alpha1, coef2, alpha2, p_max2):
            # HACK SO CAN COMPUTE PRESSURE FROM TAU - NOT CORRECT
            # PEAK CONTRIBUTION IS THAT FROM WELL BELOW PEAK - 1% OF MAX
            # return (tau_surface1 * (p / p_surface) ** (alpha1 + 1) +
            #         coef2 * (exp(alpha2 * (p_max2/100 - p_max2)) - exp(-alpha2 * p_max2)))
            return tau_surface1 * (p / p_surface) ** (alpha1 + 1)

        @staticmethod
        def above_thresh(p, tau_surface1, alpha1, coef2, alpha2, p_max2):
            # HACK SO CAN COMPUTE PRESSURE FROM TAU - NOT CORRECT
            # INCLUDE VALUE AT SURFACE ABOVE PEAK
            return tau_surface1 * (p / p_surface) ** (alpha1 + 1)

        # ensure last argument is the threshold which determines the function to use
        @staticmethod
        def below_thresh_correct(p, tau_surface1, alpha1, coef2, alpha2, p_max2):
            return (tau_surface1 * (p / p_surface) ** (alpha1 + 1) +
                    coef2 * (exp(alpha2 * (p - p_max2)) - exp(-alpha2 * p_max2)))

        @staticmethod
        def above_thresh_correct(p, tau_surface1, alpha1, coef2, alpha2, p_max2):
            return (tau_surface1 * (p / p_surface) ** (alpha1 + 1) +
                    coef2 * (2 - exp(-alpha2 * p_max2) - exp(alpha2 * (p_max2 - p))))

        @classmethod
        def eval(cls, p, tau_surface1, alpha1, coef2, alpha2, p_max2):
            if p <= p_max2:
                return cls.below_thresh_correct(p, tau_surface1, alpha1, coef2, alpha2, p_max2)
            else:
                return cls.above_thresh_correct(p, tau_surface1, alpha1, coef2, alpha2, p_max2)

    tau_func_less, tau_diff_func_less = differentiate(tau_func_sympy.below_thresh_correct)
    tau_func_more, tau_diff_func_more = differentiate(tau_func_sympy.above_thresh_correct)

    def tau_func(p, tau_surface1, alpha1, coef2, alpha2, p_max2):
        p = np.array(p)
        tau = p.copy()
        tau[p <= p_max2] = tau_func_less(p[p <= p_max2], tau_surface1, alpha1, coef2, alpha2, p_max2)
        tau[p > p_max2] = tau_func_more(p[p > p_max2], tau_surface1, alpha1, coef2, alpha2, p_max2)
        return tau

    def tau_diff_func(p, tau_surface1, alpha1, coef2, alpha2, p_max2):
        p = np.array(p)
        tau_diff = p.copy()
        tau_diff[p <= p_max2] = tau_diff_func_less(p[p <= p_max2], tau_surface1, alpha1, coef2, alpha2, p_max2)
        tau_diff[p > p_max2] = tau_diff_func_more(p[p > p_max2], tau_surface1, alpha1, coef2, alpha2, p_max2)
        return tau_diff

    tau = tau_func(p, tau_surface1, alpha1, coef2, alpha2, p_max2)
    q = g / k * tau_diff_func(p, tau_surface1, alpha1, coef2, alpha2,
                              p_max2)  # constituent mass density distribution
    return q, tau, tau_func_sympy, [tau_surface1, alpha1, coef2, alpha2, p_max2]


def differentiate(func):
    """

    :param func: differentiates func with respect to first argument
    :return func_numpy: returns func that works on numpy arrays
    :return func_diff: return differential of func that works on numpy arrays
    """
    n_params = len(inspect.signature(func).parameters)
    # get symbol for each parameter (97 is index of 'a')
    param_symbols = tuple(symbols(chr(97 + i)) for i in range(n_params))
    func_symbol = func(*param_symbols)
    func_numpy = lambdify(list(param_symbols), func_symbol, "numpy")
    func_diff_symbol = diff(func_symbol, param_symbols[0])
    # can get divide by zero error if don't simplify first
    func_diff_symbol = simplify(func_diff_symbol)
    func_diff = lambdify(list(param_symbols), func_diff_symbol, "numpy")
    return func_numpy, func_diff


def get_p_from_tau(func):
    """func calculates tau from pressure which is first argument.
    p_from_tau calculates pressure from tau which is first argument.
    Order of all other variables are the same in each function."""
    n_params = len(inspect.signature(func).parameters)
    param_symbols = tuple(symbols(chr(97 + i)) for i in range(n_params))
    tau = symbols('t')
    final_params = list((tau,) + param_symbols[1:])
    if inspect.isclass(func):
        p_from_tau_symbol_above = solve(tau - func.above_thresh(*param_symbols), param_symbols[0])
        p_from_tau_symbol_below = solve(tau - func.below_thresh(*param_symbols), param_symbols[0])
        p_from_tau_above = lambdify(final_params, p_from_tau_symbol_above, "numpy")
        p_from_tau_below = lambdify(final_params, p_from_tau_symbol_below, "numpy")

        def p_from_tau(*args):
            # pressure threshold where function changes is last argument
            # Always have high pressure and high optical depth together so threshold in same direction for tau
            tau_thresh = func.below_thresh(args[-1], *args[1:])
            tau = np.array(args[0])
            p = tau.copy()
            p[tau <= tau_thresh] = p_from_tau_below(tau[tau <= tau_thresh], *args[1:])[0]
            p[tau > tau_thresh] = p_from_tau_above(tau[tau > tau_thresh], *args[1:])[0]
            return [p]  # return list to match normal result
    else:
        p_from_tau_symbol = solve(tau - func(*param_symbols), param_symbols[0])
        p_from_tau = lambdify(final_params, p_from_tau_symbol, "numpy")
    return p_from_tau
