import numpy as np


def linear_prob(N, slope, intercept):
    """ Linear probabality with slope and intercept.
    Returns reveresed order probability because patient 0 has longest time.

    Args:
        N (int): number of patients
        slope (float): slope of linear function
        intercept (float): intecept of linear function


    Returns:
        np.array: array of probabilities
    """
    patients = np.linspace(0, 1, N)
    prob = slope * patients + intercept
    prob = prob / np.sum(prob)
    return prob[::-1]


def power_prob(N, k):
    """ a^k probability.
    Returns reveresed order probability because patient 0 has longest time.

    Args:
        N (int): number of patients
        k(float): power

    Returns:
        np.array: array of probabilities
    """
    patients = np.linspace(0, 1, N)
    prob = np.power(patients, k)
    prob = prob / np.sum(prob)
    return prob[::-1]


def exp_prob(N, k):
    """ exp(ak)/exp(k) probability.
    Returns reveresed order probability because patient 0 has longest time.

    Args:
        N (int): number of patients
        k (float): coefficient

    Returns:
        np.array: array of probabilities
    """
    patients = np.linspace(0, 1, N)
    prob = np.exp(k * (patients - 1))
    prob = prob / np.sum(prob)
    return prob[::-1]


def get_prob(N, coef, offset, kind='linear'):
    """ Probabilities of each entry.
    #FIXME Write an explanation of the function
    Higher coef will result in higher correlation.
    Linear probability will ignore offset parameter

    Args:
        N (int): number of patients
        coef (float): coefficient
        offset (float): offset
        kind (str): choice of linear, power, exponential (default: linear)

    Returns:
        np.array: array of probabilities
    """
    assert (kind in ['linear', 'power', 'exponential']), "Unrecognized kind parameter"

    # linear coefficient must be between -2 and 2 (non-inclusive)
    if kind == 'linear':
        assert (abs(coef) < 2), "Linear coef must be between -2 and 2 (non-inclusive)"
        patients = np.arange(0, N)

        if abs(coef) <= 1:
            adj_coef = 2 / (N * (N - 1)) * coef
            offset = 1 / N - ((N - 1) * adj_coef / 2)
            prob = adj_coef * patients + offset
            #FIXME I'm not sure this is the solution
            # sometimes, I get small negative value. Manually change them to zero
            if np.sum(prob < 0) > 0:
                prob[np.argwhere(prob < 0)[0]] = 0
        else:
            n = np.round((2 - abs(coef)) * N, 0)
            adj_coef = 2 / (n * (n - 1))
            prob = adj_coef * np.arange(0, n)
            if coef < 0:
                prob = np.hstack((prob[::-1], np.zeros(int(N - n))))
            else:
                prob = np.hstack((np.zeros(int(N - n)), prob))

    elif kind == 'power':
        #FIXME change this to arange (we need to exclude the actual 0)
        patients = np.linspace(0, 1, N)
        prob = np.power(patients, abs(coef)) + offset
        if coef < 0:
            prob = prob[::-1]
        prob = prob / np.sum(prob)

    elif kind == 'exponential':
        patients = np.linspace(0, 1, N)
        prob = np.exp(coef * (patients - 1)) + offset
        prob = prob / np.sum(prob)

    return prob
