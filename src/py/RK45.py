import numpy as np

# These functions are written by ChatGPT
def rk45_step(func, t, y, h, tol=1e-6, ncp = np):
    """
    Perform a single step of the RK45 method.
    
    Parameters:
        func: callable
            The ODE system function. It should have the form func(t, y).
        t: float
            The current time.
        y: numpy array
            The current state.
        h: float
            The step size.
        tol: float, optional
            The tolerance for error estimation.
    
    Returns:
        t_new: float
            The new time.
        y_new: numpy array
            The new state.
        h_new: float
            The new step size.
    """
    c = ncp.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    a = [
        [],
        [1/4],
        [3/32, 9/32],
        [1932/2197, -7200/2197, 7296/2197],
        [439/216, -8, 3680/513, -845/4104],
        [-8/27, 2, -3544/2565, 1859/4104, -11/40]
    ]
    b = ncp.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    b_star = ncp.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
    
    k = []
    for i in range(6):
        y_temp = y + h * ncp.sum(ncp.array([a[i][j] * k[j] for j in range(i)]), axis=0)[:,ncp.newaxis] if i > 0 else y
        k.append(func(y_temp.flatten(), t + c[i] * h))
    
    y_new = y + h * ncp.sum(ncp.array([b[i] * k[i] for i in range(6)]), axis=0)[:,ncp.newaxis]
    y_new_star = y + h * ncp.sum(ncp.array([b_star[i] * k[i] for i in range(6)]), axis=0)[:,ncp.newaxis]
    
    error = ncp.linalg.norm(y_new - y_new_star, ord=ncp.inf)
    if error > tol:
        h_new = h * min(0.9 * (tol / error)**0.2, 1.0)
    else:
        h_new = h * min(0.9 * (tol / error)**0.25, 1.5)
    
    return t + h, y_new, h_new

def rk45(func, t_span, y0, h0=0.1, tol=1e-6, ncp = np):
    """
    Solve an ODE using the RK45 method.
    
    Parameters:
        func: callable
            The ODE system function. It should have the form func(t, y).
        t_span: tuple
            A tuple (t0, tf) where t0 is the initial time and tf is the final time.
        y0: numpy array
            The initial state.
        h0: float, optional
            The initial step size.
        tol: float, optional
            The tolerance for error estimation.
    
    Returns:
        t_vals: list
            The times at which the solution was evaluated.
        y_vals: list
            The values of the solution at the corresponding times.
    """
    t0, tf = t_span
    t = t0
    y = ncp.array(y0, dtype=float)
    h = h0
    
    t_vals = [t]
    y_vals = [y]
    
    while t < tf:
        if t + h > tf:
            h = tf - t
        t, y, h = rk45_step(func, t, y, h, tol)
        t_vals.append(t)
        y_vals.append(y)
    
    return y