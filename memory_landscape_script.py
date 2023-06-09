# Modules
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

# Landscape class
class LandscapeAnimation(object):
    def __init__(self, grid_size=75, x_min=0, x_max=10, N_t=5, N_v=5, N_noise=20, lr=0.08, beta=0.6,
                 var_lower_x=0.3,var_lower_y = 0.3, var_upper_x = 1, var_upper_y = 1,
                 lower_var_noise_x=3, upper_var_noise_x=7, lower_var_noise_y=3, upper_var_noise_y=7,
                 A_t_lower=-2, A_t_upper=2, A_v_lower=-2, A_v_upper=2, A_noise_lower=-0.4, A_noise_upper=0.4, eps = 10**(-10),
                 max_iterations=10000):

        self.grid_size = grid_size      # Grid size

        # Rectangular grid parameters
        self.x_min = x_min
        self.x_max = x_max

        # Valleys, troughs and pertubations
        self.N_t = N_t                  # Number of troughs
        self.N_v = N_v                  # Number of valleys
        self.N_noise = N_noise          # Number of noise pertubations
        self.lr = lr                    # Learning rate
        self.beta = beta                # Momentum rate

        # Variances
        self.var_lower_x = var_lower_x  # Lower bound for variance of troughs and valleys in x - direction
        self.var_lower_y = var_lower_y  # Lower bound for variance of troughs and valleys in y - direction
        self.var_upper_x = var_upper_x  # Upper bound for variance of troughs and valleys in x - direction
        self.var_upper_y = var_upper_y  # Upper bound for variance of troughs and valleys in y - direction

        self.lower_var_noise_x = lower_var_noise_x      # Lower bound for variance of noise pertubations in x - direction
        self.lower_var_noise_y = lower_var_noise_y      # Lower bound for variance of noise pertubations in y - direction
        self.upper_var_noise_x = upper_var_noise_x      # Upper bound for variance of noise pertubations in x - direction
        self.upper_var_noise_y = upper_var_noise_y      # Upper bound for variance of noise pertubations in y - direction

        # Amplitudes
        self.A_t_lower = A_t_lower      # Lower bound amplitude for trough
        self.A_t_upper = A_t_upper      # Upper bound amplitude for trough
        self.A_v_lower = A_v_lower      # Lower bound amplitude for valley
        self.A_v_upper = A_v_upper      # Upper bound amplitude for valley

        self.A_noise_lower = A_noise_lower  # Lower bound amplitude for noise pertubation
        self.A_noise_upper = A_noise_upper  # Upper bound amplitude for noise pertubation

        # Convergence parameters
        self.max_iterations = max_iterations
        self.eps = eps

        # Optional plotting
        #self.fig = plt.figure()
        #self.ax = self.fig.add_subplot(111, projection='3d')
        #self.ax.set_xlabel('X')
        #self.ax.set_ylabel('Y')
        #self.ax.set_zlabel('Z')

        # Landscape creation
        self.X, self.Y, self.Z = self.create_mesh()
        self.add_throughs_and_valleys()
        self.add_noise()

        # Interpolation of landscape and gradient
        gradient_tensor = self.gradient_3d_surface(self.Z, (self.x_max-self.x_min)/self.grid_size)
        self.Z_interp = interp2d(self.X, self.Y, self.Z, kind='linear')
        self.gradient_interp_x = interp2d(self.X, self.Y, gradient_tensor[0], kind='linear')
        self.gradient_interp_y = interp2d(self.X, self.Y, gradient_tensor[1], kind='linear')

        # Running 'animation'
        self.avg = self.run_animation()

    # Mesh creation
    def create_mesh(self):
        x_range = np.linspace(self.x_min, self.x_max, self.grid_size)
        y_range = np.linspace(self.x_min, self.x_max, self.grid_size)
        X, Y = np.meshgrid(x_range, y_range)
        Z = X * 0
        return X, Y, Z

    # 2D Gaussian function
    def gaussian_2d(self, x, y, x0, y0, sigma_x, sigma_y, A):
        exponent = -((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2))
        return A * np.exp(exponent)

    # Adding troughs and valleys
    def add_throughs_and_valleys(self):

        xs_t, ys_t = np.random.uniform(self.x_min, self.x_max, size=self.N_t), \
                     np.random.uniform(self.x_min, self.x_max, size=self.N_t)       # Generating mean positions for troughs
        xs_v, ys_v = np.random.uniform(self.x_min, self.x_max, size=self.N_v), \
                     np.random.uniform(self.x_min, self.x_max, size=self.N_v)       # Generating mean positions for valleys

        var_t_x, var_t_y = np.random.uniform(self.var_lower_x, self.var_upper_x, size=self.N_t), \
                           np.random.uniform(self.var_lower_y, self.var_upper_y, size=self.N_t)     # Generating variances for troughs
        var_v_x, var_v_y = np.random.uniform(self.var_lower_x, self.var_upper_x, size=self.N_v), \
                           np.random.uniform(self.var_lower_y, self.var_upper_y, size=self.N_v)     # Generating variances for valleys

        A_t = np.random.uniform(self.A_t_lower, self.A_t_upper, size=self.N_t)      # Generating amplitudes for troughs
        A_v = np.random.uniform(self.A_v_lower, self.A_v_upper, size=self.N_v)      # Generating amplitudes for valleys

        for t in range(self.N_t):
            through = self.gaussian_2d(self.X, self.Y, xs_t[t], ys_t[t], var_t_x[t], var_t_y[t], A_t[t])
            self.Z += through

        for v in range(self.N_v):
            valley = self.gaussian_2d(self.X, self.Y, xs_v[v], ys_v[v], var_v_x[v], var_v_y[v], A_v[v])
            self.Z += valley

    # Adding gaussian noise pertubations
    def add_noise(self):
        xs_n, ys_n = np.random.uniform(self.x_min, self.x_max, size=self.N_noise), \
                     np.random.uniform(self.x_min, self.x_max, size=self.N_noise)   # Generating mean positions for pertubations

        # Generating variances for noise pertubations
        noise_var_x = np.random.uniform(self.lower_var_noise_x, self.upper_var_noise_x, size=self.N_noise)
        noise_var_y = np.random.uniform(self.lower_var_noise_y, self.upper_var_noise_y, size=self.N_noise)

        # Generating amplitudes for noise pertubations
        A_noise = np.random.uniform(self.A_noise_lower, self.A_noise_upper, size=self.N_noise)

        for n in range(self.N_noise):
            noise = self.gaussian_2d(self.X, self.Y, xs_n[n], ys_n[n], noise_var_x[n], noise_var_y[n], A_noise[n])
            self.Z += noise

    # Calculating gradient of landscape
    def gradient_3d_surface(self, surface, step_size):
        gradient = np.gradient(surface, step_size)
        return gradient

    # Implementation of momentum gradient descent
    def momentum_gradient_descent(self):
        # Generating random starting position
        xs, ys = np.random.uniform(self.x_min, self.x_max), np.random.uniform(self.x_min, self.x_max)
        zs = self.Z_interp(xs, ys) # Calculating corresponding z from interpolation

        # Position vectors
        pos = np.stack((xs, ys, zs[0]), axis=-1)        # Current position vector
        pos_before = pos.copy()                         # Previous position vector

        # Velocity vectors
        v_init = np.stack((0, 0, 0), axis=-1)           # Current velocity vector
        v = np.stack((0, 0, 0), axis=-1)                # Initialization velocity in case of hitting a boundary

        iteration = 0

        while iteration < self.max_iterations:

            x = pos[0]
            y = pos[1]
            z = self.Z_interp(x, y)[0]

            # Calculating gradient from interpolation
            grad_x = self.gradient_interp_x(x, y)[0]
            grad_y = self.gradient_interp_y(x, y)[0]

            grad = np.stack((grad_x, grad_y, 0), axis=-1)   # Gradient vector

            v = self.beta * v + self.lr * grad  # Update velocity with momentum
            pos -= v                            # Update position

            # Checking convergence
            if np.linalg.norm(pos - pos_before) < self.eps:
                #print(f'Convergence reached after {iteration} iterations')
                #print(f'Convergence reached at {x, y, z}')
                return iteration

            # Updating previous position
            pos_before = pos.copy()

            # Checking boundaries
            if x > self.x_max or x < self.x_min:
                pos[0] = np.clip(pos[0],self.x_min,self.x_max)
                v = v_init.copy()
            if y > self.x_max or y < self.x_min:
                pos[1] = np.clip(pos[0],self.x_min,self.x_max)
                v = v_init.copy()

            iteration += 1

        #print(f'Convergence not reached after {iteration} iterations')
        return iteration

    # Calculating average number of iterations until convergence over 200 iterations
    def run_animation(self):
        self.tot_iterations = 0
        for _ in range(200):
            self.tot_iterations += self.momentum_gradient_descent()

        return self.tot_iterations / 200

