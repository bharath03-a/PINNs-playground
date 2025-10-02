"""
Simple TensorFlow v2 conversion of Physics-Informed Neural Network (PINN)
for solving the Nonlinear Schrodinger Equation.
"""

import tensorflow as tf
import numpy as np
import time


class PhysicsInformedNN:
    """
    Physics-Informed Neural Network for solving PDEs using TensorFlow v2.
    Direct conversion from TensorFlow v1 implementation.
    """
    
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        """
        Initialize the PhysicsInformedNN model.
        """
        # Store parameters
        self.lb = lb
        self.ub = ub
        self.layers = layers
        
        # Prepare training data (same as original)
        X0 = np.concatenate((x0, 0*x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # (ub[0], tb)
        
        # Convert to tensors
        self.x0 = tf.constant(X0[:,0:1], dtype=tf.float32)
        self.t0 = tf.constant(X0[:,1:2], dtype=tf.float32)
        self.u0 = tf.constant(u0, dtype=tf.float32)
        self.v0 = tf.constant(v0, dtype=tf.float32)
        
        self.x_lb = tf.constant(X_lb[:,0:1], dtype=tf.float32)
        self.t_lb = tf.constant(X_lb[:,1:2], dtype=tf.float32)
        
        self.x_ub = tf.constant(X_ub[:,0:1], dtype=tf.float32)
        self.t_ub = tf.constant(X_ub[:,1:2], dtype=tf.float32)
        
        self.x_f = tf.constant(X_f[:,0:1], dtype=tf.float32)
        self.t_f = tf.constant(X_f[:,1:2], dtype=tf.float32)
        
        # Initialize neural network weights and biases
        self.weights, self.biases = self.initialize_NN(layers)
        
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        
    def initialize_NN(self, layers):
        """Initialize the weights and biases for the neural network."""
        weights = []
        biases = []
        num_layers = len(layers)
        
        for l in range(0, num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
            
        return weights, biases
        
    def xavier_init(self, size):
        """Initialize weights using Xavier initialization."""
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        """Forward pass through the neural network."""
        num_layers = len(weights) + 1
        
        # Normalize inputs to [-1, 1]
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        
        # Hidden layers
        for l in range(0, num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        
        # Output layer
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        
        return Y
    
    def net_uv(self, x, t):
        """Get u, v and their spatial derivatives."""
        X = tf.concat([x, t], 1)
        
        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(x)
            u_grad = self.neural_net(X, self.weights, self.biases)[:, 0:1]
            v_grad = self.neural_net(X, self.weights, self.biases)[:, 1:2]
        
        u_x = tape.gradient(u_grad, x)
        v_x = tape.gradient(v_grad, x)
        
        return u, v, u_x, v_x

    def net_f_uv(self, x, t):
        """Compute PDE residuals."""
        u, v, u_x, v_x = self.net_uv(x, t)
        
        # Compute gradients
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            u_val, v_val, u_x_val, v_x_val = self.net_uv(x, t)
            
        u_t = tape.gradient(u_val, t)
        u_xx = tape.gradient(u_x_val, x)
        
        v_t = tape.gradient(v_val, t)
        v_xx = tape.gradient(v_x_val, x)
        
        # PDE residuals
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u
        
        return f_u, f_v
    
    def compute_loss(self):
        """Compute the total loss function."""
        # Initial condition loss
        u0_pred, v0_pred, _, _ = self.net_uv(self.x0, self.t0)
        loss_ic = tf.reduce_mean(tf.square(self.u0 - u0_pred)) + \
                  tf.reduce_mean(tf.square(self.v0 - v0_pred))
        
        # Boundary condition loss
        u_lb_pred, v_lb_pred, u_x_lb_pred, v_x_lb_pred = self.net_uv(self.x_lb, self.t_lb)
        u_ub_pred, v_ub_pred, u_x_ub_pred, v_x_ub_pred = self.net_uv(self.x_ub, self.t_ub)
        
        loss_bc = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
                  tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
                  tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
                  tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred))
        
        # PDE residual loss
        f_u_pred, f_v_pred = self.net_f_uv(self.x_f, self.t_f)
        loss_pde = tf.reduce_mean(tf.square(f_u_pred)) + \
                   tf.reduce_mean(tf.square(f_v_pred))
        
        # Total loss
        total_loss = loss_ic + loss_bc + loss_pde
        
        return total_loss
    
    def train(self, nIter):
        """Train the model."""
        print(f"Starting training for {nIter} iterations...")
        
        start_time = time.time()
        for it in range(nIter):
            # Training step
            with tf.GradientTape() as tape:
                loss = self.compute_loss()
            
            # Get gradients
            trainable_vars = []
            for W in self.weights:
                trainable_vars.append(W)
            for b in self.biases:
                trainable_vars.append(b)
            
            gradients = tape.gradient(loss, trainable_vars)
            
            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Print progress
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % (it, loss.numpy(), elapsed))
                start_time = time.time()
    
    def predict(self, X_star):
        """Make predictions on new data points."""
        x_star = tf.constant(X_star[:,0:1], dtype=tf.float32)
        t_star = tf.constant(X_star[:,1:2], dtype=tf.float32)
        
        # Get u, v predictions
        u_star, v_star, _, _ = self.net_uv(x_star, t_star)
        
        # Get PDE residuals
        f_u_star, f_v_star = self.net_f_uv(x_star, t_star)
        
        return u_star.numpy(), v_star.numpy(), f_u_star.numpy(), f_v_star.numpy()