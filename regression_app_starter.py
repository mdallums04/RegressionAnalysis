import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Regression Analysis App",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def generate_synthetic_data(n_samples, noise_level, data_type, random_seed):
    """
    Generate synthetic data for regression.

    This function creates simple synthetic datasets.
    For more complex data, you can use the full Synthetic Dataset Generator.

    Args:
        n_samples: Number of data points
        noise_level: Standard deviation of noise
        data_type: Type of relationship ('linear', 'polynomial', 'sinusoidal')
        random_seed: Seed for reproducibility

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        y_true: True values without noise (for visualization)
    """
    np.random.seed(random_seed)

    # Generate x values
    X = np.random.uniform(0, 10, (n_samples, 1))

    # Generate y based on data type
    if data_type == 'linear':
        # Simple linear: y = 2x + 1
        y_true = 2 * X.flatten() + 1

    elif data_type == 'polynomial':
        # Quadratic: y = 0.5x² - 2x + 5
        y_true = 0.5 * (X.flatten() ** 2) - 2 * X.flatten() + 5

    elif data_type == 'sinusoidal':
        # Sinusoidal: y = 5*sin(x) + x
        y_true = 5 * np.sin(X.flatten()) + X.flatten()

    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise

    return X, y, y_true

def polynomial_features(X, degree):
    """
    Expand input features into polynomial features.

    Args:
        X: Original feature matrix (n_samples, 1)
        degree: Polynomial degree

    Returns:
        X_poly: Expanded feature matrix (n_samples, degree)
    """
    X_poly = np.hstack([X ** i for i in range(1, degree + 1)])
    return X_poly


# ==========================================
# LINEAR REGRESSION CLASS
# ==========================================

class LinearRegression:
    """
    Linear Regression using Gradient Descent.

    This class implements linear regression from scratch using
    gradient descent optimization.

    Model: y = Xw + b
    Loss: MSE = (1/n) * sum((y - y_pred)^2)
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.0):
        """
        Initialize the model.

        Args:
            learning_rate: Step size for gradient descent (α)
            n_iterations: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.losses = []  # Track loss over iterations

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        n_samples, n_features = X.shape

        # ==========================================
        # TODO 1: Initialize weights and bias
        # ==========================================
        # Initialize weights to small random values
        # Initialize bias to zero
        # Hint: Use np.random.randn() for weights

        self.weights = np.random.randn(n_features)  # REPLACE THIS
        self.bias = 0.0     # REPLACE THIS

        self.weight_history = []
        self.bias_history = []

        # ==========================================
        # Training loop
        # ==========================================
        for iteration in range(self.n_iterations):

            # ==========================================
            # TODO 2: Compute predictions
            # ==========================================
            # Forward pass: y_pred = X @ w + b
            # Hint: Use @ for matrix multiplication or np.dot()

            y_pred = X @ self.weights + self.bias  # REPLACE THIS

            # ==========================================
            # TODO 3: Compute loss (Mean Squared Error)
            # ==========================================
            # MSE = (1/n) * sum((y - y_pred)^2)
            # Hint: Use np.mean() and ** 2

            loss = np.mean((y - y_pred) ** 2) + self.lambda_reg * np.sum(self.weights ** 2) # REPLACE THIS

            # Store loss for visualization
            self.losses.append(loss)

            # ==========================================
            # TODO 4: Compute gradients
            # ==========================================
            # Gradient for weights: dw = -(2/n) * X.T @ (y - y_pred)
            # Gradient for bias: db = -(2/n) * sum(y - y_pred)
            # Hint: Remember to transpose X for dw calculation
            dw = -(2 / n_samples) * X.T @ (y - y_pred) + 2 * self.lambda_reg * self.weights
            db = -(2 / n_samples) * np.sum(y - y_pred)  # REPLACE THIS


            # ==========================================
            # TODO 5: Update parameters
            # ==========================================
            # Update rule: w = w - learning_rate * dw
            #              b = b - learning_rate * db

            # WRITE YOUR CODE HERE
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

            # Optional: Print progress every 100 iterations
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            y_pred: Predictions (n_samples,)
        """
        # ==========================================
        # TODO 2 (repeated): Compute predictions
        # ==========================================
        # Same as in training: y_pred = X @ w + b

        y_pred = X @ self.weights + self.bias  # REPLACE THIS
        return y_pred

def normal_equation(X, y):
    """
    Closed-form solution for linear regression.
    """
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta


# ==========================================
# EVALUATION METRICS
# ==========================================

def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for regression.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with metrics
    """
    # ==========================================
    # TODO 6: Compute evaluation metrics
    # ==========================================

    # Mean Squared Error: MSE = (1/n) * sum((y - y_pred)^2)
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error: RMSE = sqrt(MSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Error: MAE = (1/n) * sum(|y - y_pred|)
    mae = np.mean(np.abs(y_true - y_pred))

    # R² Score (Coefficient of Determination)
    # R² = 1 - (SS_res / SS_tot)
    # where SS_res = sum((y - y_pred)^2)
    #       SS_tot = sum((y - y_mean)^2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }


# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def plot_training_progress(losses):
    """
    Plot loss vs. iteration to show training progress.

    Args:
        losses: List of loss values over iterations

    Returns:
        Plotly figure
    """
    # ==========================================
    # TODO 7a: Create training progress plot
    # ==========================================

    fig = go.Figure()

    # Add line plot of losses
    # Hint: Use go.Scatter with mode='lines'
    # WRITE YOUR CODE HERE
    fig.add_trace(
        go.Scatter(
            y=losses,
            mode='lines',
            name='Training Loss'
        )
    )

    fig.update_layout(
        title='Training Progress: Loss vs Iteration',
        xaxis_title='Iteration',
        yaxis_title='Loss (MSE)',
        template='plotly_white',
        height=400
    )

    return fig


def plot_predictions(X, y_true, y_pred, y_actual=None):
    """
    Plot predictions vs actual values.

    Args:
        X: Feature values
        y_true: True values (without noise)
        y_pred: Model predictions
        y_actual: Actual noisy observations

    Returns:
        Plotly figure
    """
    # ==========================================
    # TODO 7b: Create predictions plot
    # ==========================================

    fig = go.Figure()

    # If actual values provided, plot them as scatter
    if y_actual is not None:
        # Add scatter plot for actual observations
        # Hint: Use go.Scatter with mode='markers'
        # WRITE YOUR CODE HERE
        fig.add_trace(
            go.Scatter(
                x=X.flatten(),
                y=y_actual,
                mode='markers',
                name='Actual Data'
            )
        )
        pass

    # Add line for true function (if available)
    # Sort by X for proper line plotting
    sort_idx = np.argsort(X.flatten())
    # WRITE YOUR CODE HERE
    fig.add_trace(
        go.Scatter(
            x=X.flatten()[sort_idx],
            y=y_true[sort_idx],
            mode='lines',
            name='True Function'
        )
    )

    # Add line for predictions
    # WRITE YOUR CODE HERE
    fig.add_trace(
        go.Scatter(
            x=X.flatten()[sort_idx],
            y=y_pred[sort_idx],
            mode='lines',
            name='Model Prediction'
        )
    )

    fig.update_layout(
        title='Model Predictions vs Actual Data',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_white',
        height=400
    )

    return fig


def plot_residuals(y_true, y_pred):
    """
    Plot residuals (errors) to check model fit.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Plotly figure
    """
    # ==========================================
    # TODO 7c: Create residuals plot
    # ==========================================

    residuals = y_true - y_pred

    fig = go.Figure()

    # Add scatter plot of residuals vs predictions
    # Hint: Use go.Scatter with mode='markers'
    # WRITE YOUR CODE HERE
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals'
        )
    )

    # Add horizontal line at y=0
    # Hint: Use fig.add_hline()
    # WRITE YOUR CODE HERE
    fig.add_hline(y=0)

    fig.update_layout(
        title='Residual Plot: Check for Patterns',
        xaxis_title='Predicted Value',
        yaxis_title='Residual (True - Predicted)',
        template='plotly_white',
        height=400
    )

    return fig

def animate_gradient_descent(X, y, model):
    fig = go.Figure()

    for i in range(0, len(model.weight_history), 10):
        y_line = X @ model.weight_history[i] + model.bias_history[i]
        fig.add_trace(
            go.Scatter(
                x=X.flatten(),
                y=y_line,
                mode='lines',
                visible=False
            )
        )

    fig.data[0].visible = True

    return fig


# ==========================================
# STREAMLIT APP
# ==========================================

def main():
    """Main application function."""

    # Title and description
    st.title("📈 Linear Regression with Gradient Descent")
    st.markdown("""
    This app demonstrates linear regression using gradient descent.

    **How to use:**
    1. Adjust parameters in the sidebar
    2. Click "Generate Data" to create a dataset
    3. Click "Train Model" to fit the regression model
    4. Explore the results in different tabs
    """)

    st.divider()

    # ==========================================
    # SIDEBAR - Data Generation Parameters
    # ==========================================

    st.sidebar.header("🎲 Data Generation")

    n_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Number of data points to generate"
    )

    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Standard deviation of noise (higher = more noisy)"
    )

    data_type = st.sidebar.selectbox(
        "Data Type",
        ["linear", "polynomial", "sinusoidal"],
        help="Type of underlying relationship"
    )

    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        help="For reproducible results"
    )

    # Generate data button
    if st.sidebar.button("🎲 Generate Data", type="primary"):
        X, y, y_true = generate_synthetic_data(
            n_samples, noise_level, data_type, random_seed
        )

        # Store in session state
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.y_true = y_true
        st.session_state.data_generated = True

        st.sidebar.success("✅ Data generated!")

    st.sidebar.divider()

    # ==========================================
    # SIDEBAR - Model Parameters
    # ==========================================

    st.sidebar.header("🤖 Model Parameters")

    learning_rate = st.sidebar.slider(
        "Learning Rate (α)",
        min_value=0.0001,
        max_value=0.1,
        value=0.01,
        step=0.0001,
        format="%.4f",
        help="Step size for gradient descent"
    )

    n_iterations = st.sidebar.slider(
        "Number of Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of training iterations"
    )

    degree = st.sidebar.slider(
        "Polynomial Degree",
        min_value=1,
        max_value=5,
        value=1,
        help="Degree=1 is linear regression"
    )

    lambda_reg = st.sidebar.slider(
        "L2 Regularization (λ)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="0 = no regularization (standard regression)"
    )

    algorithm = st.sidebar.selectbox(
        "Training Algorithm",
        ["Gradient Descent", "Normal Equation"]
    )

    # ==========================================
    # MAIN CONTENT
    # ==========================================

    # Check if data has been generated
    if 'data_generated' not in st.session_state:
        st.info("👈 Start by generating data using the sidebar controls")
        st.stop()

    # Get data from session state
    X = st.session_state.X
    y = st.session_state.y
    y_true = st.session_state.y_true

    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples", len(X))
    with col2:
        st.metric("Features", X.shape[1])
    with col3:
        st.metric("Data Type", data_type.capitalize())

    st.divider()

    # Feature engineering
    if degree > 1:
        X_model = polynomial_features(X, degree)
    else:
        X_model = X

    # Train model button
    if st.button("🚀 Train Model", type="primary"):

        with st.spinner("Training model... This may take a moment."):

            if algorithm == "Gradient Descent":
                model = LinearRegression(
                    learning_rate=learning_rate,
                    n_iterations=n_iterations,
                    lambda_reg=lambda_reg
                )

                model.fit(X_model, y)
                y_pred = model.predict(X_model)

            else:  # Normal Equation
                theta = normal_equation(X_model, y)

                # Split theta into bias and weights for consistency
                bias = theta[0]
                weights = theta[1:]

                y_pred = X_model @ weights + bias

                # Create a lightweight model-like object for UI compatibility
                model = type("Model", (), {})()
                model.weights = weights
                model.bias = bias
                model.losses = []

            metrics = compute_metrics(y, y_pred)

            st.session_state.model = model
            st.session_state.y_pred = y_pred
            st.session_state.metrics = metrics
            st.session_state.model_trained = True

        st.success("✅ Model trained successfully!")

    # Check if model has been trained
    if 'model_trained' not in st.session_state:
        st.info("👆 Click 'Train Model' to train the regression model")
        st.stop()

    # Get trained model and results
    model = st.session_state.model
    y_pred = st.session_state.y_pred
    metrics = st.session_state.metrics

    # ==========================================
    # RESULTS TABS
    # ==========================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Training Progress",
        "📈 Predictions",
        "📉 Residuals",
        "📋 Metrics",
        "🎞️ Gradient Descent Animation"
    ])

    # Tab 1: Training Progress
    with tab1:
        st.subheader("Training Progress")
        st.markdown("Loss should decrease over iterations, showing convergence.")

        # Plot training progress
        fig_progress = plot_training_progress(model.losses)
        st.plotly_chart(fig_progress, width="stretch")

        # Show final loss
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Initial Loss", f"{model.losses[0]:.4f}")
        with col2:
            st.metric("Final Loss", f"{model.losses[-1]:.4f}")

    # Tab 2: Predictions
    with tab2:
        st.subheader("Model Predictions")
        st.markdown("How well does the model fit the data?")

        # Plot predictions
        fig_pred = plot_predictions(X, y_true, y_pred, y_actual=y)
        st.plotly_chart(fig_pred, width="stretch")

        # Show learned parameters
        st.markdown("### Learned Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weights (w)", f"{model.weights}")
        with col2:
            st.metric("Bias (b)", f"{model.bias:.4f}")

    # Tab 3: Residuals
    with tab3:
        st.subheader("Residual Analysis")
        st.markdown("""
        Residuals should be randomly scattered around zero.
        Patterns indicate the model may not be capturing the relationship well.
        """)

        # Plot residuals
        fig_resid = plot_residuals(y, y_pred)
        st.plotly_chart(fig_resid, width="stretch")

    # Tab 4: Metrics
    with tab4:
        st.subheader("Evaluation Metrics")

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Mean Squared Error (MSE)",
                f"{metrics['MSE']:.4f}",
                help="Average squared error. Lower is better."
            )
            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"{metrics['RMSE']:.4f}",
                help="Square root of MSE. Same units as target."
            )

        with col2:
            st.metric(
                "Mean Absolute Error (MAE)",
                f"{metrics['MAE']:.4f}",
                help="Average absolute error. More robust to outliers."
            )
            st.metric(
                "R² Score",
                f"{metrics['R²']:.4f}",
                help="1 = perfect fit, 0 = no better than mean"
            )

        # Metric explanations
        with st.expander("📚 Understanding Metrics"):
            st.markdown("""
            **MSE (Mean Squared Error)**
            - Measures average squared difference between predictions and actual values
            - Penalizes large errors more heavily
            - Formula: `MSE = (1/n) Σ(y - ŷ)²`

            **RMSE (Root Mean Squared Error)**
            - Square root of MSE
            - Same units as the target variable
            - More interpretable than MSE

            **MAE (Mean Absolute Error)**
            - Average absolute difference
            - Less sensitive to outliers than MSE
            - Formula: `MAE = (1/n) Σ|y - ŷ|`

            **R² Score (Coefficient of Determination)**
            - Proportion of variance explained by the model
            - 1.0 = perfect predictions
            - 0.0 = predictions no better than the mean
            - Can be negative if model is very poor
            """)

    with tab5:
        fig_anim = animate_gradient_descent(X_model, y, model)
        st.plotly_chart(fig_anim)

    st.divider()

    # ==========================================
    # DATA EXPORT
    # ==========================================

    st.subheader("💾 Export Results")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'X': X.flatten(),
        'y_actual': y,
        'y_pred': y_pred,
        'residual': y - y_pred
    })

    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="regression_results.csv",
        mime="text/csv"
    )


# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    main()
