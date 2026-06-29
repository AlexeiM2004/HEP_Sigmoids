import pandas as pd # Library used for data handling
import numpy as np
import matplotlib.pyplot as plt # Library used for plotting
import os # Library used to check if file is present
import urllib.request # Library used to download
from dataclasses import dataclass # Library used to create dataclass to store the penguin input data
from sklearn.linear_model import LinearRegression # Library used for linear regression analysis

# Check if penguins is downloaded, if not download it (using urllib & os)

filename = "penguins_downloaded.csv"
url = "https://cernbox.cern.ch/s/wh34GhKCOv0Umh7/download"

if not os.path.exists(filename):
    print("Downloading", filename,".")
    urllib.request.urlretrieve(url, "penguins_downloaded.csv")
    print("Download complete.")
else:
    print(filename, "file found.")

## Read penguins into panda

input_penguins_df = pd.read_csv(filename) # Load penguin dataset into "pandas" module

penguins_df = input_penguins_df.dropna(inplace=False) # Rows with entries containing "N/A" or "none" are removed

# Create a class to store the penguin's features such as mass, age, sex and length measurements

@dataclass
class penguin_features:
    flipper_length_mm: list
    body_mass_g: list
    bill_length_mm: list 
    bill_depth_mm: list
    sex: list
    year: list

features = penguin_features( 
    flipper_length_mm = penguins_df["flipper_length_mm"].values,
    body_mass_g = penguins_df["body_mass_g"].values,
    bill_length_mm = penguins_df["bill_length_mm"].values,
    bill_depth_mm = penguins_df["bill_depth_mm"].values,
    sex = penguins_df["sex"].values,
    year = penguins_df["year"].values
)

# Create a class to store the penguin's general data, such as species and island 

@dataclass
class penguin_general:
    species: list
    island: list
    
general = penguin_general(
    species = penguins_df["species"].values,
    island = penguins_df["island"].values 
)

scat_plot = False
if scat_plot == True:
    ## Get a feel for the data by creating a scatter plot

    fig, ax = plt.subplots()

    def plot_regression_problem(ax, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
        ax.scatter(features.flipper_length_mm, features.body_mass_g, marker=".")
        ax.set_xlim(xlow, xhigh)
        ax.set_ylim(ylow, yhigh)
        ax.set_xlabel("flipper length (mm)")
        ax.set_ylabel("body mass (g)")

    plot_regression_problem(ax)


### Begin analysis of the data using a Linear Regression model

# - Reshape X such that its a column vector (model must be fit to a column vector)
# - Fit the linear regression model to the data and predict, use an example flipper length.
# - Plot the original data and the linear regression line, utilising an example of a text box
# - Text box contains example info of gradients and r squared

lr_analysis = False
if lr_analysis == True:
    # Linear Regression model expects input data to be "data x features" so we need to reshape input features

    X = features.flipper_length_mm.reshape(-1,1) # Def as X, std ML convention, reshaped as a column vec
    y_true = features.body_mass_g

    # Define model and perform fit

    model = LinearRegression()
    model.fit(X, y_true)

    ## Evaluating the Linear Regression model

    # This fitted model can now be used to make predictions of the input data (possible extrapolation?)

    example_flipper_length = np.asarray([300]) # Example flipper length
    predicted_body_mass = model.predict([example_flipper_length])
    print("Linear Regression model predicts a body mass of",predicted_body_mass.item(),"g, using an example flipper length of",example_flipper_length.item(),"mm.")

    # This same syntax applies to all input data

    y_pred = model.predict(X) # Run X into the predict, generates a line

    # Plot the original data and the regression line

    fig, lr = plt.subplots()

    def plot_regression_line(lr, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
        lr.scatter(X, y_true, color='blue', label='Data Points')
        lr.plot(X,y_pred, color='red', label='Linear Regression Line (LOBF)')
        lr.set_xlim(xlow, xhigh)
        lr.set_ylim(ylow, yhigh)
        lr.set_xlabel("flipper length (mm)")
        lr.set_ylabel("body mass (g)")
        lr.set_title('Linear Regression example')
        lr.legend(    
                loc='lower right',
                fontsize=9,
                frameon=True, 
                shadow=True,
                title='Legend Title')
        
        slope = model.coef_[0] # Calculates the gradient
        intercept = model.intercept_ # Calculate the intercept
        r_squared = model.score(X,y_true) # Calculate r squared on targets (not predictions)

        statistics_box = (
                f"Slope = {slope:.2f}\n" 
                f"Intercept = {intercept:.3f}\n" 
                f"R squared = {r_squared:.2f}\n" 
        )

        lr.text(
            0.05, 0.95,
            statistics_box,
            color = 'blue',
            transform=lr.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                edgecolor='black',
                alpha=1
            )
        )    

    plot_regression_line(lr)
    plt.show()


### Multiple Linear (Multilinear) Regression Task

# - Create variable x, now a stacked array containing flipper length and bill depth (multivariable)
# - Create variable y_true, "true" body mass
# - Perform a linear regression fit on the x and y data, and predict
# - Retrieve coefficients from gradients, and intercept
# - Display gradients, intercept, and R squared.
# - Create scatter plot of X[:,0] vs body mass, keeping X[:,1] as a constant mean
# - Create grids and column stack, predict and plot
# - Repeat for plot of X[:,1] vs body mass, keeping X[:,0] as constant mean

multiple_lr_analysis = False
if multiple_lr_analysis == True:

    # Split data into inputs (X features) and body mass (Y)

    X = np.column_stack((features.flipper_length_mm,features.bill_depth_mm)) # Arrays must be stacked
    y_true = features.body_mass_g

    # Create and fit LR model

    model = LinearRegression()
    model.fit(X,y_true)

    # Use model to make predictions
    y_pred = model.predict(X)

    # Retrieve coefficients
    slope_flipper = model.coef_[0]
    slope_bill = model.coef_[1]
    intercept = model.intercept_

    print(f"Equation : body_mass = {slope_flipper:.2f} * flipper_length + {slope_bill:.2f} * bill_depth + {intercept:.2f}")
    print(f"R^2 = {model.score(X,y_true):.2f}")

    # Create scatter plot of flipper length vs body mass

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.scatter(X[:,0], y_true, color = 'blue', alpha = 0.5, label = 'Data points')

    # Create a prediciton for flipper length 
    bill_mean = X[:, 1].mean() # Hold Bill depth at its mean
    flipper_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # Create a grid of evenly spaced flipper points
    X_flipper_grid = np.column_stack((flipper_grid, np.full_like(flipper_grid, bill_mean))) # Stack the arrays, with an array of bill depth set to the mean 
    y_pred_flipper = model.predict(X_flipper_grid)
    ax1.plot(flipper_grid, y_pred_flipper, color='red', label='Predicted (bill depth held at mean)')
    
    ax1.set_xlabel("Flipper length (mm)")
    ax1.set_ylabel("Body mass (g)")
    ax1.set_title('Flipper Length vs Body Mass (2-feature model)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Create scatter plot of bill depth vs mody mass
    fig, ax2 = plt.subplots(figsize=(7, 5))
    
    # Scatter plot
    ax2.scatter(X[:, 1], y_true, color='green', alpha=0.5, label='Data Points')
    
    # Create a prediction for bill dpeth
    flipper_mean = X[:, 0].mean() # Hold flipper length at its mean
    bill_grid = np.linspace(X[:, 1].min(), X[:, 1].max(), 100) # Create a grid of evenly spaced bill points
    X_bill_grid = np.column_stack((np.full_like(bill_grid, flipper_mean), bill_grid)) # Stack the arrays,  with an array of flipper length set to the mean 
    y_pred_bill = model.predict(X_bill_grid)
    ax2.plot(bill_grid, y_pred_bill, color='red', label='Predicted (flipper length held at mean)')
    
    ax2.set_xlabel("Bill depth (mm)")
    ax2.set_ylabel("Body mass (g)")
    ax2.set_title('Bill Depth vs Body Mass (2-feature model)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.show()

### Non-Linear regression analysis task 

# - Create generic polynomial, 
# - Generate data points around that polynomial
# - Add noise to the data points for "randomness"
# - Use Scikit-learn to build polynomial terms and recast for multilinear regression
# - Fit, visualise and predict, similar to linear regression

non_linear_regression = True

if non_linear_regression == True:
    # As previously mentioned, linear regression refers to rltn btwn predictions (outputs) and the parameters (not inouts)
    # Linear regression can also be performed on non-linear functions provided outputs are linear in params

    np.random.seed(67)
    # Generate synthetic data to play with
    x = np.linspace(-2,3,1000).reshape(-1,1) # Generate 1000 samples between -2 and 3. 
    y_true = 2*x**4 - 3*x**3 - 10*x**2 + 0.5*x + 3 # Example polynomial

    # Add noise to true values
    y = y_true + 0.2 * np.random.randn(*y_true.shape) # Generate random noise

    # Use scikit-learn to build polynomial terms and recast for multilinear regression

    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=4, include_bias=False)
    X_poly = poly.fit_transform(x)

    # Fit, visualise and predict similarly.

    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict
    y_pred = model.predict(X_poly)

    #Plot
    plt.scatter(x,y,label="Noisy Data")
    plt.plot(x, y_true, label = "True curve", color='green', linewidth =5)
    plt.plot(x, y_pred, label = "Predicted curve", color='blue')
    plt.legend()
    plt.show()

    # We can use linear regression here because the parameters are linear, eve though the polynomial isnt linear
    # x can be defined as different variables and we can then recast the problem as multilinear regression