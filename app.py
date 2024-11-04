from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # STEP 1: Generate initial dataset and plot
    X = np.random.uniform(0, 1, N)
    Y = np.random.normal(mu, np.sqrt(sigma2), N)

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure()
    plt.scatter(X, Y, color='blue', alpha=0.5, label='Data points')
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    Y_plot = model.predict(X_plot)
    plt.plot(X_plot, Y_plot, color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Regression Line: Y = {intercept:.2f} + {slope:.2f}*X')
    plt.legend()
    plot1_filename = "plot1.png"
    plt.savefig(os.path.join('static', plot1_filename))
    plt.close()

    # STEP 2: Run simulations and create histograms
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        Y_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_filename = "plot2.png"
    plt.savefig(os.path.join('static', plot2_filename))
    plt.close()

    # Calculate proportions
    slope_more_extreme = sum(abs(s) >= abs(slope) for s in slopes) / S
    intercept_more_extreme = sum(abs(i) >= abs(intercept) for i in intercepts) / S

    return plot1_filename, plot2_filename, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])

        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme,
                               N=N, mu=mu, sigma2=sigma2, S=S)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
