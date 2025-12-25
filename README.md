# computer-price-prediction

This  was a final project for CSCI 6364 - Machine Learning, created by Derek Chen, Cynthia Driskell, and Oliver Krisetya

## CSCI 4364/6364 - Final Project

### Abstract

This project explores the problem of predicting consumer computer prices using supervised machine learning techniques. Using a structured dataset of both desktop and laptop specifications, the model learns a mapping from hardware features - including CPU tier and core count, GPU tier and VRAM memory capacity, storage configuration, and display characteristics - to market price. Raw categorical attributes are encoded numerically, and additional engineered features, such as total pixel count, pixels-per-inch (PPI), and composite CPU/GPU performance scores, are introduced to capture nonlinear relationships between specifications and cost.

Multiple regression models are evaluated, including baseline linear regression and neural network–based approaches trained with mean squared error loss. Model performance is assessed using standard regression metrics on held-out validation data. Results demonstrate that feature engineering substantially improves predictive accuracy over raw specifications alone, and that nonlinear models better capture interactions between hardware components.

### What We Did

In our notebook, we designed two neural networks: a simple neural network, similar to the ones we created in our assignments, and another, which is more complex and expressive, using more layers, a higher learning rate, and different activation functions.

Before constructing our models, we performed feature engineering and data cleaning to prepare the dataset for proper training. This included handling missing data or inconsistent values, transforming categorical variables into numerical representations, and engineering domain-specific features intended to reflect the underlying factors influencing price. These preprocessing steps were important in improving model stability, convergence, and overall predictive performance.

The simple neural network serves as a baseline model and closely mirrors the feedforward architectures introduced in the course. It consists of a small number of fully connected layers with ReLU activation functions and is trained using the Adam optimizer with a relatively conservative learning rate. This design prioritizes stability and ease of training while still being expressive enough to capture nonlinear basic relationships in the data.

The complex neural network uses a relatively similar architecture, but it utilizes seven layers instead of four. Additionally, it increases the learning rate to 0.1 and uses a different optimizer. Instead of an Adam optimizer, we use a stochastic gradient descent, another recommended optimizer from the PyTorch documentation. Instead of using ReLU as our activation function, we use leaky ReLU, which is an activation function we learned in class. All of these changes were added to create a more expressive neural network, in hopes of navigating the loss landscape better.

### Results

When testing our models against [Brian Risk’s predictive model](https://www.kaggle.com/code/devraai/computer-price-analysis-and-prediction-model), we calculated the Mean Squared Error and R² value to quantitatively compare prediction accuracy and goodness of fit. MSE allowed us to measure the average magnitude of prediction errors, while the R² metric indicated how much variance in the target variable was explained by each model. Together, these metrics provided a clear basis for evaluating the relative performance of our approach to a preexisting model/solution.

Below are the conclusions of our model prediction accuracy compared with Brian Risk’s linear regression model results:

#### Linear Regression Results (Brian Risk)

| Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Value |
| --- | --- | --- |
| 88621.773 | 297.694 | 0.73 |

#### Simple Feedforward Neural Network (20 Epochs)

| Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Value |
| --- | --- | --- |
| 78945.648 | 280.973 | 0.77 |

#### Complex Feedforward Neural Network Result (20 Epochs)

| Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Value |
| --- | --- | --- |
| 76970.664 | 279.454 | 0.78 |

When comparing the simple and complex models, the more complex model achieved lower loss values. Adding more layers allowed the model to learn richer features and better capture the relationships between different computer components. Overall, our neural network models were more accurate than Brian Risk’s linear regression model because a purely linear function did not well represent the relationships in the dataset.

### Error Analysis

Although the Complex Neural Network (ComplexNN) slightly outperforms the Simple Neural Network (SimpleNN), the improvement is modest relative to the added complexity. The ComplexNN increases the R² value by approximately 1.3%, but needs approximately a 450% increase in training time. This indicates diminishing returns from deeper architectures and suggests that remaining prediction error is not primarily due to insufficient model capacity.

#### Model Comparison

| Model | MAE | RMSE |
| --- | --- | --- |
| Mean Baseline | 461.338 | 585.362 |
| Median Baseline | 458.004 | 588.698 |
| SimpleNN | 204.911 | 288.900 |
| ComplexNN | 199.038 | 278.592 |

![Computer Price Predictions vs Actual](/graphs/cppva.png)

Both neural networks perform well on mid-range devices but exhibit significantly higher error for high-priced machines, likely due to limited training samples in this range. The price distribution is heavily concentrated in the $1k–$3k range, with relatively few high-priced devices above $5k. This imbalance helps explain the increased prediction error observed for premium systems. We can also see that features that scale the closest with price (GPU and CPU tier, amount of RAM, etc.) also have the highest correlation with error.

#### Price vs Error

| price_bucket | simple_error (MAE) | complex_error (MAE) |
| --- | --- | --- |
| < $1k | 237.902069 | 281.790710 |
| $1-2k | 166.326630 | 167.879776 |
| $2-3k | 226.654510 | 212.349533 |
| $3-5k | 441.875854 | 384.802216 |
| $5k+ | 4813.029785 | 4751.829590 |

![Distribution of Computer Prices by Range (Log Scale)](/graphs/dcpr.png)

![Prediction Error by Price Range](/graphs/pepr.png)

![SimpleNN - Features Most Associated with Prediction Error](/graphs/snn_fpe.png)

![ComplexNN - Features Most Associated with Prediction Error](/graphs/cnn_fpe.png)

### Challenges

One of the challenges we faced when training the model was specifically selecting correlated features during the feature engineering process. In the dataset, there are a lot of columns to select from, and there are some categories that are more correlated with prices than others. Additionally, the GPU on Google Colab has a finite amount of memory. If we loaded all the features into memory, we would cause the system to crash. Thus, we had to select a limited number of features.

Another challenge we faced was overfitting the model. Similar to our memory challenges found in selecting the proper subset of features to use as inputs, our models would quickly become overfitted with the addition of new engineered features or inclusion with a large set of the dataset features. The difficulty with the complexity of the dataset is that there was a fine line between representing the true complexity in the model vs overfitting through over engineered features or model sophistication. We overcame the issue by analyzing the most impactful features/columns on the overall price and including only these features as testing and validation input.

### Conclusion

Based on the lower mean squared error and higher R² values, the simple feedforward neural network made more accurate predictions than the linear regression. The complex neural network had a slight improvement over the simple neural network. We trained and tested the complex neural network multiple times, and most results yielded improvements over the neural network. However, there were occasional results that performed worse than the simple neural network.

The results support what we learned in class. We knew that neural networks can capture non-linear relationships between features, which allows for more accurate predictions. The various components of a computer play an important role in the price of the entire computer, and the relationship is not necessarily linear. Therefore, it makes sense that neural networks, which are built to handle complex relationships in data, make better predictions than a regular linear regression model.

Additionally, the complex model had a slight improvement over the simple model, also supporting what we learned in class. This is because a neural network with more layers can capture non-linearity better, which is why the 7-layer approach was better than the 4-layer approach at predicting the prices. A higher learning rate also contributed to the lower mean squared error because it was able to escape local minima more easily than the simple neural network.
