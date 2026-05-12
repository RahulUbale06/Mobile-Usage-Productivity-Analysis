import numpy as np

# ==================================
# MEAN
# ==================================
def calculate_mean(values):
    return sum(values) / len(values)

# ==================================
# MSE
# ==================================
def calculate_mse(y_true, y_pred):

    error_sum = 0

    for actual, predicted in zip(y_true, y_pred):
        error_sum += (actual - predicted) ** 2

    mse = error_sum / len(y_true)

    return mse

# ==================================
# RMSE
# ==================================
def calculate_rmse(y_true, y_pred):

    mse = calculate_mse(y_true, y_pred)

    rmse = np.sqrt(mse)

    return rmse

# ==================================
# MAE
# ==================================
def calculate_mae(y_true, y_pred):

    error_sum = 0

    for actual, predicted in zip(y_true, y_pred):
        error_sum += abs(actual - predicted)

    mae = error_sum / len(y_true)

    return mae

# ==================================
# R2 SCORE
# ==================================
def calculate_r2(y_true, y_pred):

    y_mean = calculate_mean(y_true)

    ss_total = 0
    ss_residual = 0

    for actual, predicted in zip(y_true, y_pred):

        ss_total += (actual - y_mean) ** 2

        ss_residual += (actual - predicted) ** 2

    r2 = 1 - (ss_residual / ss_total)

    return r2

# ==================================
# CORRELATION
# ==================================
def calculate_correlation(x, y):

    x_mean = calculate_mean(x)
    y_mean = calculate_mean(y)

    numerator = 0

    x_denominator = 0
    y_denominator = 0

    for x_val, y_val in zip(x, y):

        numerator += (x_val - x_mean) * (y_val - y_mean)

        x_denominator += (x_val - x_mean) ** 2
        y_denominator += (y_val - y_mean) ** 2

    correlation = numerator / np.sqrt(x_denominator * y_denominator)

    return correlation