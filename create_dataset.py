import streamlit as st
import numpy as np

def create_dataset(data, n_steps):
    X, y = [], []
    if len(data.shape) == 1:
        # Handle 1D input array
        for i in range(len(data) - n_steps + 1):
            X.append(data[i:i + n_steps])
            y.append(data[i + n_steps - 1])
    else:
        # Handle 2D input array
        for i in range(len(data) - n_steps + 1):
            X.append(data[i:i + n_steps, :-1])
            y.append(data[i + n_steps - 1, -1])
    return np.array(X), np.array(y)

