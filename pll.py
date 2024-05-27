import pandas as pd
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
print("scikit-learn version:", sklearn.__version__)
print("joblib version:", joblib.__version__)
# Load the test data
test_spectral_data = pd.read_csv('sample_neospectra_data.csv')

# Scale the data
scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(test_spectral_data)

# Load the model (assuming it's a tuple with model and PCA)
model_path = 'plsr_model.pkl'
loaded_model = joblib.load(model_path)
print(loaded_model)
loaded_model.predict(new_data_scaled)

# Inspect and extract the components
if isinstance(loaded_model, tuple):
    model = loaded_model[1]
    pca = PCA(n_components=10)
else:
    raise ValueError("Unexpected model format. Expected a tuple.")

# Transform the data using the PCA loaded from the tuple
new_data_reshaped = pca.fit_transform(new_data_scaled)

# Predict using the trained model
predictions = model.predict(new_data_reshaped)

# Print predictions
print(predictions)
