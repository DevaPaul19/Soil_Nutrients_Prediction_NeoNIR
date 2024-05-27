import streamlit as st
import base64
import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Function to preprocess data
def preprocess_data(new_data, scaler_path, encoder_path, other, input_type):
    if input_type == 'Spectrogram with metadata':
        # Load the pre-fitted scaler
        scaler = joblib.load(scaler_path)

        # Load the pre-fitted OneHotEncoder
        encoder = joblib.load(encoder_path)

        new_nir = new_data[new_data.columns[55:]]
        new_meta = new_data[other]

        new_data = pd.concat([new_nir, new_meta], axis=1)

        # Handle missing values
        new_data.dropna(inplace=True)

        # Define numeric and categorical features
        numeric_features = new_data.select_dtypes(include=['float64', 'int64']).columns
        categorical_features = new_data.select_dtypes(include=['object']).columns

        st.sidebar.write("Number of numeric features:", len(numeric_features))
        st.sidebar.write("Number of categorical features:", len(categorical_features))

        # Scale numeric features
        new_data[numeric_features] = scaler.transform(new_data[numeric_features])

        # Encode categorical features
        encoded_cats = encoder.transform(new_data[categorical_features])

        # Manually create the feature names
        encoded_cat_columns = []
        for col, categories in zip(categorical_features, encoder.categories_):
            encoded_cat_columns.extend([f"{col}_{category}" for category in categories])

        encoded_cats = pd.DataFrame(encoded_cats, columns=encoded_cat_columns)

        # Drop original categorical columns and concatenate encoded columns
        new_data.drop(columns=categorical_features, inplace=True)

        a = new_data.reset_index(drop=True)
        b = encoded_cats.reset_index(drop=True)
        new_data = pd.concat([a, b], axis=1)

        # Convert new_data to NumPy arrays
        new_data_array = new_data.values

        # Reshape the input data for LSTM (samples, timesteps, features)
        new_data_reshaped = new_data_array.reshape((new_data_array.shape[0], 1, new_data_array.shape[1]))

        return new_data_reshaped
    elif input_type == 'Only Spectrogram Data':
        new_nir = new_data
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(new_nir)
        pca = PCA(n_components=10)
        new_data_reshaped = pca.fit_transform(new_data_scaled)

    return new_data_reshaped

def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# Function to load the model and make predictions
def predict_with_model(model_path, new_data_reshaped, model_type, soil_nut):
    if model_type == 'LSTM':
        # Load the LSTM model
        model = tf.keras.models.load_model(model_path, custom_objects={'rmse_loss': rmse_loss})
        # Predict using the trained model
        predictions = model.predict(new_data_reshaped)
    elif model_type == 'PCR':
        # Load the PLSRegression model
        model = joblib.load('pcr_model.pkl')
        # Predict using the trained model
        predictions = model[1].predict(new_data_reshaped)
    # If you need the predictions in a DataFrame format
    predictions_df = pd.DataFrame(predictions, columns=soil_nut)

    return predictions_df

# Function to calculate evaluation metrics
def calculate_metrics(true_values, predicted_values):
    rmse = mean_squared_error(true_values, predicted_values, squared=False)
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return rmse, mae, r2

# Main function
def main():
    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    def set_background(png_file):
        bin_str = get_base64(png_file)
        page_bg_img = '''
        <style>
        .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)

    set_background('OIP (1).jpg')

    st.title("Soil Nutrient Prediction")

    # Dropdown menu for input type selection
    input_type = st.sidebar.selectbox(
        "Select Input Type",
        ("Spectrogram with metadata", "Only Spectrogram Data")
    )

    # Dropdown menu for model type selection
    model_type = "LSTM" if input_type=='Spectrogram with metadata' else "PCR"

    # File uploader
    st.sidebar.title("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        # Load the data
        new_data = pd.read_csv(uploaded_file)

        # Define feature lists
        soil_nut = ['eoc_tot_c', 'c_tot_ncs', 'n_tot_ncs', 's_tot_ncs', 'ph_h2o', 'cecd_nh4', 'ca_nh4d', 'mg_nh4d', 'k_nh4d', 'na_nh4d']
        soil_prop = ['db_13b', 'clay_tot_psa', 'silt_tot_psa', 'sand_tot_psa', 'w32l2', 'w15l2', 'ec_12pre', 'efferv_1nhcl', 'al_dith']
        other = ['area.name_country', 'area.name_county', 'scanner_name', 'horizon.designation', 'texture.description', 'lay.type', 'taxonomic.classification.name', 'lay.depth.to.bottom', 'lay.depth.to.top', 'fiscal.year', 'lat.ycntr', 'long.xcntr']

        # Print lengths for confirmation
        st.sidebar.write("Number of soil nutrient features:", len(soil_nut))

        # Paths to scaler and encoder (only required for Spectrogram with metadata)
        scaler_path = 'scaler.pkl'
        encoder_path = 'encoder.pkl'

        # Plot NIR spectrograms for the first five soil samples
        num_samples = 5

        fig, ax = plt.subplots(figsize=(10, 5))
        for sample_index in range(num_samples):
            soil_sample = new_data.iloc[sample_index]
            wavelengths = new_data.columns[55:].astype(float)  # Columns from the 56th column onward are wavelengths
            reflectance = soil_sample[55:]  # Reflectance values for the selected sample

            ax.plot(wavelengths, reflectance, label=f"Sample {sample_index + 1}")

        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance')
        ax.set_title('NIR Spectrograms for Five Soil Samples')
        ax.legend()
        ax.grid(True)

        # Display the plot using st.pyplot() with the figure object
        st.pyplot(fig)

        # Preprocess data
        new_data_reshaped = preprocess_data(new_data, scaler_path, encoder_path, other, input_type)

        # Load and predict with model
        model_path = 'neo_model_lstm_wm.keras' if model_type == 'LSTM' else 'pcr_model.pkl'
        predictions_df = predict_with_model(model_path, new_data_reshaped, model_type, soil_nut)

        # Display predictions
        st.subheader("Predicted Soil Nutrient Values")
        # Convert DataFrame to HTML table with increased width
        html_table = predictions_df.to_html(index=False)
        # Display the HTML table
        st.write(html_table, unsafe_allow_html=True)

        if input_type == 'Spectrogram with metadata':
            st.image('Model_performance.png', use_column_width=True)
        else:
            st.image('model_performance2.png', use_column_width=True)

if __name__ == "__main__":
    main()
