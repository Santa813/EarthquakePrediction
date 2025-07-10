import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition
import requests
from datetime import datetime
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization, Dropout

# Register custom layer
@tf.keras.utils.register_keras_serializable()
class TimeSeriesTransformer(tf.keras.Model):
    def __init__(self, num_heads=8, ff_dim=64, num_transformer_blocks=3, input_shape=(30,2), output_dim=2, dropout_rate=0.1, **kwargs):
        super(TimeSeriesTransformer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention_layers = [MultiHeadAttention(num_heads=self.num_heads, key_dim=self.ff_dim) for _ in range(self.num_transformer_blocks)]
        self.ffn_layers = [self.build_feed_forward_network() for _ in range(self.num_transformer_blocks)]
        self.layer_norm1 = [LayerNormalization(epsilon=1e-6) for _ in range(self.num_transformer_blocks)]
        self.layer_norm2 = [LayerNormalization(epsilon=1e-6) for _ in range(self.num_transformer_blocks)]
        self.dropout_layers1 = [Dropout(self.dropout_rate) for _ in range(self.num_transformer_blocks)]
        self.dropout_layers2 = [Dropout(self.dropout_rate) for _ in range(self.num_transformer_blocks)]
        self.output_layer = Dense(self.output_dim)  # Final dense layer

    def build_feed_forward_network(self):
        return tf.keras.Sequential([
            Dense(self.ff_dim, activation='relu'),
            Dense(self.output_dim)
        ])

    def call(self, inputs):
        x = inputs
        for i in range(self.num_transformer_blocks):
            attn_output = self.attention_layers[i](x, x)
            attn_output = self.dropout_layers1[i](attn_output)
            x = self.layer_norm1[i](x + attn_output)  # Residual connection

            ffn_output = self.ffn_layers[i](x)
            ffn_output = self.dropout_layers2[i](ffn_output)
            x = self.layer_norm2[i](x + ffn_output)  # Residual connection

        return self.output_layer(x)

    def get_config(self):
        config = super(TimeSeriesTransformer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_blocks': self.num_transformer_blocks,
            'input_shape': self.input_shape,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

# Function to prepare input data based on latitude and longitude
def prepare_input_data(latitude, longitude, num_days=30):
    """
    Prepare the input data for the model with given latitude and longitude.
    Set previous magnitude and previous depth to zero.
    """
    input_data = np.zeros((1, num_days, 2))  # Shape (1, num_days, 4)

    # Fill the input data with zeros for previous magnitude and depth
    for i in range(num_days):
        # input_data[0, i, 0] = 0  # Previous magnitude (set to 0)
        # input_data[0, i, 1] = 0  # Previous depth (set to 0)
        input_data[0, i, 0] = latitude   # Latitude
        input_data[0, i, 1] = longitude  # Longitude

    return input_data

# Function to predict magnitude and depth
def predict_magnitude_and_depth(model, latitude, longitude, num_days=30):
    input_data = prepare_input_data(latitude, longitude, num_days)
    predictions = model.predict(input_data)
    predicted_magnitude = predictions[0][:,0].tolist()
    predicted_depth = predictions[0][:, 1].tolist()

    prediction_df = pd.DataFrame({
        'Predicted Magnitude': predicted_magnitude,
        'Predicted Depth': predicted_depth
    })

    return prediction_df

def fetch_earthquake_news(api_key, query="earthquake", language="en", page_size=15):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles


# News API key
api_key = '30b1e903f059409b92340d75a056c666'

# Fetch recent earthquake news
news_articles = fetch_earthquake_news(api_key)

# Set up Streamlit page layout
st.set_page_config(layout="centered")

# Streamlit sidebar for recent earthquake news
st.sidebar.title("Recent Earthquake News")

for article in news_articles:
    if "earthquake" in article.get('title').lower() or "earthquakes" in article.get('title').lower() or "disaster" in article.get('title').lower():
        title = article.get('title')
        description = article.get('description')
        url = article.get('url')
        published_at = article.get('publishedAt')
        if published_at:
            published_at = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            published_at = published_at.strftime("%B %d, %Y %H:%M")
        
        st.sidebar.subheader(title)
        if published_at:
            st.sidebar.caption(f"Published on: {published_at}")
        if description:
            st.sidebar.write(description)
        if url:
            st.sidebar.markdown(f"[Read more]({url})")





# Load the model
model = load_model('EQPmodel.keras', custom_objects={'TimeSeriesTransformer': TimeSeriesTransformer})

st.write("""
    <div style="min-height: 0vh; padding: 0px;">
""", unsafe_allow_html=True)

st.title("Earthquake Prediction System")
st.header("Using Deep Learning")
st.write('')
st.subheader("Click on area to predict earthquake in the next 30 days.")
st.write('')

# Initialize the map visibility state if it does not exist
if 'map_open' not in st.session_state:
    st.session_state['map_open'] = False  # Initially, the map is closed

# Toggle button: Check the state and toggle the map
if st.session_state['map_open']:
    # If the map is open, display a 'Close Map' button
    if st.button('Close Map',):
        st.session_state['map_open'] = False
    else:
        st.session_state['map_open'] = True
else:
    # If the map is closed, display an 'Open Map' button
    if st.button('Open Map'):
        st.session_state['map_open'] = True
    else:
        st.session_state['map_open'] = False

if st.session_state['map_open']:
    # Create a Folium map centered at a specific location (e.g., [20, 0])
    india_bounds = [[6.5546079, 68.1113787], [35.6745457, 97.395561]]
    m = folium.Map(location=[22, 78], zoom_start=7, min_zoom=5, max_bounds=True,)
    m.fit_bounds([[6.462, 68.176], [37.084, 97.395]])
    m.max_bounds = True
    m.options['maxBounds'] = [[6.462, 68.176], [37.084, 97.395]]
    # Render the map    
    map_data = st_folium(m, width=700, height=800)

    # Check if a click has been registered
    if map_data and map_data.get('last_clicked'):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        st.write(f"### Coordinates of the clicked point:")
        st.write(f"**Latitude**: {lat}")
        st.write(f"**Longitude**: {lon}")
        st.info(f"**Latitude**, **Longitude**: {lat,lon}")
        class_btn = st.button("Predict")
        if class_btn:
            st.write(predict_magnitude_and_depth(model, lat, lon))
    else:
        st.write("Click on the map to display latitude and longitude.")

st.write("</div>", unsafe_allow_html=True)

footer = """
    <style>
    /* Container that holds the page content and ensures footer sticks to the bottom */
    body {
        display: flex;
        min-height: 100vh;
        flex-direction: column;
        justify-content: space-between;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000000;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: white;
    }
     .main-content {
        flex: 1 0 auto;
    }
    </style>

    <div class="footer">
        Made by Santa, Tisha and Anushka!!
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
