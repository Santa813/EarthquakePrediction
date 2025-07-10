🌍 Earthquake Prediction System
Forecasting seismic activity using Transformer-based time series analysis and real-time data from USGS.

🔍 Overview
This project is focused on predicting earthquakes using real-time seismic data. By leveraging a Time Series Transformer model, we aim to anticipate the likelihood and magnitude of future seismic events. It combines data science, deep learning, and real-world datasets to provide insightful and proactive earthquake predictions.

🚀 Features
	•	Real-time data ingestion from USGS API
	•	Transformer-based deep learning model for prediction
	•	Visualization of seismic trends and magnitudes
	•	Live earthquake news integration
	•	Streamlit-based user interface for ease of interaction

🧠 Tech Stack
	•	Language: Python
 •	Libraries: TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
	•	UI: Streamlit
	•	API: USGS Earthquake API, News API
	•	Platform: Google Colab / Jupyter Notebook

📊 Dataset
	•	Source: USGS Earthquake API
	•	Data Type: Time-series seismic data
	•	Features Used: Timestamp, Magnitude, Depth, Latitude, Longitude
	•	Sequence Length: 30-day rolling window

🧾 Model Details
	•	Transformer model built for time series forecasting
	•	Trained to predict upcoming earthquake magnitudes
  •	Captures complex temporal dependencies across regions and time

🖥 User Interface
	•	Built using Streamlit
	•	Real-time dashboard for visualization
	•	Prediction results displayed clearly
	•	News section showing global seismic updates

 📦 Folder Structure

EarthquakePrediction/
├── data/                  # Preprocessed datasets
├── model/                 # Transformer model scripts
├── streamlit_app.py       # Streamlit UI file
├── utils/                 # Helper functions
├── README.md              # Project overview
└── requirements.txt       # Python dependencies


📈 Future Improvements
	•	Real-time alert system via SMS/Email
	•	Model accuracy improvement using regional fault-line data
	•	Integration with cloud platforms (AWS/GCP)

👩‍💻 Contributor
 https://github.com/Anushka2427 



