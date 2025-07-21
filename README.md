# Hotel Reservation Prediction 🏨

A machine learning web application to predict whether a hotel reservation will be cancelled or not, based on user input features. The app provides a simple interface for users to enter reservation details and get instant predictions.

---

## Features
- Predicts hotel booking cancellation likelihood
- User-friendly web interface (Flask)
- Automated data ingestion, preprocessing, and model training pipeline
- Model training with LightGBM and MLflow tracking
- Docker and Jenkins support for CI/CD and deployment

---

## Demo

![App Screenshot](images/hotel_app_ui.png)

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd hotel-reservation-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the training pipeline (optional)
```bash
python pipeline/training_pipeline.py
```

### 4. Start the web app
```bash
python app.py
```

The app will be available at `http://localhost:8080` by default.

---

## Project Structure
- `app.py` : Flask web application
- `pipeline/` : Training pipeline scripts
- `src/` : Source code for data ingestion, preprocessing, and model training
- `artifacts/` : Stores raw, processed data and trained models
- `templates/` & `static/` : Frontend files (HTML, CSS)
- `config/` : Configuration files
- `Dockerfile`, `Jenkinsfile` : Deployment and CI/CD

---

## Usage
1. Open the web app in your browser.
2. Fill in the reservation details in the form.
3. Click "Predict the booking status 🧠" to see if the booking is likely to be cancelled.

---

## License
This project is for educational purposes.

---

## Author
- rohangaikar 