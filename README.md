# 🐧 Penguin Classifier

A web-based application for classifying penguin species using the Palmer Penguins dataset and Gaussian Naive Bayes algorithm. Built with Streamlit.

## 📋 Description

This application allows users to explore the Palmer Penguins dataset, train a machine learning model, and make predictions on penguin species based on physical measurements.

## ✨ Features

- **Data Preview**: View the first 10 rows of the dataset
- **Statistics**: Display descriptive statistics for key features
- **Model Training**: Automatic training of Gaussian Naive Bayes model with 80/20 train-test split
- **Model Evaluation**: Accuracy score and classification report
- **Interactive Prediction**: Input measurements to predict penguin species
- **Responsive UI**: Clean and user-friendly interface

## 🚀 Live Demo

Experience the application live at: [https://ircham3.streamlit.app](https://ircham3.streamlit.app)

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd dataapps-2
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit pandas scikit-learn pillow matplotlib
   ```

## 📊 Dataset

The application uses `penguins_cleaned.csv`, which contains measurements from the Palmer Penguins dataset including:
- Species (Adelie, Chinstrap, Gentoo)
- Bill length and depth (mm)
- Flipper length (mm)
- Body mass (g)
- Island and sex (not used in modeling)

## ▶️ Usage

1. Ensure `penguins_cleaned.csv` is in the same directory as `main.py`
2. Run the application:
   ```bash
   streamlit run main.py
   ```
3. Open your browser to the provided local URL (usually http://localhost:8501)

## 🤖 Model Details

- **Algorithm**: Gaussian Naive Bayes
- **Features**: Bill length, bill depth, flipper length, body mass
- **Target**: Species
- **Preprocessing**: Label encoding for species, stratified train-test split

## 📈 Future Enhancements

- Add data visualization plots
- Implement additional ML algorithms
- Add model comparison functionality
- Include hyperparameter tuning

## 📄 License

This project is open-source. Feel free to use and modify.

## 👥 Contributing

Contributions are welcome! Please open an issue or submit a pull request.
