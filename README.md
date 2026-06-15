# Weather Data Analyzer & Temperature Prediction System

An interactive desktop application built in Python using Tkinter. It cleans historical weather datasets, generates exploratory charts, fits a linear regression model, and exports detailed analytical reports.

Designed as an academic project presentation for the **Introduction to Open Source Software (OSS)** course.

---

## ☼ Key Features
* **Dynamic Loading**: Open any local weather CSV file via graphical dialogs.
* **Auto Detection**: Recognizes date, temp, humidity, wind, and conditions columns using regex.
* **Data Cleaning**: Imputes missing numbers with means, and missing categories with modes.
* **Aggregation**: Summarizes descriptive statistics (highest/lowest/average metrics).
* **Exploratory Visualizations**: Saves 7 publication-quality charts inside `output/`.
* **Predictive Regressor**: Fits a Scikit-Learn Linear Regression model (80% train, 20% test split).
* **Export Options**: Saves a structured text summary to `output/weather_report.txt` and a premium PDF report to `output_report.pdf`.

---

## ⚙ Installation & Setup

### Prerequisites
* **Python 3.10 or higher**

### Setup
Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🚀 How To Run

Start the graphical desktop application by executing:
```bash
python main.py
```

### Steps to Run Weather Analysis:
1. Click **Load Dataset** in the sidebar, select `data/seoul_weather_2022_to_2024.csv`.
2. Inspect data rows in **Dataset Overview** and statistics in **Weather Statistics**.
3. Click **Generate Visualizations** to generate PNG plots in the `output/` directory.
4. Click **Train ML Model** to fit the regressor and print weights.
5. Review evaluation curves in **Evaluate Model**.
6. Navigate to **Predict Temperature**, fill out the input values, and click **Run Prediction Model** to get a live prediction.
7. Click **Generate Report** to export text and PDF summary files.

---

## 📈 Model Performance & Output Results

Below are the evaluation metrics for the Linear Regression model fitted on the **Seoul Weather** dataset:

* **Training Set**: 584 samples (80%) | **Test Set**: 147 samples (20%)
* **Mean Absolute Error (MAE)**: `4.1071 °C`
* **Mean Squared Error (MSE)**: `27.5747`
* **Root Mean Squared Error (RMSE)**: `5.2512 °C`
* **R² Score (Variance Explained)**: `0.7698` (76.98% accuracy)

*All results, tables, and visualization charts are compiled into the premium [output_report.pdf](output_report.pdf) report.*

---

## 📁 Directory Structure
```text
Weather-Data-Analyzer/
├── data/                       # Sample weather datasets
├── output/                     # Generated charts and text reports
├── output screenshots/         # Pushed output graphs and pickle files
├── assets/                     # UI graphics & icons
├── main.py                     # Core Tkinter application entrypoint
├── requirements.txt            # Package dependencies (pandas, reportlab, etc.)
└── output_report.pdf           # Premium compiled PDF analysis report
```

---

## 🛠 Libraries Used
* **Tkinter**: Desktop GUI environment.
* **Pandas & NumPy**: Dataframes & matrix operations.
* **Scikit-Learn**: Model fitting & metrics.
* **Matplotlib & Seaborn**: Charting & heatmaps.
* **ReportLab**: PDF compilation & layouts.

---

## 📄 License
Licensed under the [MIT License](LICENSE). Suitable for academic presentation, redistribution, and open-source forks.
