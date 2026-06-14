# Weather Data Analyzer and Temperature Prediction System

An interactive desktop application built in Python using Tkinter that provides end-to-end exploratory analysis of historical weather datasets and predicts future temperatures using a Scikit-Learn Linear Regression machine learning model.

This project is tailored as a university-level application presentation for an **Introduction to Open Source Software (OSS)** course, showcasing structural modularity, clear data processing pipelines, user interface design, and scientific computing packages.

---

## Project Overview

Understanding climate patterns and temperature trends is a fundamental challenge in meteorology and data science. This application offers a desktop interface that empowers users to load any weather-related CSV file, perform instant statistical aggregation, generate exploratory charts, and build a linear regression model.

### Key Features
* **Dynamic Dataset Loading**: Allows users to open any local weather CSV using standard file dialogs.
* **Automatic Column Detection**: Automatically detects column headers corresponding to date, temperature, humidity, wind speed, precipitation, and conditions using smart regex-based pattern matching.
* **Robust Data Cleaning**: Cleans datasets by automatically imputing missing numerical items with column means, filling missing categories with column modes, and parsing dates.
* **Weather Statistics Summary**: Instantly calculates average/highest/lowest readings for temperature, humidity, rainfall, and wind speeds, alongside condition frequencies.
* **Interactive Visualization Gallery**: Generates and saves 7 publication-quality charts inside `output/` and displays them directly in the application pane.
* **Linear Regression Engine**: Selects predictor attributes dynamically, trains a Scikit-Learn regression model (80% train, 20% test split), and saves the model binary.
* **Model Evaluation**: Displays error metrics (MAE, MSE, RMSE, R²) and renders a comparison plot of actual vs. predicted temperatures.
* **Dynamic Prediction Form**: Generates text field inputs based on features available in the loaded dataset for instant predictive scoring.
* **Analytical Report Export**: Compiles analysis outputs and metrics into a standardized `output/weather_report.txt` text report.

---

## Machine Learning Workflow

```
[ Load Dataset CSV ]
         ↓
[ Column Detection ]  ← Regular Expressions (Date, Temp, Humidity, Precip, Wind)
         ↓
[ Clean & Impute ]    ← Fill numerical voids with mean; sort chronologically
         ↓
[ Exploratory Stats ]  ← Compute descriptive statistics & frequencies
         ↓
[ Split Data (80/20) ]
         ↓
[ Fit Linear Regressor ] ← Scikit-Learn Linear Regression
         ↓
[ Evaluate Model ]    ← Calculate MAE, MSE, RMSE, R² & Save Comparison Plot
         ↓
[ Predict Temperature ] ← Live user prediction input fields
         ↓
[ Generate Report ]   ← Export summary to 'output/weather_report.txt'
```

---

## Visualizations Built & Saved

The system outputs the following 8 plots directly inside the `output/` folder and displays them dynamically in the GUI:
1. **Temperature Trend** (`temperature_trend.png`): A line chart illustrating chronological temperature fluctuations.
2. **Monthly Average Temperature** (`monthly_temperature.png`): A styled bar chart aggregating temperatures by month.
3. **Monthly Rainfall Distribution** (`monthly_rainfall.png`): A bar chart displaying precipitation sums per month.
4. **Weather Conditions Distribution** (`weather_conditions.png`): A pie chart illustrating the frequency of climate conditions.
5. **Humidity Distribution** (`humidity_histogram.png`): A histogram displaying the spread and density of humidity.
6. **Temperature vs Humidity** (`temp_vs_humidity.png`): A scatter plot detailing the relation between heat and moisture.
7. **Feature Correlation Heatmap** (`correlation_heatmap.png`): A Seaborn heatmap illustrating coefficients between numeric predictors.
8. **Actual vs Predicted Temperature** (`actual_vs_predicted.png`): A validation line plot comparing actual vs. predicted temperatures on test data.

---

## Project Structure

```text
Weather-Data-Analyzer/
│
├── data/
│   └── seoul_weather_2022_to_2024.csv    # Default weather dataset sample
│
├── output/                               # Folder generated during execution
│   ├── temperature_trend.png
│   ├── monthly_temperature.png
│   ├── monthly_rainfall.png
│   ├── weather_conditions.png
│   ├── humidity_histogram.png
│   ├── temp_vs_humidity.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── weather_report.txt                # Exported analytical summary report
│
├── assets/                               # UI icons, logos, static shapes
│
├── main.py                               # Core entry script (GUI + Engine)
│
├── README.md                             # Comprehensive documentation
│
├── requirements.txt                      # List of external library dependencies
│
├── LICENSE                               # MIT license details
│
└── screenshots/                          # Saved window captures
```

---

## Installation & Setup

### Prerequisites
* **Python 3.10 or higher** must be installed on your system.

### Install Dependencies
Navigate to the root directory and install scientific packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## How To Run

Execute the main entry script to start the desktop GUI:
```bash
python main.py
```

### Steps to Test:
1. Click the **Load Dataset** button in the sidebar and navigate to `data/seoul_weather_2022_to_2024.csv`.
2. Navigate to **Dataset Overview** and **Weather Statistics** to observe parsed metrics.
3. Click **Generate Visualizations** to generate plots inside `output/` and embed them in the main window.
4. Click **Train ML Model** to fit the regressor and print intercept coefficients.
5. Navigate to **Evaluate Model** to calculate accuracy metrics and review actual vs. predicted curves.
6. Click **Predict Temperature**, fill out the input values, and click **Run Prediction Model** to see the predicted temperature output.
7. Click **Generate Report** to export all analytical details to `output/weather_report.txt`.

---

## Libraries Used
* **Tkinter**: Standard Python graphical interface library.
* **Pandas**: Fast and flexible data analysis library for structured datasets.
* **NumPy**: Fundamental package for scientific and matrix calculations.
* **Matplotlib**: Low-level plotting utility for custom visuals.
* **Seaborn**: High-level statistical visualization suite built on top of Matplotlib.
* **Scikit-Learn**: Simple and efficient tools for predictive data analytics.

---

## Future Improvements
* **Advanced Regressors**: Support Random Forest or Gradient Boosting regressors.
* **Live API Integrations**: Integrate OpenWeatherMap or meteorological APIs for real-time local forecasts.
* **Interactive Tooltips**: Add interactive hover info overlays on charts.
* **Multi-Format Export**: Support exporting reports directly to PDF or HTML formats.

---

## License Information

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for more information. Suitable for redistribution, modification, and educational/academic integrations.
