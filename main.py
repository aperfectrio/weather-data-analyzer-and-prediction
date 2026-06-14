import os
import re
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for embedding in Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Color Palette Definitions (Modern Slate/Indigo Dark Mode)
DARK_BG = "#0f172a"        # Slate 900
SIDEBAR_BG = "#1e293b"     # Slate 800
CARD_BG = "#1e293b"        # Slate 800
TEXT_COLOR = "#f8fafc"     # Slate 50
SUBTEXT_COLOR = "#94a3b8"  # Slate 400
ACCENT_COLOR = "#6366f1"   # Indigo 500
ACCENT_HOVER = "#4f46e5"   # Indigo 600
SUCCESS_COLOR = "#10b981"  # Emerald 500
SUCCESS_BG = "#064e3b"     # Emerald 900
WARNING_COLOR = "#f59e0b"  # Amber 500
DANGER_COLOR = "#f43f5e"   # Rose 500
BORDER_COLOR = "#334155"   # Slate 700


class WeatherDataProcessor:
    """Handles data loading, cleaning, statistics, visualization, and ML model workflows."""
    
    def __init__(self):
        self.df = None
        self.file_name = ""
        self.columns = []
        self.date_col = None
        self.temp_col = None
        self.humidity_col = None
        self.precip_col = None
        self.wind_col = None
        self.conditions_col = None
        
        self.model = None
        self.model_features = []
        self.is_trained = False
        
        # Train/Test splits and metrics
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}

    def load_data(self, file_path):
        """Loads and cleans the CSV dataset, automatically detecting weather columns."""
        try:
            self.df = pd.read_csv(file_path)
            self.file_name = os.path.basename(file_path)
            self.columns = list(self.df.columns)
            self.detect_columns()
            self.clean_data()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def detect_columns(self):
        """Detects weather columns based on name patterns."""
        date_patterns = [r'\bdate\b', r'\bdatetime\b', r'\btimestamp\b', r'\btime\b']
        temp_patterns = [r'^temp$', r'^temperature$', r'^temp_c$', r'^temp_f$']
        humidity_patterns = [r'\bhumidity\b', r'\bhum\b', r'\bhumid\b']
        precip_patterns = [r'\bprecip\b', r'\bprecipitation\b', r'\brain\b', r'\brainfall\b', r'\bprecipcover\b']
        wind_patterns = [r'\bwindspeed\b', r'\bwind_speed\b', r'\bwind\b']
        conditions_patterns = [r'\bconditions\b', r'\bweather\b', r'\bcondition\b', r'\bsky\b']

        self.date_col = self._match_column(date_patterns)
        self.temp_col = self._match_column(temp_patterns)
        
        # Strict fallback for target temperature
        if not self.temp_col:
            for col in self.columns:
                if 'temp' in col.lower() and not any(x in col.lower() for x in ['max', 'min', 'feels', 'dew']):
                    self.temp_col = col
                    break
            if not self.temp_col:
                # Absolute fallback: first column containing temp
                for col in self.columns:
                    if 'temp' in col.lower():
                        self.temp_col = col
                        break

        self.humidity_col = self._match_column(humidity_patterns)
        self.precip_col = self._match_column(precip_patterns)
        self.wind_col = self._match_column(wind_patterns)
        self.conditions_col = self._match_column(conditions_patterns)

    def _match_column(self, patterns):
        for pattern in patterns:
            for col in self.columns:
                if re.search(pattern, col.lower()):
                    return col
        return None

    def clean_data(self):
        """Cleans columns, parses dates, and imputes missing values."""
        if self.df is None:
            return
        
        # Date column parsing and sorting
        if self.date_col:
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col], errors='coerce')
            # Drop rows with invalid dates
            self.df = self.df.dropna(subset=[self.date_col])
            self.df = self.df.sort_values(by=self.date_col).reset_index(drop=True)
            
        # Impute missing numerical columns with mean
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().any():
                self.df[col] = self.df[col].fillna(self.df[col].mean())
                
        # Impute missing categorical columns with mode
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            if self.df[col].isnull().any():
                mode_vals = self.df[col].mode()
                self.df[col] = self.df[col].fillna(mode_vals[0] if not mode_vals.empty else "Unknown")

    def calculate_statistics(self):
        """Calculates weather descriptive stats."""
        stats = {}
        
        # Temperature stats
        if self.temp_col:
            stats['temp'] = {
                'avg': self.df[self.temp_col].mean(),
                'max': self.df[self.temp_col].max(),
                'min': self.df[self.temp_col].min()
            }
        else:
            stats['temp'] = {'avg': 'N/A', 'max': 'N/A', 'min': 'N/A'}
            
        # Humidity stats
        if self.humidity_col:
            stats['humidity'] = {
                'avg': self.df[self.humidity_col].mean(),
                'max': self.df[self.humidity_col].max(),
                'min': self.df[self.humidity_col].min()
            }
        else:
            stats['humidity'] = {'avg': 'N/A', 'max': 'N/A', 'min': 'N/A'}
            
        # Rainfall stats
        if self.precip_col:
            stats['rainfall'] = {
                'total': self.df[self.precip_col].sum(),
                'avg': self.df[self.precip_col].mean(),
                'max': self.df[self.precip_col].max()
            }
        else:
            stats['rainfall'] = {'total': 'N/A', 'avg': 'N/A', 'max': 'N/A'}
            
        # Wind stats
        if self.wind_col:
            stats['wind'] = {
                'avg': self.df[self.wind_col].mean(),
                'max': self.df[self.wind_col].max()
            }
        else:
            stats['wind'] = {'avg': 'N/A', 'max': 'N/A'}
            
        # Conditions stats
        if self.conditions_col:
            val_counts = self.df[self.conditions_col].value_counts()
            if not val_counts.empty:
                stats['condition'] = {
                    'most_common': val_counts.index[0],
                    'least_common': val_counts.index[-1] if len(val_counts) > 1 else val_counts.index[0]
                }
            else:
                stats['condition'] = {'most_common': 'N/A', 'least_common': 'N/A'}
        else:
            stats['condition'] = {'most_common': 'N/A', 'least_common': 'N/A'}
            
        return stats

    def create_figure(self, fig_id):
        """Creates and returns a specific styled Matplotlib figure for displaying in Tkinter."""
        # Dark theme stylesheet rules
        plt.rcParams['figure.facecolor'] = '#1e293b'
        plt.rcParams['axes.facecolor'] = '#0f172a'
        plt.rcParams['text.color'] = '#f8fafc'
        plt.rcParams['axes.labelcolor'] = '#94a3b8'
        plt.rcParams['xtick.color'] = '#94a3b8'
        plt.rcParams['ytick.color'] = '#94a3b8'
        plt.rcParams['grid.color'] = '#334155'
        plt.rcParams['axes.edgecolor'] = '#334155'
        
        if fig_id == 1:
            # Temperature Trend
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if self.date_col and self.temp_col:
                ax.plot(self.df[self.date_col], self.df[self.temp_col], color='#6366f1', linewidth=1.5)
                ax.set_title("Temperature Trend Over Time", fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel("Date", fontsize=9)
                ax.set_ylabel("Temperature (°C)", fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.6)
                fig.autofmt_xdate()
            else:
                ax.text(0.5, 0.5, "Date or Temperature Column not available", ha='center', va='center')
            return fig
            
        elif fig_id == 2:
            # Monthly Average Temperature
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if self.date_col and self.temp_col:
                df_temp = self.df.copy()
                df_temp['Month'] = df_temp[self.date_col].dt.strftime('%b')
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_avg = df_temp.groupby('Month')[self.temp_col].mean().reindex(months_order).dropna()
                
                if not monthly_avg.empty:
                    # Select palette
                    sns.barplot(x=monthly_avg.index, y=monthly_avg.values, ax=ax, palette="coolwarm", hue=monthly_avg.index, legend=False)
                    ax.set_title("Monthly Average Temperature", fontsize=11, fontweight='bold', pad=10)
                    ax.set_xlabel("Month", fontsize=9)
                    ax.set_ylabel("Avg Temperature (°C)", fontsize=9)
                    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
                else:
                    ax.text(0.5, 0.5, "No monthly data available", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Date or Temperature Column not available", ha='center', va='center')
            return fig
            
        elif fig_id == 3:
            # Monthly Rainfall Distribution
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if self.date_col and self.precip_col:
                df_temp = self.df.copy()
                df_temp['Month'] = df_temp[self.date_col].dt.strftime('%b')
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_rain = df_temp.groupby('Month')[self.precip_col].sum().reindex(months_order).dropna()
                
                if not monthly_rain.empty:
                    sns.barplot(x=monthly_rain.index, y=monthly_rain.values, ax=ax, color='#0284c7')
                    ax.set_title("Monthly Rainfall Distribution (Total Precip)", fontsize=11, fontweight='bold', pad=10)
                    ax.set_xlabel("Month", fontsize=9)
                    ax.set_ylabel("Total Rainfall (mm)", fontsize=9)
                    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
                else:
                    ax.text(0.5, 0.5, "No rainfall data available", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Date or Precipitation Column not available", ha='center', va='center')
            return fig
            
        elif fig_id == 4:
            # Weather Conditions Distribution
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if self.conditions_col:
                val_counts = self.df[self.conditions_col].value_counts()
                if not val_counts.empty:
                    if len(val_counts) > 5:
                        top5 = val_counts.head(5)
                        others_sum = val_counts.iloc[5:].sum()
                        top5['Other'] = others_sum
                        val_counts = top5
                    
                    colors = ['#38bdf8', '#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa']
                    ax.pie(val_counts.values, labels=val_counts.index, autopct='%1.1f%%', 
                           startangle=90, colors=colors[:len(val_counts)],
                           textprops={'color': '#f8fafc', 'fontsize': 8})
                    ax.set_title("Weather Conditions Distribution", fontsize=11, fontweight='bold', pad=10)
                else:
                    ax.text(0.5, 0.5, "No condition data available", ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "Conditions Column not available", ha='center', va='center')
            return fig
            
        elif fig_id == 5:
            # Humidity Distribution
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if self.humidity_col:
                sns.histplot(self.df[self.humidity_col], bins=20, kde=True, ax=ax, color='#10b981', edgecolor='#047857')
                ax.set_title("Humidity Distribution Histogram", fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel("Humidity (%)", fontsize=9)
                ax.set_ylabel("Frequency", fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.6)
            else:
                ax.text(0.5, 0.5, "Humidity Column not available", ha='center', va='center')
            return fig
            
        elif fig_id == 6:
            # Temperature vs Humidity
            fig, ax = plt.subplots(figsize=(7, 3.5))
            if self.temp_col and self.humidity_col:
                ax.scatter(self.df[self.temp_col], self.df[self.humidity_col], alpha=0.5, color='#fbbf24', edgecolors='#d97706')
                ax.set_title("Temperature vs Humidity Scatter Plot", fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel("Temperature (°C)", fontsize=9)
                ax.set_ylabel("Humidity (%)", fontsize=9)
                ax.grid(True, linestyle=':', alpha=0.6)
            else:
                ax.text(0.5, 0.5, "Temp or Humidity Column not available", ha='center', va='center')
            return fig
            
        elif fig_id == 7:
            # Feature Correlation Heatmap
            fig, ax = plt.subplots(figsize=(7, 3.5))
            key_numeric_cols = []
            for col in [self.temp_col, self.humidity_col, self.precip_col, self.wind_col]:
                if col:
                    key_numeric_cols.append(col)
            # Add other numeric columns
            all_numeric = self.df.select_dtypes(include=[np.number]).columns
            for col in all_numeric:
                if col not in key_numeric_cols and not any(k in col.lower() for k in ['temp', 'feels', 'dew', 'sunrise', 'sunset']):
                    key_numeric_cols.append(col)
            
            key_numeric_cols = key_numeric_cols[:6]  # Limit to 6 features for neat heatmap
            
            if len(key_numeric_cols) > 1:
                corr = self.df[key_numeric_cols].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, 
                            annot_kws={'size': 8}, cbar=True)
                ax.set_title("Feature Correlation Heatmap", fontsize=11, fontweight='bold', pad=10)
                fig.autofmt_xdate()
            else:
                ax.text(0.5, 0.5, "Not enough numeric variables", ha='center', va='center')
            return fig

    def save_all_visualizations(self, output_dir="output"):
        """Generates all 7 plots and saves them as PNG files inside output/."""
        os.makedirs(output_dir, exist_ok=True)
        filenames = [
            "temperature_trend.png",
            "monthly_temperature.png",
            "monthly_rainfall.png",
            "weather_conditions.png",
            "humidity_histogram.png",
            "temp_vs_humidity.png",
            "correlation_heatmap.png"
        ]
        
        for i, fname in enumerate(filenames, start=1):
            fig = self.create_figure(i)
            # Set transparent or specific facecolor for saving
            fig.savefig(os.path.join(output_dir, fname), dpi=150, facecolor=fig.get_facecolor())
            plt.close(fig)

    def train_model(self):
        """Trains a Linear Regression model to predict temperature using available numeric weather features."""
        if self.df is None or not self.temp_col:
            raise ValueError("No dataset loaded or temperature target column not detected.")
        
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        numerical_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        
        # Robust selection of predictors (exclude indices, date info, dew, apparent temp, and target temp itself)
        exclude_keywords = ['temp', 'feels', 'dew', 'index', 'id', 'year', 'month', 'day', 'sunrise', 'sunset']
        self.model_features = []
        for col in numerical_cols:
            if not any(kw in col.lower() for kw in exclude_keywords):
                if self.df[col].std() > 0:  # Exclude features with zero variance
                    self.model_features.append(col)
                    
        # Fallback if no specific numerical features found
        if not self.model_features:
            for col in [self.humidity_col, self.precip_col, self.wind_col]:
                if col and col in numerical_cols and col != self.temp_col:
                    self.model_features.append(col)
                    
        if not self.model_features:
            raise ValueError("No suitable numerical weather attributes found for model features.")
            
        X = self.df[self.model_features]
        y = self.df[self.temp_col]
        
        # Train-Test Split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Fit Linear Regression Model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        
        # Save model using pickle
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", "temperature_model.pkl"), "wb") as f:
            pickle.dump({
                'model': self.model,
                'features': self.model_features,
                'target': self.temp_col
            }, f)
            
        return {
            'features': self.model_features,
            'intercept': self.model.intercept_,
            'coefficients': dict(zip(self.model_features, self.model.coef_)),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test)
        }

    def evaluate_model(self, output_dir="output"):
        """Evaluates model performance and generates Actual vs Predicted plot."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        self.y_pred = self.model.predict(self.X_test)
        
        mae = mean_absolute_error(self.y_test, self.y_pred)
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.y_pred)
        
        self.metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        # Generate and save actual_vs_predicted.png
        subset_len = min(50, len(self.y_test))
        y_test_subset = self.y_test.iloc[:subset_len].values
        y_pred_subset = self.y_pred[:subset_len]
        
        plt.rcParams['figure.facecolor'] = '#1e293b'
        plt.rcParams['axes.facecolor'] = '#0f172a'
        
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(subset_len), y_test_subset, color='#3b82f6', label='Actual Temp', marker='o', linewidth=1.5, markersize=4)
        ax.plot(range(subset_len), y_pred_subset, color='#f43f5e', label='Predicted Temp', linestyle='--', marker='x', linewidth=1.5, markersize=4)
        ax.set_title("Actual vs Predicted Temperature (First 50 Test Samples)", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Sample Index", fontsize=9)
        ax.set_ylabel("Temperature (°C)", fontsize=9)
        ax.legend()
        ax.grid(True, color='#334155', linestyle=':')
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "actual_vs_predicted.png"), dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        
        return self.metrics

    def predict_temperature(self, input_data):
        """Predicts temperature given custom dictionary of features."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet.")
        
        input_df = pd.DataFrame([input_data])
        # Force column ordering to match training features
        input_df = input_df[self.model_features]
        pred = self.model.predict(input_df)
        return pred[0]

    def generate_report(self, output_path="output/weather_report.txt"):
        """Generates detailed weather text report and saves it to output/weather_report.txt."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        stats = self.calculate_statistics()
        
        report = []
        report.append("======================================================================")
        report.append("          WEATHER DATA ANALYSIS AND TEMPERATURE PREDICTION REPORT")
        report.append("======================================================================")
        report.append(f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset File: {self.file_name}")
        report.append(f"Total Records: {len(self.df)}")
        report.append(f"Total Columns: {len(self.columns)}")
        report.append("")
        
        report.append("----------------------------------------------------------------------")
        report.append("1. DATASET OVERVIEW")
        report.append("----------------------------------------------------------------------")
        if self.date_col:
            start_date = self.df[self.date_col].min().strftime('%Y-%m-%d')
            end_date = self.df[self.date_col].max().strftime('%Y-%m-%d')
            report.append(f"Date Range: {start_date} to {end_date}")
        else:
            report.append("Date Range: N/A")
        
        report.append(f"Detected Date Column: {self.date_col or 'None'}")
        report.append(f"Detected Target (Temp) Column: {self.temp_col or 'None'}")
        report.append(f"Detected Humidity Column: {self.humidity_col or 'None'}")
        report.append(f"Detected Precipitation Column: {self.precip_col or 'None'}")
        report.append(f"Detected Wind Column: {self.wind_col or 'None'}")
        report.append(f"Detected Conditions Column: {self.conditions_col or 'None'}")
        report.append("")
        
        report.append("----------------------------------------------------------------------")
        report.append("2. WEATHER STATISTICS SUMMARY")
        report.append("----------------------------------------------------------------------")
        
        report.append("TEMPERATURE:")
        report.append(f"  - Average Temperature: {stats['temp']['avg']:.2f} °C" if isinstance(stats['temp']['avg'], float) else f"  - Average Temperature: {stats['temp']['avg']}")
        report.append(f"  - Highest Temperature: {stats['temp']['max']:.2f} °C" if isinstance(stats['temp']['max'], float) else f"  - Highest Temperature: {stats['temp']['max']}")
        report.append(f"  - Lowest Temperature:  {stats['temp']['min']:.2f} °C" if isinstance(stats['temp']['min'], float) else f"  - Lowest Temperature:  {stats['temp']['min']}")
        report.append("")
        
        report.append("HUMIDITY:")
        report.append(f"  - Average Humidity:    {stats['humidity']['avg']:.2f} %" if isinstance(stats['humidity']['avg'], float) else f"  - Average Humidity:    {stats['humidity']['avg']}")
        report.append(f"  - Highest Humidity:    {stats['humidity']['max']:.2f} %" if isinstance(stats['humidity']['max'], float) else f"  - Highest Humidity:    {stats['humidity']['max']}")
        report.append(f"  - Lowest Humidity:     {stats['humidity']['min']:.2f} %" if isinstance(stats['humidity']['min'], float) else f"  - Lowest Humidity:     {stats['humidity']['min']}")
        report.append("")
        
        report.append("RAINFALL:")
        report.append(f"  - Total Rainfall:      {stats['rainfall']['total']:.2f} mm" if isinstance(stats['rainfall']['total'], float) else f"  - Total Rainfall:      {stats['rainfall']['total']}")
        report.append(f"  - Average Rainfall:    {stats['rainfall']['avg']:.2f} mm" if isinstance(stats['rainfall']['avg'], float) else f"  - Average Rainfall:    {stats['rainfall']['avg']}")
        report.append(f"  - Maximum Rainfall:    {stats['rainfall']['max']:.2f} mm" if isinstance(stats['rainfall']['max'], float) else f"  - Maximum Rainfall:    {stats['rainfall']['max']}")
        report.append("")
        
        report.append("WIND SPEED:")
        report.append(f"  - Average Wind Speed:  {stats['wind']['avg']:.2f} m/s" if isinstance(stats['wind']['avg'], float) else f"  - Average Wind Speed:  {stats['wind']['avg']}")
        report.append(f"  - Maximum Wind Speed:  {stats['wind']['max']:.2f} m/s" if isinstance(stats['wind']['max'], float) else f"  - Maximum Wind Speed:  {stats['wind']['max']}")
        report.append("")
        
        report.append("WEATHER CONDITIONS:")
        report.append(f"  - Most Common Condition:  {stats['condition']['most_common']}")
        report.append(f"  - Least Common Condition: {stats['condition']['least_common']}")
        report.append("")
        
        report.append("----------------------------------------------------------------------")
        report.append("3. MACHINE LEARNING RESULTS & PERFORMANCE")
        report.append("----------------------------------------------------------------------")
        if self.is_trained:
            report.append("Model Type: Linear Regression")
            report.append(f"Target Variable: {self.temp_col}")
            report.append(f"Features Used: {', '.join(self.model_features)}")
            report.append(f"Train/Test Samples: {len(self.X_train)} / {len(self.X_test)}")
            report.append("")
            report.append("Model Coefficients:")
            for feat, coef in zip(self.model_features, self.model.coef_):
                report.append(f"  - {feat}: {coef:.4f}")
            report.append(f"  - Intercept: {self.model.intercept_:.4f}")
            report.append("")
            report.append("Evaluation Metrics on Test Set:")
            report.append(f"  - Mean Absolute Error (MAE):     {self.metrics.get('mae', 0):.4f} °C")
            report.append(f"  - Mean Squared Error (MSE):      {self.metrics.get('mse', 0):.4f}")
            report.append(f"  - Root Mean Squared Error (RMSE): {self.metrics.get('rmse', 0):.4f} °C")
            report.append(f"  - R² Score (R-squared):           {self.metrics.get('r2', 0):.4f}")
        else:
            report.append("Model Status: NOT TRAINED")
            
        report_txt = "\n".join(report)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
            
        return report_txt


class ScrollableFrame(tk.Frame):
    """Custom scrollable frame container for modern scrollable view panes."""
    
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=DARK_BG)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=DARK_BG)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Mouse Wheel binding (only active when hovering over the panel)
        self.canvas.bind('<Enter>', lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind('<Leave>', lambda e: self.canvas.unbind_all("<MouseWheel>"))
        
        # Bind canvas resize to inner frame width
        self.canvas.bind('<Configure>', self._configure_canvas)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def _on_mousewheel(self, event):
        if self.canvas.winfo_height() < self.scrollable_frame.winfo_height():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
    def _configure_canvas(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)


class WeatherApp:
    """The modern desktop Tkinter application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Data Analyzer & Temperature Prediction System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.configure(bg=DARK_BG)
        
        # Initialize processing sub-engine
        self.processor = WeatherDataProcessor()
        
        # Style layout configuration
        self.sidebar_buttons = {}
        self.active_button = None
        
        # Build UI layout
        self.setup_styles()
        self.setup_layout()
        
        # Display Welcome screen
        self.about_project_view()
        self.update_status("Welcome. Please load a dataset to begin.", "info")

    def setup_styles(self):
        """Configures ttk styles for treeviews, scrollbars and entries."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark Treeview styles
        style.configure("Treeview",
                        background=CARD_BG,
                        foreground=TEXT_COLOR,
                        fieldbackground=CARD_BG,
                        rowheight=26,
                        borderwidth=0,
                        font=("Segoe UI", 9))
        style.map('Treeview', background=[('selected', ACCENT_COLOR)])
        
        style.configure("Treeview.Heading",
                        background=SIDEBAR_BG,
                        foreground=TEXT_COLOR,
                        borderwidth=1,
                        font=("Segoe UI", 9, "bold"))
        style.map('Treeview.Heading', background=[('active', ACCENT_COLOR)])

        # Scrollbar styling
        style.configure("Vertical.TScrollbar",
                        troughcolor=DARK_BG,
                        background=SIDEBAR_BG,
                        bordercolor=BORDER_COLOR,
                        arrowcolor=TEXT_COLOR)

    def setup_layout(self):
        """Creates top header, sidebar navigation, content pane, and status bar."""
        # --- HEADER BANNER ---
        header_frame = tk.Frame(self.root, bg=SIDEBAR_BG, height=65, bd=0)
        header_frame.pack(side="top", fill="x")
        header_frame.pack_propagate(False)
        
        header_title = tk.Label(
            header_frame, 
            text="WEATHER DATA ANALYZER & TEMPERATURE PREDICTION SYSTEM", 
            font=("Segoe UI", 16, "bold"), 
            fg=TEXT_COLOR, 
            bg=SIDEBAR_BG,
            anchor="w",
            padx=20
        )
        header_title.pack(side="left", fill="y")
        
        # Sub-badge
        badge_frame = tk.Frame(header_frame, bg=ACCENT_COLOR, padx=10, pady=4)
        badge_frame.pack(side="right", padx=20)
        badge_label = tk.Label(badge_frame, text="OPEN SOURCE", font=("Segoe UI", 8, "bold"), fg=TEXT_COLOR, bg=ACCENT_COLOR)
        badge_label.pack()

        # --- BOTTOM STATUS BAR ---
        self.status_frame = tk.Frame(self.root, bg=SIDEBAR_BG, height=30, bd=0)
        self.status_frame.pack(side="bottom", fill="x")
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            self.status_frame, 
            text="Ready", 
            font=("Segoe UI", 9), 
            fg=SUBTEXT_COLOR, 
            bg=SIDEBAR_BG, 
            anchor="w",
            padx=15
        )
        self.status_label.pack(side="left", fill="both", expand=True)

        # --- LEFT SIDEBAR ---
        sidebar_frame = tk.Frame(self.root, bg=SIDEBAR_BG, width=220, bd=0)
        sidebar_frame.pack(side="left", fill="y")
        sidebar_frame.pack_propagate(False)
        
        # Decorative sidebar separator
        sidebar_border = tk.Frame(self.root, bg=BORDER_COLOR, width=1)
        sidebar_border.pack(side="left", fill="y")

        # Sidebar navigation items
        nav_items = [
            ("Load Dataset", self.load_dataset_view),
            ("Dataset Overview", self.dataset_overview_view),
            ("Weather Statistics", self.weather_statistics_view),
            ("Generate Visualizations", self.generate_visualizations_view),
            ("Train ML Model", self.train_model_view),
            ("Evaluate Model", self.evaluate_model_view),
            ("Predict Temperature", self.predict_temperature_view),
            ("Generate Report", self.generate_report_view),
            ("About Project", self.about_project_view),
            ("Exit", self.exit_app)
        ]
        
        # Logo/Icon area in sidebar top
        brand_frame = tk.Frame(sidebar_frame, bg=SIDEBAR_BG, height=70)
        brand_frame.pack(fill="x")
        brand_lbl = tk.Label(brand_frame, text="☼ ANALYZER v1.0", font=("Segoe UI", 12, "bold"), fg=ACCENT_COLOR, bg=SIDEBAR_BG, anchor="w", padx=20)
        brand_lbl.pack(fill="both", expand=True)
        
        # Separator line
        sep = tk.Frame(sidebar_frame, bg=BORDER_COLOR, height=1)
        sep.pack(fill="x", padx=10, pady=(0, 10))

        # Pack navigation buttons
        for label, command in nav_items:
            btn = tk.Button(
                sidebar_frame,
                text=f"  {label}",
                font=("Segoe UI", 10),
                fg=SUBTEXT_COLOR,
                bg=SIDEBAR_BG,
                activeforeground=TEXT_COLOR,
                activebackground=ACCENT_COLOR,
                bd=0,
                anchor="w",
                padx=15,
                pady=8,
                cursor="hand2",
                command=command
            )
            btn.pack(fill="x", padx=8, pady=2)
            
            # Hover bindings
            btn.bind("<Enter>", lambda e, b=btn: self.on_button_hover(b))
            btn.bind("<Leave>", lambda e, b=btn: self.on_button_leave(b))
            
            self.sidebar_buttons[label] = btn

        # --- MAIN SCROLLABLE CONTENT AREA ---
        self.main_container = tk.Frame(self.root, bg=DARK_BG)
        self.main_container.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.scroll_frame = ScrollableFrame(self.main_container)
        self.scroll_frame.pack(fill="both", expand=True)

    # --- UI INTERACTION FUNCTIONS ---
    def on_button_hover(self, btn):
        if btn != self.active_button:
            btn.configure(bg=BORDER_COLOR, fg=TEXT_COLOR)

    def on_button_leave(self, btn):
        if btn != self.active_button:
            btn.configure(bg=SIDEBAR_BG, fg=SUBTEXT_COLOR)

    def set_active_sidebar_button(self, label):
        """Highlights the active navigation button and resets others."""
        if self.active_button:
            self.active_button.configure(bg=SIDEBAR_BG, fg=SUBTEXT_COLOR)
        
        if label in self.sidebar_buttons:
            self.active_button = self.sidebar_buttons[label]
            self.active_button.configure(bg=ACCENT_COLOR, fg=TEXT_COLOR)

    def update_status(self, message, level="info"):
        """Updates bottom status bar text and color highlights based on action level."""
        self.status_label.configure(text=f" Status: {message}")
        if level == "success":
            self.status_frame.configure(bg=SUCCESS_BG)
            self.status_label.configure(bg=SUCCESS_BG, fg=TEXT_COLOR)
        elif level == "error":
            self.status_frame.configure(bg="#7f1d1d")
            self.status_label.configure(bg="#7f1d1d", fg=TEXT_COLOR)
        elif level == "warning":
            self.status_frame.configure(bg="#78350f")
            self.status_label.configure(bg="#78350f", fg=TEXT_COLOR)
        else:
            self.status_frame.configure(bg=SIDEBAR_BG)
            self.status_label.configure(bg=SIDEBAR_BG, fg=SUBTEXT_COLOR)

    def clear_content_pane(self):
        """Clears all children in the scrollable content view frame."""
        for child in self.scroll_frame.scrollable_frame.winfo_children():
            child.destroy()

    def show_no_data_message(self):
        """Display placeholder if user tries to analyze before loading CSV."""
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=40, pady=40)
        card.pack(pady=100, padx=50, fill="both", expand=True)
        
        icon = tk.Label(card, text="⚠", font=("Segoe UI", 48), fg=WARNING_COLOR, bg=CARD_BG)
        icon.pack()
        
        lbl = tk.Label(
            card, 
            text="No Dataset Loaded", 
            font=("Segoe UI", 16, "bold"), 
            fg=TEXT_COLOR, 
            bg=CARD_BG,
            pady=10
        )
        lbl.pack()
        
        sublbl = tk.Label(
            card, 
            text="Please load a weather CSV dataset file to perform statistics, generate charts, or train predictions.",
            font=("Segoe UI", 10), 
            fg=SUBTEXT_COLOR, 
            bg=CARD_BG,
            wraplength=400,
            justify="center",
            pady=10
        )
        sublbl.pack()
        
        btn = tk.Button(
            card,
            text="Load CSV Dataset Now",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT_COLOR,
            fg=TEXT_COLOR,
            activebackground=ACCENT_HOVER,
            activeforeground=TEXT_COLOR,
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.load_dataset_view
        )
        btn.pack(pady=10)

    # --- VIEW: 1. LOAD DATASET ---
    def load_dataset_view(self):
        self.set_active_sidebar_button("Load Dataset")
        
        # Check if CSV path was selected
        file_path = filedialog.askopenfilename(
            title="Open Weather CSV Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            # Revert to last selection visual if loaded, or about project if empty
            if self.processor.df is not None:
                self.set_active_sidebar_button("Dataset Overview")
                self.dataset_overview_view()
            else:
                self.set_active_sidebar_button("About Project")
                self.about_project_view()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        success = self.processor.load_data(file_path)
        if success:
            self.update_status("Dataset Loaded Successfully", "success")
            
            # Draw Load Confirmation Screen
            title = tk.Label(pane, text="Dataset Loading Status", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
            title.pack(fill="x", pady=(10, 20))
            
            card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=25, pady=25)
            card.pack(fill="x", pady=10)
            
            # Success header
            header_lbl = tk.Label(card, text="✔ Dataset Successfully Processed", font=("Segoe UI", 14, "bold"), fg=SUCCESS_COLOR, bg=CARD_BG, anchor="w")
            header_lbl.pack(fill="x", pady=(0, 15))
            
            # File details
            details = [
                ("File Name", self.processor.file_name),
                ("Total Records", f"{len(self.processor.df)} rows"),
                ("Total Columns", f"{len(self.processor.columns)} columns")
            ]
            
            for label, val in details:
                row_f = tk.Frame(card, bg=CARD_BG, pady=5)
                row_f.pack(fill="x")
                lbl = tk.Label(row_f, text=f"{label}:", font=("Segoe UI", 10, "bold"), fg=SUBTEXT_COLOR, bg=CARD_BG, width=15, anchor="w")
                lbl.pack(side="left")
                value = tk.Label(row_f, text=val, font=("Segoe UI", 10), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
                value.pack(side="left")
                
            # Available Features summary
            feat_frame = tk.Frame(card, bg=CARD_BG, pady=15)
            feat_frame.pack(fill="x")
            
            feat_lbl = tk.Label(feat_frame, text="Detected Mappings & Available Features:", font=("Segoe UI", 11, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
            feat_lbl.pack(fill="x", pady=(0, 10))
            
            mappings = [
                ("Date Column", self.processor.date_col),
                ("Target Variable (Temp)", self.processor.temp_col),
                ("Humidity Column", self.processor.humidity_col),
                ("Rainfall/Precip Column", self.processor.precip_col),
                ("Wind Speed Column", self.processor.wind_col),
                ("Conditions Column", self.processor.conditions_col),
            ]
            
            for mapping_name, col_name in mappings:
                row_f = tk.Frame(feat_frame, bg=CARD_BG, pady=3)
                row_f.pack(fill="x")
                lbl = tk.Label(row_f, text=f"  • {mapping_name}:", font=("Segoe UI", 10), fg=SUBTEXT_COLOR, bg=CARD_BG, width=22, anchor="w")
                lbl.pack(side="left")
                
                if col_name:
                    status_text = f"Mapped to '{col_name}'"
                    status_color = SUCCESS_COLOR
                else:
                    status_text = "Not Detected / Missing"
                    status_color = WARNING_COLOR
                    
                val_lbl = tk.Label(row_f, text=status_text, font=("Segoe UI", 10, "italic"), fg=status_color, bg=CARD_BG)
                val_lbl.pack(side="left")
                
            # Instructions button
            btn_f = tk.Frame(pane, bg=DARK_BG, pady=20)
            btn_f.pack(fill="x")
            
            next_btn = tk.Button(
                btn_f,
                text="Proceed to Dataset Overview",
                font=("Segoe UI", 10, "bold"),
                bg=ACCENT_COLOR,
                fg=TEXT_COLOR,
                activebackground=ACCENT_HOVER,
                activeforeground=TEXT_COLOR,
                bd=0,
                padx=20,
                pady=10,
                cursor="hand2",
                command=self.dataset_overview_view
            )
            next_btn.pack(side="left")
            
        else:
            self.update_status("Error loading dataset. Invalid CSV file.", "error")
            messagebox.showerror("Loading Error", "The file could not be parsed as a weather dataset. Please make sure it is a valid CSV.")
            self.show_no_data_message()

    # --- VIEW: 2. DATASET OVERVIEW ---
    def dataset_overview_view(self):
        self.set_active_sidebar_button("Dataset Overview")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        # Title
        title = tk.Label(pane, text="Dataset Structure Overview", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        # Overview stats container (Horizontal Cards)
        stats_frame = tk.Frame(pane, bg=DARK_BG)
        stats_frame.pack(fill="x", pady=(0, 20))
        
        # Calculate Date range strings
        start_date_str = "N/A"
        end_date_str = "N/A"
        if self.processor.date_col and not self.processor.df[self.processor.date_col].empty:
            start_date_str = self.processor.df[self.processor.date_col].min().strftime('%Y-%m-%d')
            end_date_str = self.processor.df[self.processor.date_col].max().strftime('%Y-%m-%d')

        cards_data = [
            ("Total Records", f"{len(self.processor.df)}", ACCENT_COLOR),
            ("Total Columns", f"{len(self.processor.columns)}", SUCCESS_COLOR),
            ("Start Date", start_date_str, WARNING_COLOR),
            ("End Date", end_date_str, WARNING_COLOR)
        ]
        
        for name, value, col in cards_data:
            card = tk.Frame(stats_frame, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=15, pady=15, width=170, height=90)
            card.pack(side="left", padx=(0, 15), fill="both", expand=True)
            card.pack_propagate(False)
            
            lbl_name = tk.Label(card, text=name, font=("Segoe UI", 9, "bold"), fg=SUBTEXT_COLOR, bg=CARD_BG)
            lbl_name.pack(anchor="w")
            lbl_val = tk.Label(card, text=value, font=("Segoe UI", 13, "bold"), fg=col, bg=CARD_BG)
            lbl_val.pack(anchor="w", pady=(5, 0))

        # Dataset schema card
        schema_card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=20)
        schema_card.pack(fill="x", pady=(0, 20))
        
        schema_title = tk.Label(schema_card, text="Column Data Types & Missing Values", font=("Segoe UI", 12, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        schema_title.pack(fill="x", pady=(0, 10))
        
        # Grid list for columns
        list_hdr = tk.Frame(schema_card, bg=BORDER_COLOR, pady=5, padx=10)
        list_hdr.pack(fill="x")
        tk.Label(list_hdr, text="Column Name", font=("Segoe UI", 9, "bold"), fg=TEXT_COLOR, bg=BORDER_COLOR, width=28, anchor="w").pack(side="left")
        tk.Label(list_hdr, text="Data Type", font=("Segoe UI", 9, "bold"), fg=TEXT_COLOR, bg=BORDER_COLOR, width=18, anchor="w").pack(side="left")
        tk.Label(list_hdr, text="Missing Values", font=("Segoe UI", 9, "bold"), fg=TEXT_COLOR, bg=BORDER_COLOR, anchor="w").pack(side="left")
        
        # Show column types (limit rows if too many columns)
        max_cols_to_show = 12
        for i, col in enumerate(self.processor.columns):
            if i >= max_cols_to_show:
                row_f = tk.Frame(schema_card, bg=CARD_BG, pady=4, padx=10)
                row_f.pack(fill="x")
                tk.Label(row_f, text=f"... and {len(self.processor.columns) - max_cols_to_show} more columns", font=("Segoe UI", 9, "italic"), fg=SUBTEXT_COLOR, bg=CARD_BG).pack(side="left")
                break
                
            dtype = str(self.processor.df[col].dtype)
            missing = self.processor.df[col].isnull().sum()
            
            bg_col = CARD_BG if i % 2 == 0 else "#24334c"
            row_f = tk.Frame(schema_card, bg=bg_col, pady=4, padx=10)
            row_f.pack(fill="x")
            
            tk.Label(row_f, text=col, font=("Segoe UI", 9), fg=TEXT_COLOR, bg=bg_col, width=28, anchor="w").pack(side="left")
            tk.Label(row_f, text=dtype, font=("Segoe UI", 9), fg=SUBTEXT_COLOR, bg=bg_col, width=18, anchor="w").pack(side="left")
            
            missing_color = DANGER_COLOR if missing > 0 else SUCCESS_COLOR
            tk.Label(row_f, text=str(missing), font=("Segoe UI", 9, "bold"), fg=missing_color, bg=bg_col, anchor="w").pack(side="left")

        # Treeview data preview card
        preview_card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=20)
        preview_card.pack(fill="x")
        
        preview_title = tk.Label(preview_card, text="Data Preview (First 10 Records)", font=("Segoe UI", 12, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        preview_title.pack(fill="x", pady=(0, 10))
        
        # Scrollable table container
        table_container = tk.Frame(preview_card, bg=CARD_BG)
        table_container.pack(fill="x")
        
        # Display up to 7 columns to fit layout nicely
        cols_to_preview = self.processor.columns[:7]
        
        tree = ttk.Treeview(table_container, columns=cols_to_preview, show="headings", height=10)
        
        # Define Headings
        for col in cols_to_preview:
            # Truncate header if too long
            disp_hdr = col[:15] + ".." if len(col) > 15 else col
            tree.heading(col, text=disp_hdr, anchor="w")
            tree.column(col, width=120, anchor="w")
            
        # Add values
        for idx, row in self.processor.df.head(10).iterrows():
            vals = []
            for col in cols_to_preview:
                val = row[col]
                # Format float values nicely
                if isinstance(val, (float, np.float64)):
                    vals.append(f"{val:.2f}")
                elif isinstance(val, pd.Timestamp):
                    vals.append(val.strftime('%Y-%m-%d'))
                else:
                    vals.append(str(val))
            tree.insert("", "end", values=vals)
            
        tree.pack(side="left", fill="x", expand=True)

    # --- VIEW: 3. WEATHER STATISTICS ---
    def weather_statistics_view(self):
        self.set_active_sidebar_button("Weather Statistics")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="Historical Weather Statistics Dashboard", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        # Get metrics dictionary
        stats = self.processor.calculate_statistics()
        
        # Cards configurations
        dashboard_configs = [
            ("TEMPERATURE ANALYSIS", [
                ("Average Temperature", f"{stats['temp']['avg']:.2f} °C" if isinstance(stats['temp']['avg'], float) else stats['temp']['avg'], ACCENT_COLOR),
                ("Highest Temperature", f"{stats['temp']['max']:.2f} °C" if isinstance(stats['temp']['max'], float) else stats['temp']['max'], DANGER_COLOR),
                ("Lowest Temperature", f"{stats['temp']['min']:.2f} °C" if isinstance(stats['temp']['min'], float) else stats['temp']['min'], SUCCESS_COLOR),
            ]),
            ("HUMIDITY ANALYSIS", [
                ("Average Humidity", f"{stats['humidity']['avg']:.2f} %" if isinstance(stats['humidity']['avg'], float) else stats['humidity']['avg'], ACCENT_COLOR),
                ("Highest Humidity", f"{stats['humidity']['max']:.2f} %" if isinstance(stats['humidity']['max'], float) else stats['humidity']['max'], DANGER_COLOR),
                ("Lowest Humidity", f"{stats['humidity']['min']:.2f} %" if isinstance(stats['humidity']['min'], float) else stats['humidity']['min'], SUCCESS_COLOR),
            ]),
            ("RAINFALL ANALYSIS", [
                ("Total Rainfall", f"{stats['rainfall']['total']:.2f} mm" if isinstance(stats['rainfall']['total'], float) else stats['rainfall']['total'], ACCENT_COLOR),
                ("Average Rainfall", f"{stats['rainfall']['avg']:.2f} mm" if isinstance(stats['rainfall']['avg'], float) else stats['rainfall']['avg'], ACCENT_COLOR),
                ("Maximum Rainfall", f"{stats['rainfall']['max']:.2f} mm" if isinstance(stats['rainfall']['max'], float) else stats['rainfall']['max'], DANGER_COLOR),
            ]),
            ("WIND & CONDITIONS", [
                ("Average Wind Speed", f"{stats['wind']['avg']:.2f} m/s" if isinstance(stats['wind']['avg'], float) else stats['wind']['avg'], ACCENT_COLOR),
                ("Maximum Wind Speed", f"{stats['wind']['max']:.2f} m/s" if isinstance(stats['wind']['max'], float) else stats['wind']['max'], DANGER_COLOR),
                ("Most Common Condition", stats['condition']['most_common'], SUCCESS_COLOR),
                ("Least Common Condition", stats['condition']['least_common'], WARNING_COLOR)
            ])
        ]
        
        # Render cards in grid/rows
        for section_title, params in dashboard_configs:
            sec_lbl = tk.Label(pane, text=section_title, font=("Segoe UI", 12, "bold"), fg=SUBTEXT_COLOR, bg=DARK_BG, anchor="w")
            sec_lbl.pack(fill="x", pady=(15, 8))
            
            cards_row = tk.Frame(pane, bg=DARK_BG)
            cards_row.pack(fill="x", pady=(0, 15))
            
            for name, val, color in params:
                card = tk.Frame(cards_row, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=20, pady=20, height=100)
                card.pack(side="left", padx=(0, 15), fill="both", expand=True)
                
                # Title
                lbl_name = tk.Label(card, text=name, font=("Segoe UI", 9, "bold"), fg=SUBTEXT_COLOR, bg=CARD_BG, anchor="w")
                lbl_name.pack(fill="x")
                
                # Value
                lbl_val = tk.Label(card, text=str(val), font=("Segoe UI", 14, "bold"), fg=color, bg=CARD_BG, anchor="w")
                lbl_val.pack(fill="x", pady=(8, 0))

    # --- VIEW: 4. GENERATE VISUALIZATIONS ---
    def generate_visualizations_view(self):
        self.set_active_sidebar_button("Generate Visualizations")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="Exploratory Data Visualizations", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 5))
        
        subtitle = tk.Label(pane, text="Charts are automatically generated and saved inside output/ directory.", font=("Segoe UI", 10), fg=SUBTEXT_COLOR, bg=DARK_BG, anchor="w")
        subtitle.pack(fill="x", pady=(0, 15))
        
        # Trigger saving charts first
        self.processor.save_all_visualizations()
        self.update_status("Visualizations Generated and Saved", "success")
        
        # Display interactive widgets gallery (embed Matplotlib canvases dynamically)
        gallery_frame = tk.Frame(pane, bg=DARK_BG)
        gallery_frame.pack(fill="both", expand=True)
        
        # Generate 7 plots inside the GUI
        for i in range(1, 8):
            fig = self.processor.create_figure(i)
            
            chart_card = tk.Frame(gallery_frame, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=15, pady=15)
            chart_card.pack(fill="x", pady=(0, 25))
            
            # Embed matplotlib canvas
            canvas = FigureCanvasTkAgg(fig, master=chart_card)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- VIEW: 5. TRAIN ML MODEL ---
    def train_model_view(self):
        self.set_active_sidebar_button("Train ML Model")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="Train Temperature Prediction Model", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        # Training logic card
        train_card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=25, pady=25)
        train_card.pack(fill="x", pady=10)
        
        train_hdr = tk.Label(train_card, text="Linear Regression Trainer Engine", font=("Segoe UI", 14, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        train_hdr.pack(fill="x", pady=(0, 15))
        
        # Run training
        try:
            results = self.processor.train_model()
            self.update_status("Model Trained Successfully", "success")
            
            # Display success indicators
            status_row = tk.Frame(train_card, bg=CARD_BG, pady=10)
            status_row.pack(fill="x")
            
            dot = tk.Label(status_row, text="●", font=("Segoe UI", 14, "bold"), fg=SUCCESS_COLOR, bg=CARD_BG)
            dot.pack(side="left")
            status_lbl = tk.Label(status_row, text=" Model trained successfully on local dataset variables.", font=("Segoe UI", 11, "bold"), fg=TEXT_COLOR, bg=CARD_BG)
            status_lbl.pack(side="left")
            
            # Details:
            details_f = tk.Frame(train_card, bg=CARD_BG, pady=10)
            details_f.pack(fill="x")
            
            details = [
                ("Target Variable", self.processor.temp_col),
                ("Training Set Size", f"{results['train_size']} records (80%)"),
                ("Testing Set Size", f"{results['test_size']} records (20%)"),
                ("Intercept Variable", f"{results['intercept']:.4f}"),
            ]
            
            for lbl, val in details:
                row_f = tk.Frame(details_f, bg=CARD_BG, pady=4)
                row_f.pack(fill="x")
                tk.Label(row_f, text=f"{lbl}:", font=("Segoe UI", 10, "bold"), fg=SUBTEXT_COLOR, bg=CARD_BG, width=22, anchor="w").pack(side="left")
                tk.Label(row_f, text=val, font=("Segoe UI", 10), fg=TEXT_COLOR, bg=CARD_BG).pack(side="left")
                
            # Features selected:
            feat_f = tk.Frame(train_card, bg=CARD_BG, pady=10)
            feat_f.pack(fill="x")
            
            feat_title = tk.Label(feat_f, text="Selected Predictor Features & Coefficients:", font=("Segoe UI", 11, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
            feat_title.pack(fill="x", pady=(0, 10))
            
            for idx, (feature, coef) in enumerate(results['coefficients'].items()):
                bg_c = CARD_BG if idx % 2 == 0 else "#24334c"
                row_f = tk.Frame(feat_f, bg=bg_c, pady=4, padx=10)
                row_f.pack(fill="x")
                
                tk.Label(row_f, text=feature, font=("Segoe UI", 10), fg=TEXT_COLOR, bg=bg_c, width=28, anchor="w").pack(side="left")
                
                coef_color = SUCCESS_COLOR if coef >= 0 else DANGER_COLOR
                sign = "+" if coef >= 0 else ""
                tk.Label(row_f, text=f"{sign}{coef:.4f}", font=("Segoe UI", 10, "bold"), fg=coef_color, bg=bg_c).pack(side="left")
                
            # Button for next step
            btn_f = tk.Frame(pane, bg=DARK_BG, pady=20)
            btn_f.pack(fill="x")
            
            eval_btn = tk.Button(
                btn_f,
                text="Evaluate Model Performance",
                font=("Segoe UI", 10, "bold"),
                bg=ACCENT_COLOR,
                fg=TEXT_COLOR,
                activebackground=ACCENT_HOVER,
                activeforeground=TEXT_COLOR,
                bd=0,
                padx=20,
                pady=10,
                cursor="hand2",
                command=self.evaluate_model_view
            )
            eval_btn.pack(side="left")
            
        except Exception as e:
            self.update_status(f"Error training model: {str(e)}", "error")
            err_lbl = tk.Label(train_card, text=f"⚠ Training Failed: {str(e)}", font=("Segoe UI", 11, "bold"), fg=DANGER_COLOR, bg=CARD_BG, anchor="w")
            err_lbl.pack(fill="x")

    # --- VIEW: 6. EVALUATE MODEL ---
    def evaluate_model_view(self):
        self.set_active_sidebar_button("Evaluate Model")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        if not self.processor.is_trained:
            self.clear_content_pane()
            pane = self.scroll_frame.scrollable_frame
            
            card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=30, pady=30)
            card.pack(pady=100, padx=50, fill="both", expand=True)
            
            tk.Label(card, text="⚠ Model Not Trained", font=("Segoe UI", 14, "bold"), fg=WARNING_COLOR, bg=CARD_BG).pack()
            tk.Label(card, text="You must train the machine learning model first before evaluating its metrics.", font=("Segoe UI", 10), fg=SUBTEXT_COLOR, bg=CARD_BG, pady=10).pack()
            
            tk.Button(
                card,
                text="Train Model Now",
                font=("Segoe UI", 10, "bold"),
                bg=ACCENT_COLOR,
                fg=TEXT_COLOR,
                bd=0,
                padx=15,
                pady=8,
                cursor="hand2",
                command=self.train_model_view
            ).pack()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="Model Performance & Evaluation", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        # Run evaluation metrics
        metrics = self.processor.evaluate_model()
        self.update_status("Model Evaluated Successfully", "success")
        
        # Display metrics as horizontal cards
        metrics_frame = tk.Frame(pane, bg=DARK_BG)
        metrics_frame.pack(fill="x", pady=(0, 20))
        
        metrics_data = [
            ("R² SCORE", f"{metrics['r2']:.4f}", SUCCESS_COLOR, "Accuracy score (variance explained)"),
            ("MAE", f"{metrics['mae']:.4f} °C", ACCENT_COLOR, "Mean Absolute Error in predictions"),
            ("MSE", f"{metrics['mse']:.4f}", WARNING_COLOR, "Mean Squared Error coefficient"),
            ("RMSE", f"{metrics['rmse']:.4f} °C", ACCENT_COLOR, "Root Mean Squared deviation")
        ]
        
        for m_name, val, color, tooltip in metrics_data:
            card = tk.Frame(metrics_frame, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=15, pady=15, width=170, height=110)
            card.pack(side="left", padx=(0, 15), fill="both", expand=True)
            card.pack_propagate(False)
            
            tk.Label(card, text=m_name, font=("Segoe UI", 9, "bold"), fg=SUBTEXT_COLOR, bg=CARD_BG, anchor="w").pack(fill="x")
            tk.Label(card, text=val, font=("Segoe UI", 14, "bold"), fg=color, bg=CARD_BG, anchor="w").pack(fill="x", pady=(5, 0))
            tk.Label(card, text=tooltip, font=("Segoe UI", 7, "italic"), fg=SUBTEXT_COLOR, bg=CARD_BG, wraplength=140, justify="left").pack(fill="x", pady=(5, 0))

        # Render embedded actual_vs_predicted chart
        chart_card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=15, pady=15)
        chart_card.pack(fill="x", pady=(0, 20))
        
        fig, ax = plt.subplots(figsize=(7, 3.5))
        subset_len = min(50, len(self.processor.y_test))
        y_test_subset = self.processor.y_test.iloc[:subset_len].values
        y_pred_subset = self.processor.y_pred[:subset_len]
        
        ax.plot(range(subset_len), y_test_subset, color='#3b82f6', label='Actual Temperature', marker='o', linewidth=1.5, markersize=4)
        ax.plot(range(subset_len), y_pred_subset, color='#f43f5e', label='Predicted Temperature', linestyle='--', marker='x', linewidth=1.5, markersize=4)
        ax.set_title("Actual vs Predicted Temperature (Test Validation Samples)", fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Sample Index", fontsize=9)
        ax.set_ylabel("Temperature (°C)", fontsize=9)
        ax.legend()
        ax.grid(True, color='#334155', linestyle=':')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=chart_card)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Navigate to Predict Button
        btn_f = tk.Frame(pane, bg=DARK_BG, pady=10)
        btn_f.pack(fill="x")
        
        pred_btn = tk.Button(
            btn_f,
            text="Go to Temperature Predictor Panel",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT_COLOR,
            fg=TEXT_COLOR,
            activebackground=ACCENT_HOVER,
            activeforeground=TEXT_COLOR,
            bd=0,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self.predict_temperature_view
        )
        pred_btn.pack(side="left")

    # --- VIEW: 7. PREDICT TEMPERATURE ---
    def predict_temperature_view(self):
        self.set_active_sidebar_button("Predict Temperature")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        if not self.processor.is_trained:
            self.clear_content_pane()
            pane = self.scroll_frame.scrollable_frame
            
            card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=30, pady=30)
            card.pack(pady=100, padx=50, fill="both", expand=True)
            
            tk.Label(card, text="⚠ Model Not Trained", font=("Segoe UI", 14, "bold"), fg=WARNING_COLOR, bg=CARD_BG).pack()
            tk.Label(card, text="You must train the machine learning model first before generating predictions.", font=("Segoe UI", 10), fg=SUBTEXT_COLOR, bg=CARD_BG, pady=10).pack()
            
            tk.Button(
                card,
                text="Train Model Now",
                font=("Segoe UI", 10, "bold"),
                bg=ACCENT_COLOR,
                fg=TEXT_COLOR,
                bd=0,
                padx=15,
                pady=8,
                cursor="hand2",
                command=self.train_model_view
            ).pack()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="Predict Temperature Value", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        main_layout = tk.Frame(pane, bg=DARK_BG)
        main_layout.pack(fill="both", expand=True)
        
        # Left side inputs form
        form_card = tk.Frame(main_layout, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=25, pady=25)
        form_card.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        form_hdr = tk.Label(form_card, text="Enter Input Weather Parameters", font=("Segoe UI", 12, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        form_hdr.pack(fill="x", pady=(0, 15))
        
        # Store input entry widgets
        self.input_entries = {}
        
        # Dynamically create inputs for features
        for idx, feat in enumerate(self.processor.model_features):
            row = tk.Frame(form_card, bg=CARD_BG, pady=6)
            row.pack(fill="x")
            
            # Beautify text
            label_text = feat.replace('_', ' ').title()
            # If standard, show suffix units
            if 'hum' in feat.lower():
                label_text += " (%)"
            elif 'wind' in feat.lower():
                label_text += " (m/s)"
            elif 'precip' in feat.lower() or 'rain' in feat.lower():
                label_text += " (mm)"
            elif 'pressure' in feat.lower():
                label_text += " (hPa)"
                
            lbl = tk.Label(row, text=label_text, font=("Segoe UI", 9, "bold"), fg=SUBTEXT_COLOR, bg=CARD_BG, width=25, anchor="w")
            lbl.pack(side="left")
            
            # Auto-calculate training dataset feature mean as default value
            mean_val = self.processor.df[feat].mean()
            
            ent = tk.Entry(
                row, 
                font=("Segoe UI", 10), 
                fg=TEXT_COLOR, 
                bg=DARK_BG, 
                insertbackground=TEXT_COLOR, 
                bd=1, 
                relief="solid",
                highlightthickness=0
            )
            ent.insert(0, f"{mean_val:.2f}")
            ent.pack(side="left", fill="x", expand=True, padx=(10, 0))
            
            self.input_entries[feat] = ent
            
        # Predict Button inside form
        pred_btn = tk.Button(
            form_card,
            text="Run Prediction Model",
            font=("Segoe UI", 10, "bold"),
            bg=ACCENT_COLOR,
            fg=TEXT_COLOR,
            activebackground=ACCENT_HOVER,
            activeforeground=TEXT_COLOR,
            bd=0,
            padx=15,
            pady=10,
            cursor="hand2",
            command=self.run_prediction
        )
        pred_btn.pack(fill="x", pady=(20, 0))
        
        # Right side output card
        self.output_card = tk.Frame(main_layout, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=25, pady=25, width=350)
        self.output_card.pack(side="right", fill="both")
        self.output_card.pack_propagate(False)
        
        output_hdr = tk.Label(self.output_card, text="Prediction Output", font=("Segoe UI", 12, "bold"), fg=TEXT_COLOR, bg=CARD_BG)
        output_hdr.pack(pady=(0, 20))
        
        self.pred_icon_lbl = tk.Label(self.output_card, text="⚙", font=("Segoe UI", 64), fg=SUBTEXT_COLOR, bg=CARD_BG)
        self.pred_icon_lbl.pack(pady=20)
        
        self.pred_val_lbl = tk.Label(self.output_card, text="Enter values & click\n'Run Prediction Model'", font=("Segoe UI", 10, "italic"), fg=SUBTEXT_COLOR, bg=CARD_BG, justify="center")
        self.pred_val_lbl.pack(pady=10)

    def run_prediction(self):
        """Extracts input, performs prediction, updates outputs."""
        input_data = {}
        try:
            for feat, ent in self.input_entries.items():
                val_str = ent.get().strip()
                if not val_str:
                    raise ValueError(f"Input for feature '{feat}' cannot be empty.")
                input_data[feat] = float(val_str)
        except ValueError as ex:
            messagebox.showerror("Invalid Input", f"Please enter valid decimal numbers.\nDetails: {str(ex)}")
            return
            
        try:
            pred_temp = self.processor.predict_temperature(input_data)
            self.update_status("Prediction Completed", "success")
            
            # Update prediction values and styles
            self.pred_val_lbl.configure(
                text=f"Predicted Temperature:\n\n{pred_temp:.2f} °C", 
                font=("Segoe UI", 16, "bold"),
                fg=TEXT_COLOR
            )
            
            # Color indicator and symbol based on hot/cold
            if pred_temp < 10.0:
                # Cold
                self.pred_icon_lbl.configure(text="❄", fg="#38bdf8")
            elif pred_temp >= 10.0 and pred_temp <= 25.0:
                # Moderate
                self.pred_icon_lbl.configure(text="☘", fg=SUCCESS_COLOR)
            else:
                # Hot
                self.pred_icon_lbl.configure(text="☀", fg=WARNING_COLOR)
                
        except Exception as e:
            self.update_status("Prediction Error", "error")
            messagebox.showerror("Prediction Error", f"Model could not predict value. Reason: {str(e)}")

    # --- VIEW: 8. GENERATE REPORT ---
    def generate_report_view(self):
        self.set_active_sidebar_button("Generate Report")
        if self.processor.df is None:
            self.show_no_data_message()
            return
            
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="Generate Analytical Weather Report", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        # Save and generate text
        report_text = self.processor.generate_report()
        self.update_status("Report Saved Successfully", "success")
        
        # Confirmation box
        conf_card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=15, pady=15)
        conf_card.pack(fill="x", pady=(0, 20))
        tk.Label(conf_card, text="✔ Report generated and saved to: output/weather_report.txt", font=("Segoe UI", 10, "bold"), fg=SUCCESS_COLOR, bg=CARD_BG, anchor="w").pack(fill="x")
        
        # Displays in scrollable read-only text field
        text_frame = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=10, pady=10)
        text_frame.pack(fill="both", expand=True)
        
        txt_widget = tk.Text(
            text_frame, 
            height=20, 
            bg=DARK_BG, 
            fg=TEXT_COLOR, 
            insertbackground=TEXT_COLOR, 
            font=("Consolas", 10), 
            bd=0, 
            padx=10, 
            pady=10
        )
        txt_widget.insert(tk.END, report_text)
        txt_widget.configure(state=tk.DISABLED)  # Read-only
        txt_widget.pack(fill="both", expand=True)

    # --- VIEW: 9. ABOUT PROJECT ---
    def about_project_view(self):
        self.set_active_sidebar_button("About Project")
        self.clear_content_pane()
        pane = self.scroll_frame.scrollable_frame
        
        title = tk.Label(pane, text="About This Project", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=DARK_BG, anchor="w")
        title.pack(fill="x", pady=(10, 20))
        
        card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=30, pady=30)
        card.pack(fill="both", expand=True)
        
        # Project header
        proj_name = tk.Label(card, text="Weather Data Analyzer & Temperature Prediction System", font=("Segoe UI", 14, "bold"), fg=ACCENT_COLOR, bg=CARD_BG, anchor="w")
        proj_name.pack(fill="x", pady=(0, 10))
        
        purpose_lbl = tk.Label(
            card, 
            text="Designed as a university-level desktop application demo for an Introduction to Open Source Software course. It implements historical weather dataset loading, comprehensive stats processing, visualization rendering, and predictive machine learning models.", 
            font=("Segoe UI", 10), 
            fg=TEXT_COLOR, 
            bg=CARD_BG, 
            justify="left", 
            wraplength=700,
            anchor="w"
        )
        purpose_lbl.pack(fill="x", pady=(0, 20))
        
        # Split features details
        grid_frame = tk.Frame(card, bg=CARD_BG)
        grid_frame.pack(fill="x", pady=(0, 20))
        
        left_col = tk.Frame(grid_frame, bg=CARD_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        right_col = tk.Frame(grid_frame, bg=CARD_BG)
        right_col.pack(side="left", fill="both", expand=True)
        
        # Technologies list (Left Column)
        tech_title = tk.Label(left_col, text="Core Technologies Used", font=("Segoe UI", 11, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        tech_title.pack(fill="x", pady=(0, 10))
        
        techs = [
            "Python (Programming Language Core)",
            "Tkinter (Desktop GUI Interface Library)",
            "Pandas (CSV Loading & Manipulation)",
            "NumPy (Vectorized Calculations)",
            "Matplotlib & Seaborn (Chart Rendering)",
            "Scikit-Learn (Linear Regression Engine)"
        ]
        
        for tech in techs:
            tk.Label(left_col, text=f"• {tech}", font=("Segoe UI", 9), fg=SUBTEXT_COLOR, bg=CARD_BG, anchor="w").pack(fill="x", pady=2)
            
        # Features list (Right Column)
        feats_title = tk.Label(right_col, text="App Functionalities", font=("Segoe UI", 11, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        feats_title.pack(fill="x", pady=(0, 10))
        
        features = [
            "Automatic Weather Attribute Column Mapping",
            "Missing Records Imputation & Cleaning",
            "Monthly and Condition Descriptive Stats",
            "7 Saved Static Exploratory Charts + 1 ML Validation Chart",
            "Dynamic Scikit-Learn Prediction Form Layout",
            "Comprehensive Exportable Text Summary Report",
            "MIT Open Source Academic License"
        ]
        
        for feat in features:
            tk.Label(right_col, text=f"• {feat}", font=("Segoe UI", 9), fg=SUBTEXT_COLOR, bg=CARD_BG, anchor="w").pack(fill="x", pady=2)
            
        # Divider line
        sep = tk.Frame(card, bg=BORDER_COLOR, height=1)
        sep.pack(fill="x", pady=15)
        
        # Licensing info
        lic_title = tk.Label(card, text="License & Developers", font=("Segoe UI", 11, "bold"), fg=TEXT_COLOR, bg=CARD_BG, anchor="w")
        lic_title.pack(fill="x", pady=(0, 5))
        
        lic_txt = tk.Label(
            card, 
            text="Licensed under the MIT Open Source License. Suitable for redistribution, modification, and academic integration.\nDeveloper: [Academic Project Presentation Placeholder]\nCourse: Introduction to Open Source Software (OSS)", 
            font=("Segoe UI", 9, "italic"), 
            fg=SUBTEXT_COLOR, 
            bg=CARD_BG, 
            justify="left",
            anchor="w"
        )
        lic_txt.pack(fill="x")

    # --- VIEW: 10. EXIT APPLICATION ---
    def exit_app(self):
        self.set_active_sidebar_button("Exit")
        confirm = messagebox.askyesno(
            "Exit Confirmation", 
            "Are you sure you want to exit the Weather Data Analyzer and Temperature Prediction System?"
        )
        if confirm:
            self.root.destroy()
        else:
            # Revert to About page visual
            self.set_active_sidebar_button("About Project")
            self.about_project_view()


if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()
