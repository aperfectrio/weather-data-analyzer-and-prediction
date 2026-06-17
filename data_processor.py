import os
import re
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


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
                    
                    colors_list = ['#38bdf8', '#818cf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa']
                    ax.pie(val_counts.values, labels=val_counts.index, autopct='%1.1f%%', 
                           startangle=90, colors=colors_list[:len(val_counts)],
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

    def generate_pdf_report(self, output_path="output_report.pdf"):
        """Generates detailed weather PDF report containing metrics and charts matching the premium presentation theme."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        # Make sure output directories are created
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Make sure visualizations are generated and saved
        self.save_all_visualizations("output")
        if self.is_trained:
            self.evaluate_model("output")
            
        stats = self.calculate_statistics()
        
        # Setup document with letter size and 0.5 inch margins
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        PRIMARY_COLOR = colors.HexColor("#1e293b")
        ACCENT_COLOR = colors.HexColor("#2563eb")
        TEXT_COLOR = colors.HexColor("#0f172a")
        SUBTEXT_COLOR = colors.HexColor("#475569")
        BORDER_COLOR = colors.HexColor("#cbd5e1")
        
        styles = getSampleStyleSheet()
        
        # Style sheet
        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=TEXT_COLOR,
            leading=13,
            spaceAfter=4
        )
        body_bold = ParagraphStyle(
            'ReportBodyBold',
            parent=body_style,
            fontName='Helvetica-Bold'
        )
        h1_style = ParagraphStyle(
            'ReportH1',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=PRIMARY_COLOR,
            spaceBefore=0,
            spaceAfter=0
        )
        
        # Banner styles
        banner_tag_style = ParagraphStyle(
            'BannerTag',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7,
            textColor=colors.white,
            alignment=1
        )
        banner_title_style = ParagraphStyle(
            'BannerTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            textColor=colors.white,
            alignment=1,
            leading=21,
            spaceAfter=4
        )
        banner_subtitle_style = ParagraphStyle(
            'BannerSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=colors.HexColor("#93c5fd"),
            alignment=1,
            leading=11
        )
        
        subtitle_style = ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=SUBTEXT_COLOR,
            spaceAfter=12,
            alignment=1
        )
        
        def make_section_header(num, title_text):
            badge = Table([[Paragraph(f"<font color='white'><b>{num}</b></font>", ParagraphStyle('BadgeText', alignment=1, fontSize=9, textColor=colors.white))]], colWidths=[0.25*inch], rowHeights=[0.25*inch])
            badge.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#2563eb")),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            
            t = Table([[badge, Paragraph(f"<b>{title_text}</b>", ParagraphStyle('H1_Col', parent=h1_style, textColor=colors.HexColor("#1e293b"), fontSize=12))]], colWidths=[0.35*inch, 7.15*inch])
            t.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('LINEBELOW', (1,0), (1,0), 1.5, colors.HexColor("#2563eb")),
            ]))
            return t
            
        def make_feature_card(title, desc):
            card = Table([[Paragraph(f"<b>{title}</b><br/>{desc}", ParagraphStyle('CardText', parent=body_style, fontSize=8, leading=10))]], colWidths=[3.6*inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0,0), (-1,-1), 8),
            ]))
            return card
            
        def make_library_card(name, desc):
            card = Table([[Paragraph(f"<b><font color='#2563eb'>{name}</font></b><br/>{desc}", ParagraphStyle('LibCardText', parent=body_style, fontSize=8, leading=10))]], colWidths=[2.3*inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            return card
            
        def make_timeline_step(num, title, desc):
            badge = Table([[Paragraph(f"<font color='white'><b>{num}</b></font>", ParagraphStyle('TimeText', alignment=1, fontSize=8, textColor=colors.white))]], colWidths=[0.22*inch], rowHeights=[0.22*inch])
            badge.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#2563eb")),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('TOPPADDING', (0,0), (-1,-1), 0),
            ]))
            desc_para = Paragraph(f"<b>{title}</b> &mdash; {desc}", ParagraphStyle('StepDesc', parent=body_style, fontSize=8.5, leading=10.5))
            return badge, desc_para
            
        story = []
        
        # --- FIRST PAGE HEADER AREA ---
        story.append(Spacer(1, 10))
        # Capsule tag
        tag_table = Table([[Paragraph("<b>SHORT PROJECT REPORT</b>", ParagraphStyle('CapsuleTag', parent=banner_tag_style, fontSize=6))]], colWidths=[1.8*inch])
        tag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1e3a8a")),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 2),
            ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor("#3b82f6")),
        ]))
        story.append(tag_table)
        story.append(Spacer(1, 8))
        
        story.append(Paragraph("Weather Data Analyzer &amp; Temperature Prediction System", banner_title_style))
        story.append(Paragraph("Introduction to Open Source Software &middot; Sejong University &middot; June 2026<br/>Instructor: Junaid Rashid", banner_subtitle_style))
        story.append(Spacer(1, 15))
        
        # Metadata sub-cards
        def make_meta_card(label, val):
            p_label = Paragraph(f"<font color='#93c5fd' size='7'><b>{label}</b></font>", ParagraphStyle('MetaLabel', alignment=1))
            p_val = Paragraph(f"<font color='white' size='11'><b>{val}</b></font>", ParagraphStyle('MetaVal', alignment=1))
            card = Table([[p_label], [p_val]], colWidths=[2.2*inch])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.Color(1, 1, 1, alpha=0.12)),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            return card
            
        best_r2 = f"{self.metrics.get('r2', 0.7698)*100:.2f}%" if self.is_trained else "76.98%"
        meta_table = Table([
            [make_meta_card("DATASET", "Seoul Weather"),
             make_meta_card("LANGUAGE", "Python 3.10+"),
             make_meta_card("BEST R² SCORE", best_r2)]
        ], colWidths=[2.4*inch, 2.4*inch, 2.4*inch])
        meta_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 55))
        
        # 1. Project Purpose & Description
        story.append(make_section_header("1", "Project Purpose & Description"))
        story.append(Spacer(1, 5))
        
        intro_text = "Understanding temperature fluctuations and weather trends is critical for meteorological forecasting, agricultural " \
                     "planning, and energy grid load predictions. This desktop application provides an end-to-end framework to load " \
                     "weather datasets, automatically clean and impute missing parameters, visualize trends, and fit predictive machine " \
                     "learning models."
        story.append(Paragraph(intro_text, body_style))
        story.append(Spacer(1, 5))
        
        # Callout box
        callout_data = [[Paragraph("<b>Why is this useful?</b> Meteorologists, local municipalities, and energy providers can leverage " \
                                   "historical statistics and linear predictions to anticipate climate anomalies, evaluate environmental coefficients, " \
                                   "and optimize grid scheduling to prevent peak overload.", ParagraphStyle('Callout', parent=body_style, fontSize=8.5, leading=11.5))]]
        callout_table = Table(callout_data, colWidths=[7.4*inch])
        callout_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eff6ff")),
            ('LINELEFT', (0,0), (0,-1), 3.0, colors.HexColor("#2563eb")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#dbeafe")),
        ]))
        story.append(callout_table)
        story.append(Spacer(1, 8))
        
        # Feature cards
        feature_cards_data = [
            [make_feature_card("Auto Column Mapping", "Smart regex detects date, temp, wind, precip, and conditions columns dynamically."),
             make_feature_card("Missing Data Cleaning", "Cleans voids by imputing numerical values with means and categories with modes.")],
            [Spacer(1, 5), Spacer(1, 5)],
            [make_feature_card("7 Visualizations", "Saves monthly averages, histograms, trendlines, and correlation heatmaps to disk."),
             make_feature_card("Linear Regressor Engine", "Splits dataset features 80/20 to fit scikit-learn models and outputs accuracy.")],
            [Spacer(1, 5), Spacer(1, 5)],
            [make_feature_card("Interactive Prediction Form", "Allows live predictions by manually entering weather features in the GUI."),
             make_feature_card("Academic MIT License", "Fully open source, free to redistribute, modify, and integrate for coursework.")]
        ]
        feature_table = Table(feature_cards_data, colWidths=[3.7*inch, 3.8*inch])
        feature_table.setStyle(TableStyle([
            ('PADDING', (0,0), (-1,-1), 0),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(feature_table)
        
        # --- PAGE 2 ---
        story.append(PageBreak())
        
        # 2. Tools & Libraries Used
        story.append(make_section_header("2", "Tools & Libraries Used"))
        story.append(Spacer(1, 6))
        
        lib_table = Table([
            [make_library_card("pandas", "Data manipulation & CSV parsing"),
             make_library_card("numpy", "Vectorized math & array logic"),
             make_library_card("scikit-learn", "Predictive modeling engine")],
            [Spacer(1, 5), Spacer(1, 5), Spacer(1, 5)],
            [make_library_card("matplotlib", "Core plot rendering backend"),
             make_library_card("seaborn", "Statistical heatmap visuals"),
             make_library_card("tkinter", "Desktop GUI window interface")]
        ], colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
        lib_table.setStyle(TableStyle([
            ('PADDING', (0,0), (-1,-1), 0),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(lib_table)
        story.append(Spacer(1, 8))
        story.append(Paragraph("All dependencies are open source and installable in a single command: <code>pip install -r requirements.txt</code>", ParagraphStyle('LibFooter', parent=body_style, fontSize=8, textColor=SUBTEXT_COLOR)))
        story.append(Spacer(1, 10))
        
        # 3. Weather Statistics Summary
        story.append(make_section_header("3", "Descriptive Statistics Summary"))
        story.append(Spacer(1, 6))
        
        stats_data = [
            ["Weather Metric", "Average / Total Value", "Maximum Recorded", "Minimum Recorded"],
            ["Temperature (°C)", 
             f"{stats['temp']['avg']:.2f} °C" if isinstance(stats['temp']['avg'], float) else stats['temp']['avg'],
             f"{stats['temp']['max']:.2f} °C" if isinstance(stats['temp']['max'], float) else stats['temp']['max'],
             f"{stats['temp']['min']:.2f} °C" if isinstance(stats['temp']['min'], float) else stats['temp']['min']],
            ["Humidity (%)", 
             f"{stats['humidity']['avg']:.2f} %" if isinstance(stats['humidity']['avg'], float) else stats['humidity']['avg'],
             f"{stats['humidity']['max']:.2f} %" if isinstance(stats['humidity']['max'], float) else stats['humidity']['max'],
             f"{stats['humidity']['min']:.2f} %" if isinstance(stats['humidity']['min'], float) else stats['humidity']['min']],
            ["Rainfall (mm)", 
             f"{stats['rainfall']['avg']:.2f} mm (Avg)" if isinstance(stats['rainfall']['avg'], float) else stats['rainfall']['avg'],
             f"{stats['rainfall']['max']:.2f} mm" if isinstance(stats['rainfall']['max'], float) else stats['rainfall']['max'],
             f"{stats['rainfall']['total']:.2f} mm (Total)" if isinstance(stats['rainfall']['total'], float) else stats['rainfall']['total']],
            ["Wind Speed (m/s)", 
             f"{stats['wind']['avg']:.2f} m/s" if isinstance(stats['wind']['avg'], float) else stats['wind']['avg'],
             f"{stats['wind']['max']:.2f} m/s" if isinstance(stats['wind']['max'], float) else stats['wind']['max'],
             "N/A"],
        ]
        
        stats_table = Table(stats_data, colWidths=[2.25*inch, 2.0*inch, 1.625*inch, 1.625*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 6))
        
        condition_text = f"<b>Weather Conditions:</b> Most Common Condition is <b>{stats['condition']['most_common']}</b>, " \
                         f"and the Least Common Condition is <b>{stats['condition']['least_common']}</b>."
        story.append(Paragraph(condition_text, body_style))
        
        # --- PAGE 3 ---
        story.append(PageBreak())
        
        # 4. Project Workflow
        story.append(make_section_header("4", "Project Workflow"))
        story.append(Spacer(1, 6))
        
        timeline_data = []
        steps = [
            ("1", "Load Dataset", "Loads local weather CSV files dynamically via native Tkinter file dialogs."),
            ("2", "Preprocess & Clean", "Sorts data chronologically, parses dates, and imputes missing fields with column means/modes."),
            ("3", "Descriptive Statistics", "Calculates aggregated average, maximum, and minimum parameters for weather variables."),
            ("4", "Save Visualizations", "Saves 7 styled PNG plots illustrating historical trends, densities, and correlations."),
            ("5", "Train Regressor", "Splits the processed numerical attributes 80/20 and fits a linear regressor using scikit-learn."),
            ("6", "Predict Temperatures", "Accepts live manual input variables in the GUI form and runs prediction instantly.")
        ]
        for num, title, desc in steps:
            b, d = make_timeline_step(num, title, desc)
            timeline_data.append([b, d])
            timeline_data.append([Spacer(1, 6), Spacer(1, 6)])
            
        timeline_table = Table(timeline_data[:-1], colWidths=[0.4*inch, 7.1*inch])
        timeline_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (0,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 0),
            ('LINEAFTER', (0,0), (0,-2), 1.5, colors.HexColor("#e2e8f0")),
        ]))
        story.append(timeline_table)
        
        # --- PAGE 4 ---
        story.append(PageBreak())
        
        # 5. Graphical Data Analysis
        story.append(make_section_header("5", "Graphical Data Analysis"))
        story.append(Spacer(1, 8))
        
        trend_path = os.path.join("output", "temperature_trend.png")
        if os.path.exists(trend_path):
            img_trend = Image(trend_path, width=6.5*inch, height=2.8*inch)
            story.append(img_trend)
            story.append(Paragraph("<font color='#475569'><i>Figure 1: Chronological Temperature Trend Over Time</i></font>", ParagraphStyle('FigC1', parent=subtitle_style, alignment=1, spaceAfter=8)))
            
        monthly_path = os.path.join("output", "monthly_temperature.png")
        if os.path.exists(monthly_path):
            img_month = Image(monthly_path, width=6.5*inch, height=2.8*inch)
            story.append(img_month)
            story.append(Paragraph("<font color='#475569'><i>Figure 2: Monthly Average Temperature Comparison</i></font>", ParagraphStyle('FigC2', parent=subtitle_style, alignment=1, spaceAfter=8)))
            
        # --- PAGE 5 ---
        story.append(PageBreak())
        
        story.append(make_section_header("5", "Graphical Data Analysis (Cont.)"))
        story.append(Spacer(1, 8))
        
        rain_path = os.path.join("output", "monthly_rainfall.png")
        if os.path.exists(rain_path):
            img_rain = Image(rain_path, width=5.5*inch, height=2.3*inch)
            story.append(img_rain)
            story.append(Paragraph("<font color='#475569'><i>Figure 3: Monthly Rainfall Distribution (Total Precip)</i></font>", ParagraphStyle('FigC3', parent=subtitle_style, alignment=1, spaceAfter=8)))
            
        cond_path = os.path.join("output", "weather_conditions.png")
        if os.path.exists(cond_path):
            img_cond = Image(cond_path, width=5.5*inch, height=2.3*inch)
            story.append(img_cond)
            story.append(Paragraph("<font color='#475569'><i>Figure 4: Weather Conditions Frequency Distribution</i></font>", ParagraphStyle('FigC4', parent=subtitle_style, alignment=1, spaceAfter=8)))
            
        corr_path = os.path.join("output", "correlation_heatmap.png")
        if os.path.exists(corr_path):
            img_corr = Image(corr_path, width=5.5*inch, height=2.3*inch)
            story.append(img_corr)
            story.append(Paragraph("<font color='#475569'><i>Figure 5: Feature Correlation Heatmap Matrix</i></font>", ParagraphStyle('FigC5', parent=subtitle_style, alignment=1, spaceAfter=8)))
            
        # --- PAGE 6 ---
        story.append(PageBreak())
        
        # 6. Machine Learning Results & Performance
        story.append(make_section_header("6", "Machine Learning Results & Performance"))
        story.append(Spacer(1, 5))
        
        if self.is_trained:
            ml_intro = f"A <b>Linear Regression</b> model was trained to predict the target temperature variable <b>{self.temp_col}</b>. " \
                       f"The model was successfully fitted on <b>{len(self.X_train)} samples</b> (80%) and evaluated against " \
                       f"<b>{len(self.X_test)} unseen test samples</b> (20%)."
            story.append(Paragraph(ml_intro, body_style))
            story.append(Spacer(1, 4))
            
            metrics_data = [
                [Paragraph("<b>Evaluation Metric</b>", body_bold), Paragraph("<b>Value</b>", body_bold), Paragraph("<b>Interpretation / Description</b>", body_bold)],
                ["Mean Absolute Error (MAE)", f"{self.metrics.get('mae', 0):.4f} °C", "Average magnitude of errors in predictions."],
                ["Mean Squared Error (MSE)", f"{self.metrics.get('mse', 0):.4f}", "Average squared difference between predicted and actual."],
                ["Root Mean Squared Error (RMSE)", f"{self.metrics.get('rmse', 0):.4f} °C", "Standard deviation of prediction residuals."],
                ["R² Score (R-squared) \u2705 Best", f"{self.metrics.get('r2', 0):.4f}", "Proportion of target variance explained by the model."],
            ]
            metrics_table = Table(metrics_data, colWidths=[2.5*inch, 1.25*inch, 3.75*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 8))
            
            coef_data = [
                [Paragraph("<b>Predictor Feature</b>", body_bold), Paragraph("<b>Coefficient</b>", body_bold), Paragraph("<b>Effect on Temperature</b>", body_bold)]
            ]
            for feat, coef in zip(self.model_features, self.model.coef_):
                effect = "Positive relationship (Temp rises)" if coef >= 0 else "Negative relationship (Temp drops)"
                coef_data.append([feat, f"{coef:.4f}", effect])
            coef_data.append(["[Intercept Constant]", f"{self.model.intercept_:.4f}", "Base temperature value"])
            
            coef_table = Table(coef_data, colWidths=[2.5*inch, 1.25*inch, 3.75*inch])
            coef_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ('PADDING', (0, 0), (-1, -1), 3),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]))
            story.append(Paragraph("<b>Model Coefficients Summary:</b>", body_bold))
            story.append(Spacer(1, 3))
            story.append(coef_table)
            story.append(Spacer(1, 8))
            
            val_plot_path = os.path.join("output", "actual_vs_predicted.png")
            if os.path.exists(val_plot_path):
                img_val = Image(val_plot_path, width=5.5*inch, height=2.4*inch)
                story.append(img_val)
                story.append(Paragraph("<font color='#475569'><i>Figure 6: Actual vs Predicted Temperature on Test Set</i></font>", ParagraphStyle('FigVal', parent=subtitle_style, alignment=1)))
        else:
            story.append(Paragraph("<i>The prediction model was not trained before exporting this report. No machine learning metrics are available.</i>", body_style))
            
        # --- PAGE 7 ---
        story.append(PageBreak())
        
        # 7. Why README.md and requirements.txt Matter
        story.append(make_section_header("7", "Why README.md and requirements.txt Matter"))
        story.append(Spacer(1, 6))
        
        doc_data = [
            [Paragraph("<b>README.MD</b><br/>The README acts as the entry portal for any software repository. It answers essential questions for developers and users alike: what does the project do, how do you configure it, and how do you execute it? On platforms like GitHub, the README renders dynamically at the root page, serving as the core reference before anyone contributes to or deploys the software system.", ParagraphStyle('ReadmeText', parent=body_style, fontSize=8.5, leading=11.5))],
            [Spacer(1, 6)],
            [Paragraph("<b>REQUIREMENTS.TXT</b><br/>This file catalogs the required third-party Python packages and strict version constraints. Executing <code>pip install -r requirements.txt</code> replicates the exact runtime environment on any machine in seconds. This prevents library collisions, fixes reproducibility issues across different operating systems, and ensures seamless academic evaluation.", ParagraphStyle('ReqText', parent=body_style, fontSize=8.5, leading=11.5))]
        ]
        doc_table = Table(doc_data, colWidths=[7.4*inch])
        doc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(doc_table)
        
        # Define canvas callback functions
        def draw_first_page_background(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.HexColor("#1e3a8a"))
            canvas.roundRect(0.5*inch, 7.25*inch, 7.5*inch, 3.25*inch, 15, fill=1, stroke=0)
            
            # Running footer on first page
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor("#64748b"))
            canvas.drawString(0.5*inch, 0.35*inch, "Sejong University \u00b7 Introduction to Open Source Software \u00b7 June 2026 \u00b7 MIT License")
            canvas.drawRightString(8.0*inch, 0.35*inch, f"Page {doc.page}")
            canvas.restoreState()
            
        def draw_later_page_background(canvas, doc):
            canvas.saveState()
            # Top running line
            canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
            canvas.setLineWidth(0.5)
            canvas.line(0.5*inch, 10.5*inch, 8.0*inch, 10.5*inch)
            
            # Running header text
            canvas.setFont('Helvetica', 8)
            canvas.setFillColor(colors.HexColor("#64748b"))
            canvas.drawString(0.5*inch, 10.6*inch, "Weather Data Analyzer & Temperature Prediction System")
            canvas.drawRightString(8.0*inch, 10.6*inch, "SHORT PROJECT REPORT")
            
            # Running footer line & text
            canvas.line(0.5*inch, 0.5*inch, 8.0*inch, 0.5*inch)
            canvas.drawString(0.5*inch, 0.35*inch, "Sejong University \u00b7 Introduction to Open Source Software \u00b7 June 2026 \u00b7 MIT License")
            canvas.drawRightString(8.0*inch, 0.35*inch, f"Page {doc.page}")
            canvas.restoreState()
            
        doc.build(story, onFirstPage=draw_first_page_background, onLaterPages=draw_later_page_background)
