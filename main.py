"""
Weather Data Analyzer & Temperature Prediction System
Main entry point for the desktop application.
"""

import tkinter as tk
from gui_app import WeatherApp


if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()
