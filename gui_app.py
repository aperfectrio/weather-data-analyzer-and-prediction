import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import *
from data_processor import WeatherDataProcessor
from gui_widgets import ScrollableFrame


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
        
        file_path = filedialog.askopenfilename(
            title="Open Weather CSV Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
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
        
        # Save and generate text and PDF reports
        report_text = self.processor.generate_report()
        pdf_success = True
        try:
            self.processor.generate_pdf_report("output_report.pdf")
        except Exception as e:
            pdf_success = False
            print(f"Error generating PDF report: {e}")
            
        self.update_status("Reports Saved Successfully", "success")
        
        # Confirmation box
        conf_card = tk.Frame(pane, bg=CARD_BG, highlightbackground=BORDER_COLOR, highlightthickness=1, padx=15, pady=15)
        conf_card.pack(fill="x", pady=(0, 20))
        
        txt_msg = "✔ Text report generated and saved to: output/weather_report.txt"
        tk.Label(conf_card, text=txt_msg, font=("Segoe UI", 10, "bold"), fg=SUCCESS_COLOR, bg=CARD_BG, anchor="w").pack(fill="x", pady=(0, 5))
        
        if pdf_success:
            pdf_msg = "✔ PDF report generated and saved to: output_report.pdf"
            pdf_color = SUCCESS_COLOR
        else:
            pdf_msg = "⚠ Failed to generate PDF report (check application console)"
            pdf_color = WARNING_COLOR
        tk.Label(conf_card, text=pdf_msg, font=("Segoe UI", 10, "bold"), fg=pdf_color, bg=CARD_BG, anchor="w").pack(fill="x")
        
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
