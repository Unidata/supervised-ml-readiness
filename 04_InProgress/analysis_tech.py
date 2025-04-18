# Data manipulation and analysis
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Callable

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Machine learning
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Utilities
import time
from typing import Any
from sklearn.base import BaseEstimator
from sklearn.metrics import root_mean_squared_error, r2_score

STATIONS = ['BEAR', 'BURN', 'FRYI', 'JEFF', 'NCAT', 'SALI', 'SASS', 'UNCA', 'WINE']

def create_ml_knowledgecheck():
    """
    Creates an interactive knowledge check about ML concepts with two buttons and feedback.
    Returns the display elements for use in a Jupyter notebook.
    """
    output = widgets.Output()

    question = widgets.HTML(
        "Which type of machine learning analysis is most appropriate for this scenario?"
    )
    
    # Create buttons
    classification_button = widgets.Button(
        description='Classification',
        layout=widgets.Layout(width='200px', height='200px', margin='10px')
    )
    
    regression_button = widgets.Button(
        description='Regression',
        layout=widgets.Layout(width='200px', height='200px', margin='10px')
    )
    
    def show_feedback(is_correct):
        """Helper function to display feedback"""
        with output:
            clear_output(wait=True)
            if is_correct:
                display(HTML("""
                    <div class="alert alert-info" role="feedback">
                        <p class="admonition-title" style="font-weight:bold">Correct</p>
                        <p>This scenario requires a numerical output, so we will use a regression algorithm for this scenario.</p>
                    </div>
                """))
            else:
                display(HTML("""
                    <div class="alert alert-info" role="feedback">
                        <p class="admonition-title" style="font-weight:bold">Incorrect</p>
                        <p>Classification tasks work for scenarios that require classifying data into categories. 
                        This task needs a <i>numerical value </i>for output, and therefore requires a different approach.</p>
                    </div>
                """))
    
    # Define click handlers
    classification_button.on_click(lambda b: show_feedback(False))
    regression_button.on_click(lambda b: show_feedback(True))
    
    # Create button container
    buttons = widgets.HBox([classification_button, regression_button])
    
    return question, buttons, output

def display_knowledgecheck():
    """Creates and displays the knowledge check in the notebook."""
    question, buttons, output = create_ml_knowledgecheck()
    display(question, buttons, output)


def display_golden_wier_dashboard(hydrograph_data):
    """Creates and displays interactive dashboard for Golden Wier hydrograph data."""
    # Create interface controls
    plot_dropdown = widgets.Dropdown(
        options=['Histogram', 'Annual Time Series', 'Yearly Comparison'],
        value='Histogram',
        description='Plot Type:',
        disabled=False,
    )
    
    plot_button = widgets.Button(
        description='Generate Plot',
        button_style='primary',
        tooltip='Click to generate the plot',
    )
    
    output = widgets.Output()
    
    def on_plot_button_click(b):
        with output:
            clear_output(wait=True)
            
            if plot_dropdown.value == 'Histogram':
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.hist(hydrograph_data['daily_mean_discharge_DT_GOLDEN'].dropna(), bins=30, 
                       color='skyblue', edgecolor='black')
                ax.set_title("Histogram of Daily Mean Discharge at Golden Wier", fontsize=14)
                ax.set_xlabel("Daily Mean Discharge")
                ax.set_ylabel("Number of records")
                plt.show()
            
            elif plot_dropdown.value == 'Annual Time Series':
                # Group by year and day of year for annual comparison
                years = sorted(hydrograph_data['year'].unique())
                fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
                
                for year in years:
                    year_data = hydrograph_data[hydrograph_data['year'] == year]
                    year_data = year_data.sort_values('day_of_year')
                    ax.plot(year_data['day_of_year'], year_data['daily_mean_discharge_DT_GOLDEN'], 
                           label=str(year))
                
                ax.set_title("Annual Comparison of Daily Mean Discharge at Golden Wier", fontsize=14)
                ax.set_xlabel("Day of Year")
                ax.set_ylabel("Daily Mean Discharge")
                ax.legend(title="Year")
                plt.show()
                
            elif plot_dropdown.value == 'Yearly Comparison':
                # Calculate yearly averages
                yearly_avg = hydrograph_data.groupby('year')['daily_mean_discharge_DT_GOLDEN'].mean().reset_index()
                
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.bar(yearly_avg['year'].astype(str), yearly_avg['daily_mean_discharge_DT_GOLDEN'], 
                      color='green', edgecolor='black')
                ax.set_title("Yearly Average Discharge at Golden Wier", fontsize=14)
                ax.set_xlabel("Year")
                ax.set_ylabel("Average Daily Mean Discharge")
                plt.xticks(rotation=45)
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Golden Wier Hydrograph Dashboard</h3>"), 
           plot_dropdown, 
           plot_button, 
           output)

def display_input_stations_dashboard(weather_data):
    """Creates and displays interactive dashboard for input stations weather data."""
    # Create interface controls
    station_dropdown, var_dropdown, plot_dropdown, plot_button, output = create_input_station_controls()
    
    def on_plot_button_click(b):
        # Construct variable name by combining station and weather variable
        selected_var = f"{station_dropdown.value}_{var_dropdown.value}"
        
        with output:
            clear_output(wait=True)
            
            if plot_dropdown.value == 'Histogram':
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.hist(weather_data[selected_var], bins=30, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {var_dropdown.label} at {station_dropdown.value}", fontsize=14)
                ax.set_xlabel(var_dropdown.label)
                ax.set_ylabel("Number of records")
                plt.show()
            
            elif plot_dropdown.value == 'Time Series':
                xdates = pd.to_datetime(weather_data['observation_datetime'])
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.plot(xdates[::100], weather_data[selected_var][::100], 
                       label=var_dropdown.label, color='orange')
                ax.set_title(f"Time Series of {var_dropdown.label} at {station_dropdown.value}", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel(var_dropdown.label)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Input Stations</h3>"), 
           station_dropdown,
           var_dropdown, 
           plot_dropdown, 
           plot_button, 
           output)

def create_correlation_plot_controls():
    """Creates control widgets for correlation plots."""
    # Add title label with larger size and bold styling
    title = widgets.HTML(value='<h3 style="font-weight: bold; margin: 0; padding: 0;">Comparison Plot</h3>')
    
    var_dropdown = widgets.Dropdown(
        options=[
            ('Temperature (F)', 'airtemp_degF'),
            ('Precipitation (in)', 'precip_in'),
            ('Relative Humidity (%)', 'rh_percent'),
            ('Wind Gust (mph)', 'windgust_mph'),
            ('Average Wind Speed (mph)', 'windspeed_mph')
        ],
        description='Variable:',
        disabled=False
    )

    plot_button = widgets.Button(description="Plot")
    output = widgets.Output()
    
    return title, var_dropdown, plot_button, output


def display_correlation_plot_dashboard(base_url="https://elearning.unidata.ucar.edu/dataeLearning/Cybertraining/analysis/media/pairplot_"):
    """Creates and displays interactive dashboard for correlation plots."""
    # Create interface controls
    title, var_dropdown, plot_button, output = create_correlation_plot_controls()
    
    def update_image(_):
        selected_var = var_dropdown.value
        image_url = f"{base_url}{selected_var}.png"
        
        with output:
            output.clear_output(wait=True)
            display(HTML(
                f'<center><i>Click to enlarge</i><br>'
                f'<a href="{image_url}" target="blank">'
                f'<img src="{image_url}" width="600px"></a></center>'
            ))
    
    plot_button.on_click(update_image)
    display(title, var_dropdown, plot_button, output)


def algorithm_selection():
    """Creates widget for algorithm selection and returns selected value via callback."""
    algorithm_options = {
        "Multi-Linear Regressor": "linear_regression",
        "XGBoost": "xgboost"
    }
    
    # Use a list to store the selection (mutable)
    selection = [None]
    output = widgets.Output()

    buttons = [
        widgets.Button(
            description=name,
            layout=widgets.Layout(width='200px', height='200px', margin='10px')
        )
        for name in algorithm_options
    ]

    def on_button_clicked(b):
        selection[0] = algorithm_options[b.description]
        for button in buttons:
            button.style.button_color = '#b2ebf2' if button == b else None
        with output:
            clear_output(wait=True)
            print(f"Selected Algorithm: {b.description}")

    for button in buttons:
        button.on_click(on_button_clicked)

    display(widgets.HBox(buttons), output)
    
    # Function to get current selection
    def get_selection():
        return selection[0]
        
    return get_selection

# Extract station names from your dataframe columns
STATIONS = [
    'hourly_precip_mean_inch_0CO',
    'hourly_precip_sum_inch_0CO',
    'daily_mean_discharge_LEAV_GTOWN',
    'daily_mean_discharge_WF_EMPIRE',
    'daily_mean_discharge_MAIN_LAWSN',
    'daily_mean_discharge_N_BLKHAWK',
    'daily_mean_discharge_DT_GOLDEN'
]

def create_station_selector():
    """Creates grid of checkboxes for station selection."""
    checkboxes = {
        station: widgets.Checkbox(
            value=False,
            description=station,
            disabled=False,
            indent=False
        ) 
        for station in STATIONS
    }
    
    checkbox_grid = widgets.GridBox(
        children=[checkboxes[station] for station in STATIONS],
        layout=widgets.Layout(
            grid_template_columns='repeat(2, minmax(350px, 1fr))',
            grid_gap='10px',
            width='100%',
            padding='2px',
            overflow='hidden'
        )
    )
    
    output = widgets.Output()
    
    def on_change(change):
        with output:
            output.clear_output()
            selected = [station for station, checkbox in checkboxes.items() if checkbox.value]
            print(f"Selected stations: {', '.join(selected) if selected else 'None'}")
    
    for checkbox in checkboxes.items():
        checkbox[1].observe(on_change, names='value')
    
    display(widgets.VBox([
        widgets.HTML(value="<h3>Select Weather and Stream Gauge Stations</h3>"),
        checkbox_grid,
        output
    ]))
    
    return checkboxes



selected_model = None  # Global variable for model access

def train_button(selected_algo, X_train_filtered, y_train):
    """Creates a single 'Train ML Model' button, using the provided selected_algo."""
    global selected_model
    selected_model = None  # Reset the model at start
    output = widgets.Output()

    # Create a label to show model status
    status_label = widgets.Label(value='Click button to train model')
    
    train_button = widgets.Button(
        description='Train Algorithm', 
        layout=widgets.Layout(width='200px', height='200px')
    )

    def train_model(b):
        global selected_model
        with output:
            clear_output()
            try:
                if selected_algo == "xgboost":
                    print("Running XGBoost model...")
                    base_model = XGBRegressor(
                        n_estimators=100,
                        tree_method='hist',
                        random_state=42
                    )
                    selected_model = MultiXGBRegressor(base_model)
                    selected_model.fit(X_train_filtered, y_train)
                    print("XGBoost model training completed!")
                    print(f"Model object created and trained: {selected_model is not None}")
                    status_label.value = 'Model trained successfully!'
                    
                elif selected_algo == "linear_regression":
                    print("Running Linear Regression model...")
                    selected_model = MultiLinearRegressor()
                    selected_model.fit(X_train_filtered, y_train)
                    print("Linear Regression training completed!")
                    print(f"Model object created and trained: {selected_model is not None}")
                    status_label.value = 'Model trained successfully!'
                    
                else:
                    print("No algorithm selected. Cannot train.")
                    selected_model = None
                    status_label.value = 'No algorithm selected'
                    
            except Exception as e:
                print(f"Error during training: {str(e)}")
                selected_model = None
                status_label.value = 'Error during training'
            
            # Print final status of selected_model
            print("\nFinal status:")
            print(f"selected_model object exists: {selected_model is not None}")
            if selected_model is not None:
                print(f"Model type: {type(selected_model)}")
                
            # Update button state
            train_button.description = 'Model Trained'
            train_button.disabled = True

    train_button.on_click(train_model)

    display(widgets.VBox([train_button, status_label]))
    display(output)

    def get_model():
        """Function to get the trained model"""
        return selected_model

    return get_model  # Return the function instead of the model directly

## Data

class MultiXGBRegressor(MultiOutputRegressor):
    def __init__(self, estimator):
        super().__init__(estimator)
        self.estimators_ = []

    def fit(self, X, y):
        start_time = time.time()
        print("\nStarting Multi-Target XGBoost Training Process...")
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        n_outputs = y_np.shape[1]
        target_names = y.columns if hasattr(y, 'columns') else [f"target_{i}" for i in range(n_outputs)]
        
        self.estimators_ = [
            XGBRegressor(**{k: v for k, v in self.estimator.get_params().items() 
                          if k != 'verbose'}) 
            for _ in range(n_outputs)
        ]
        
        for i, (est, target) in enumerate(zip(self.estimators_, target_names)):
            target_start = time.time()
            print(f"\nTraining target {i+1}/{n_outputs}: {target}", flush=True)
            est.fit(X, y_np[:, i], verbose=False)
            target_time = time.time() - target_start
            print(f"Target completed in {target_time:.2f} seconds", flush=True)

        total_time = time.time() - start_time
        print(f"\nTotal training completed in {total_time:.2f} seconds")
        return self

class MultiLinearRegressor(MultiOutputRegressor):
    def __init__(self):
        super().__init__(LinearRegression())
        self.estimators_ = []

    def fit(self, X, y):
        start_time = time.time()
        print("\nStarting Multi-Target Linear Regression Training...")
        y_np = y.values if hasattr(y, 'values') else np.array(y)
        n_outputs = y_np.shape[1]
        target_names = y.columns if hasattr(y, 'columns') else [f"target_{i}" for i in range(n_outputs)]
        
        self.estimators_ = [LinearRegression() for _ in range(n_outputs)]
        
        for i, (est, target) in enumerate(zip(self.estimators_, target_names)):
            target_start = time.time()
            print(f"\nTraining target {i+1}/{n_outputs}: {target}", flush=True)
            est.fit(X, y_np[:, i])
            target_time = time.time() - target_start
            print(f"Target completed in {target_time:.2f} seconds", flush=True)

        total_time = time.time() - start_time
        print(f"\nTotal training completed in {total_time:.2f} seconds")
        return self

def split_data_temporal(df, train_years, val_years, test_years, target_column, year_column='year'):
    """
    Split data temporally based on specified years for time series forecasting.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing your data with a year column
    train_years : list
        List of years to use for training (e.g., [2015, 2016, 2017, 2018, 2019])
    val_years : list
        List of years to use for validation (e.g., [2020, 2021])
    test_years : list
        List of years to use for testing
    target_column : str
        The name of the target variable column
    year_column : str, default 'year'
        The name of the column containing the year information
        
    Returns:
    --------
    X_train, y_train, X_val, y_val, X_test, y_test, X_true_test, y_true_test
    """
    # Verify that the years exist in the dataset
    available_years = set(df[year_column].unique())
    for year in train_years + val_years + test_years:
        if year not in available_years:
            print(f"Warning: Year {year} not found in dataset")
    
    # Create masks for each set
    train_mask = df[year_column].isin(train_years)
    val_mask = df[year_column].isin(val_years)
    
    # Split test years in half for regular test and true test
    if len(test_years) > 1:
        split_idx = len(test_years) // 2
        regular_test_years = test_years[:split_idx]
        true_test_years = test_years[split_idx:]
    else:
        regular_test_years = test_years
        true_test_years = test_years  # Same as regular test in this case
    
    test_mask = df[year_column].isin(regular_test_years)
    true_test_mask = df[year_column].isin(true_test_years)
    
    # Get feature columns (all columns except target)
    X_columns = [col for col in df.columns if col != target_column]
    
    # Create the splits
    X_train = df.loc[train_mask, X_columns]
    y_train = df.loc[train_mask, target_column]
    
    X_val = df.loc[val_mask, X_columns]
    y_val = df.loc[val_mask, target_column]
    
    X_test = df.loc[test_mask, X_columns]
    y_test = df.loc[test_mask, target_column]
    
    X_true_test = df.loc[true_test_mask, X_columns]
    y_true_test = df.loc[true_test_mask, target_column]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_true_test, y_true_test

def filter_dataframe(df, prefix_values):
    """
    Filter DataFrame to keep only columns with specified prefixes plus day_index and hour_index.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    prefix_values (list): List of prefix values to match
    
    Returns:
    pandas.DataFrame: Filtered DataFrame with only the specified columns
    """
    # Print original column count
    print(f"Original DataFrame: {len(df.columns)} columns")
    
    # Start with day_index and hour_index
    columns_to_keep = ['day_index', 'hour_index']
    
    # Add any column that starts with our prefix values
    for prefix in prefix_values:
        matching_columns = [col for col in df.columns if col.startswith(prefix)]
        columns_to_keep.extend(matching_columns)
    
    # Create filtered dataframe
    filtered_df = df[columns_to_keep]
    
    # Print new column count
    print(f"Filtered DataFrame: {len(filtered_df.columns)} columns")
    
    return filtered_df

def model_eval_MITC(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    eval_type: str = 'Validation'
) -> None:
    """
    Evaluates a trained model using test data and prints performance metrics for MITC.
    """
    if eval_type not in ['Testing', 'Validation', None]:
        raise ValueError(f"eval_type must be one of ['Testing', 'Validation', None]")
    
    # Define header first
    header = f"{eval_type} Metrics" if eval_type else "Metrics"

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    
    # Get stations from X_test that are in STATIONS list
    used_stations = sorted(set(col.split('_')[0] for col in X_test.columns 
                             if col.split('_')[0] in STATIONS))
    
    print(header)
    print(f"\nModel Type: {type(model).__name__}")
    print(f"\nStations used ({len(used_stations)}/{len(STATIONS)}):")
    print(', '.join(used_stations))
    
    print("\nRMSE for each target feature:")
    for target, error in zip(y_test.columns, rmse):
        print(f" {target}:\t{error:.4f}")
    
    print("\nR² Score for each target feature:")
    for target, score in zip(y_test.columns, r2):
        print(f" {target}:\t{score:.4f}")
    
    print(f"\nAverage R² Score:\t{np.mean(r2):.2f}")

def display_discharge_dashboard(hydrograph_data):
    """Creates and displays interactive dashboard for multiple hydrograph stations."""
    
    # Define the discharge columns (excluding Golden Wier)
    discharge_columns = [col for col in hydrograph_data.columns 
                         if 'daily_mean_discharge' in col and 'GOLDEN' not in col]
    
    # Create more descriptive location names
    location_names = {
        'daily_mean_discharge_LEAV_GTOWN': 'Georgetown',
        'daily_mean_discharge_WF_EMPIRE': 'West Fork at Empire',
        'daily_mean_discharge_MAIN_LAWSN': 'Lawson',
        'daily_mean_discharge_N_BLKHAWK': 'North Clear Creek at Blackhawk'
    }
    
    # Create interface controls
    location_dropdown = widgets.Dropdown(
        options=[(location_names.get(col, col), col) for col in discharge_columns],
        description='Location:',
        disabled=False,
    )
    
    # Set default value after creating options
    if len(discharge_columns) > 0:
        location_dropdown.value = discharge_columns[0]
    
    plot_dropdown = widgets.Dropdown(
        options=['Histogram', 'Annual Time Series', 'Yearly Comparison'],
        value='Histogram',
        description='Plot Type:',
        disabled=False,
    )
    
    plot_button = widgets.Button(
        description='Generate Plot',
        button_style='primary',
        tooltip='Click to generate the plot',
    )
    
    output = widgets.Output()
    
    def on_plot_button_click(b):
        with output:
            clear_output(wait=True)
            
            if not discharge_columns:
                print("No discharge columns found in the dataset (excluding Golden).")
                return
                
            selected_column = location_dropdown.value
            # Get the location name directly from the dictionary
            location_name = location_names.get(selected_column, selected_column)
            
            if plot_dropdown.value == 'Histogram':
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                data_to_plot = hydrograph_data[selected_column].dropna()
                if len(data_to_plot) == 0:
                    print(f"No data available for {location_name}")
                    return
                ax.hist(data_to_plot, bins=30, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of Daily Mean Discharge at {location_name}", fontsize=14)
                ax.set_xlabel("Daily Mean Discharge")
                ax.set_ylabel("Number of records")
                plt.show()
            
            elif plot_dropdown.value == 'Annual Time Series':
                # Ensure required columns exist
                if 'year' not in hydrograph_data.columns or 'day_of_year' not in hydrograph_data.columns:
                    print("Error: Data missing 'year' or 'day_of_year' columns needed for this plot")
                    return
                
                # Group by year and day of year for annual comparison
                years = sorted(hydrograph_data['year'].unique())
                if len(years) == 0:
                    print("No year data available for plotting")
                    return
                    
                fig, ax = plt.subplots(1, 1, figsize=(10, 6), tight_layout=True)
                
                for year in years:
                    year_data = hydrograph_data[hydrograph_data['year'] == year]
                    if len(year_data) == 0:
                        continue
                    year_data = year_data.sort_values('day_of_year')
                    # Only plot if we have data
                    if not year_data[selected_column].isna().all():
                        ax.plot(year_data['day_of_year'], year_data[selected_column], 
                               label=str(year))
                
                ax.set_title(f"Annual Comparison of Daily Mean Discharge at {location_name}", fontsize=14)
                ax.set_xlabel("Day of Year")
                ax.set_ylabel("Daily Mean Discharge")
                ax.legend(title="Year")
                plt.show()
                
            elif plot_dropdown.value == 'Yearly Comparison':
                # Ensure required column exists
                if 'year' not in hydrograph_data.columns:
                    print("Error: Data missing 'year' column needed for this plot")
                    return
                
                # Calculate yearly averages
                yearly_avg = hydrograph_data.groupby('year')[selected_column].mean().reset_index()
                
                if len(yearly_avg) == 0:
                    print("No data available for yearly comparison")
                    return
                    
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.bar(yearly_avg['year'].astype(str), yearly_avg[selected_column], 
                      color='green', edgecolor='black')
                ax.set_title(f"Yearly Average Discharge at {location_name}", fontsize=14)
                ax.set_xlabel("Year")
                ax.set_ylabel("Average Daily Mean Discharge")
                plt.xticks(rotation=45)
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Hydrograph Stations Dashboard</h3>"), 
           location_dropdown,
           plot_dropdown, 
           plot_button, 
           output)


def year_selection_widget(auto_display=True):
    """Creates widget for specifying training/validation/testing splits allowing multiple years per category.
    
    Returns:
        widget_box: The widget interface
        get_selection: A function that returns the current selection as integers
    """
    # Predefined years
    years = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    
    # Convert years to list of strings for the checkboxes
    year_options = [str(year) for year in years]
    
    # Create section headers with larger font
    header_style = "font-size: 16px; font-weight: bold; margin-bottom: 5px;"
    training_header = widgets.HTML(value=f"<span style='{header_style}'>Training Years:</span>")
    validation_header = widgets.HTML(value=f"<span style='{header_style}'>Validation Years:</span>")
    testing_header = widgets.HTML(value=f"<span style='{header_style}'>Testing Years:</span>")
    
    # Custom CSS for checkbox labels to increase font size and reduce spacing
    checkbox_style = """
    <style>
    .widget-checkbox {
        margin-right: -10px !important;  /* Less aggressive spacing reduction */
        width: auto !important;
    }
    .widget-checkbox > label {
        font-size: 14px !important;  /* Increase font size by ~1 from default */
        padding-right: 15px !important; /* More padding to the right for better readability */
    }
    /* Add some padding to the checkbox containers to prevent clipping */
    .widget-hbox {
        padding-right: 15px !important;
    }
    </style>
    """
    style_widget = widgets.HTML(value=checkbox_style)
    
    # Create checkbox widgets for each year for each category
    training_checkboxes = [widgets.Checkbox(value=False, description=str(year), indent=False) 
                          for year in years]
    validation_checkboxes = [widgets.Checkbox(value=False, description=str(year), indent=False) 
                            for year in years]
    testing_checkboxes = [widgets.Checkbox(value=False, description=str(year), indent=False) 
                         for year in years]
    
    # Create horizontal box layouts for each set of checkboxes with reduced spacing
    box_layout = widgets.Layout(margin='0px 0px 0px 0px', padding='0px 0px 0px 0px')
    training_box = widgets.HBox(training_checkboxes, layout=box_layout)
    validation_box = widgets.HBox(validation_checkboxes, layout=box_layout)
    testing_box = widgets.HBox(testing_checkboxes, layout=box_layout)
    
    submit_button = widgets.Button(
        description="Submit",
        button_style='primary',
        layout=widgets.Layout(width='150px', margin='10px 0px 10px 0px')
    )
    
    warning_output = widgets.HTML(value="")
    selection_display = widgets.HTML(value="", layout=widgets.Layout(margin='10px 0px 10px 0px'))
    result_output = widgets.Output()
    
    # Store the selection state
    selection_state = {'submitted': False, 'values': None}
    
    # Function to get selected years for each category
    def get_selected_years():
        training_years = [int(cb.description) for cb in training_checkboxes if cb.value]
        validation_years = [int(cb.description) for cb in validation_checkboxes if cb.value]
        testing_years = [int(cb.description) for cb in testing_checkboxes if cb.value]
        
        return {
            'training': training_years,
            'validation': validation_years,
            'testing': testing_years
        }
    
    # Function to check for overlaps and missing selections
    def check_selections():
        selected = get_selected_years()
        
        # Check if any selections are made
        has_training = len(selected['training']) > 0
        has_validation = len(selected['validation']) > 0
        has_testing = len(selected['testing']) > 0
        
        missing = []
        if not has_training:
            missing.append("Training")
        if not has_validation:
            missing.append("Validation")
        if not has_testing:
            missing.append("Testing")
        
        # Check for overlaps
        training_set = set(selected['training'])
        validation_set = set(selected['validation'])
        testing_set = set(selected['testing'])
        
        overlaps = []
        
        train_val_overlap = training_set.intersection(validation_set)
        if train_val_overlap:
            overlaps.append(f"Training and Validation overlap on years: {', '.join(map(str, train_val_overlap))}")
            
        train_test_overlap = training_set.intersection(testing_set)
        if train_test_overlap:
            overlaps.append(f"Training and Testing overlap on years: {', '.join(map(str, train_test_overlap))}")
            
        val_test_overlap = validation_set.intersection(testing_set)
        if val_test_overlap:
            overlaps.append(f"Validation and Testing overlap on years: {', '.join(map(str, val_test_overlap))}")
        
        # Generate warning messages
        warning_html = ""
        if missing:
            warning_html += f"<span style='color:orange; font-size:14px;'>⚠️ Please select at least one year for: {', '.join(missing)}</span><br>"
            
        if overlaps:
            warning_html += "<span style='color:red; font-size:14px;'>⚠️ Years cannot be used in multiple categories:</span><br>"
            for overlap in overlaps:
                warning_html += f"<span style='color:red; font-size:14px;'>- {overlap}</span><br>"
            
        # Update the warning display
        warning_output.value = warning_html
        
        return len(missing) == 0 and len(overlaps) == 0
    
    # Function to update the selection display
    def update_display(change=None):
        selected = get_selected_years()
        
        # Format the display content with larger font
        display_html = "<div style='font-size:15px;'>"
        display_html += "<b>Current Selection:</b><br>"
        display_html += f"<b>Training:</b> {', '.join(map(str, selected['training'])) if selected['training'] else 'None'}<br>"
        display_html += f"<b>Validation:</b> {', '.join(map(str, selected['validation'])) if selected['validation'] else 'None'}<br>"
        display_html += f"<b>Testing:</b> {', '.join(map(str, selected['testing'])) if selected['testing'] else 'None'}"
        display_html += "</div>"
        
        # Update the selection display
        selection_display.value = display_html
        
        # Check for warnings
        check_selections()
    
    # Register observers for all checkboxes
    for cb in training_checkboxes + validation_checkboxes + testing_checkboxes:
        cb.observe(update_display, names='value')
    
    def on_submit_clicked(b):
        valid = check_selections()
        selected = get_selected_years()
        
        with result_output:
            result_output.clear_output()
            if valid:
                # Store the selection
                selection_state['submitted'] = True
                selection_state['values'] = selected
                
                print(f"✓ Submitted successfully!")
                print(f"- Training years: {', '.join(map(str, selected['training']))}")
                print(f"- Validation years: {', '.join(map(str, selected['validation']))}")
                print(f"- Testing years: {', '.join(map(str, selected['testing']))}")
            else:
                selection_state['submitted'] = False
                selection_state['values'] = None
                print("⚠️ Please fix the errors before submitting.")
    
    submit_button.on_click(on_submit_clicked)
    
    # Function to get the current selection
    def get_selection():
        if selection_state['submitted']:
            return selection_state['values']
        else:
            return None
    
    # Create layout
    widget_box = widgets.VBox([
        style_widget,
        training_header, training_box,
        validation_header, validation_box,
        testing_header, testing_box,
        selection_display,
        warning_output,
        submit_button, 
        result_output
    ])
    
    # Initial display update
    update_display()
    
    if auto_display:
        display(widget_box)
        
    return widget_box, get_selection