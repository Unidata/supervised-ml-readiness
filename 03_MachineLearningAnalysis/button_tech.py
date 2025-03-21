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

# Constants for configuration
WEATHER_VARS = [
    ('Temperature (F)', 'airtemp_degF'),
    ('Average Wind Speed (mph)', 'windspeed_mph'),
    ('Wind Gust (mph)', 'windgust_mph'),
    ('Relative Humidity (%)', 'rh_percent'),
    ('Precipitation (in)', 'precip_in')
]

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

def create_weather_visualization_controls():
    """Creates control widgets for the Mt. Mitchell weather data visualization."""
    # Variable dropdown
    var_dropdown = widgets.Dropdown(
        options=[
            ('Temperature (F)', 'MITC_airtemp_degF'),
            ('Average Wind Speed (mph)', 'MITC_windspeed_mph'),
            ('Wind Gust (mph)', 'MITC_windgust_mph'),
            ('Relative Humidity (%)', 'MITC_rh_percent'),
            ('Precipitation (in)', 'MITC_precip_in')
        ],
        description='Variable:',
        disabled=False
    )

    # Plot type dropdown
    plot_dropdown = widgets.Dropdown(
        options=['Histogram', 'Time Series'],
        description='Plot type:',
        disabled=False
    )

    # Button for plotting
    plot_button = widgets.Button(description="Plot")
    
    # Output widget to render plots
    output = widgets.Output()
    
    return var_dropdown, plot_dropdown, plot_button, output

def display_mt_mitchell_weather_dashboard(weather_data):
    """Creates and displays interactive dashboard for Mt. Mitchell weather data."""
    # Create interface controls
    var_dropdown, plot_dropdown, plot_button, output = create_weather_visualization_controls()
    
    def on_plot_button_click(b):
        with output:
            clear_output(wait=True)
            
            if plot_dropdown.value == 'Histogram':
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.hist(weather_data[var_dropdown.value], bins=30, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {var_dropdown.label} at Mt. Mitchell (MITC)", fontsize=14)
                ax.set_xlabel(var_dropdown.label)
                ax.set_ylabel("Number of records")
                plt.show()
            
            elif plot_dropdown.value == 'Time Series':
                xdates = pd.to_datetime(weather_data['observation_datetime'])
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.plot(xdates[::100], weather_data[var_dropdown.value][::100], 
                       label=var_dropdown.label, color='orange')
                ax.set_title(f"Time Series of {var_dropdown.label} at Mt. Mitchell (MITC)", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel(var_dropdown.label)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Mt. Mitchell</h3>"), 
           var_dropdown, 
           plot_dropdown, 
           plot_button, 
           output)

def create_input_station_controls():
    """Creates control widgets for input stations data visualization."""
    var_dropdown = widgets.Dropdown(
        options=WEATHER_VARS,
        description='Variable:',
        disabled=False
    )

    plot_dropdown = widgets.Dropdown(
        options=['Histogram', 'Time Series'],
        description='Plot type:',
        disabled=False
    )

    station_dropdown = widgets.Dropdown(
        options=STATIONS,
        description='Station:',
        disabled=False
    )

    plot_button = widgets.Button(description="Plot")
    output = widgets.Output()
    
    return station_dropdown, var_dropdown, plot_dropdown, plot_button, output

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

def create_percentage_widget():
    """Creates widget for specifying training/validation/testing splits."""
    # Create text widgets for percentages
    training = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        description='Training %:',
        layout=widgets.Layout(width='200px')
    )

    validation = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        description='Validation %:',
        layout=widgets.Layout(width='200px')
    )

    testing = widgets.BoundedIntText(
        value=0,
        min=0,
        max=100,
        description='Testing %:',
        layout=widgets.Layout(width='200px')
    )

    submit_button = widgets.Button(description="Submit")
    output = widgets.Output()
    
    def check_percentages(change=None):
        with output:
            output.clear_output()
            total = training.value + validation.value + testing.value
            print(f"Total: {total}%")
                
    def on_submit_clicked(b):
        with output:
            output.clear_output()
            check_percentages()
            total = training.value + validation.value + testing.value
            print("✓ Submitted" if total == 100 else 
                  "⚠️ Make sure the percentages sum to 100% and resubmit.")
    
    # Add observers
    training.observe(check_percentages, names='value')
    validation.observe(check_percentages, names='value')
    testing.observe(check_percentages, names='value')
    submit_button.on_click(on_submit_clicked)
    
    # Layout
    widget_box = widgets.VBox([
        widgets.HTML(value="<h3>Dataset Split Percentages</h3>"),
        training,
        validation,
        testing,
        output,
        submit_button
    ])
    
    display(widget_box)
    
    def get_decimal_values():
        return {
            'training': training.value / 100,
            'validation': validation.value / 100,
            'testing': testing.value / 100
        }
    
    return widget_box, get_decimal_values

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
            grid_template_columns='repeat(3, auto)',
            grid_gap='10px'
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
        widgets.HTML(value="<h3>Select Weather Stations</h3>"),
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

def split_data_temporal(df, final_cutoff='2024-09-28', train_pct=0.6, val_pct=0.2, test_pct=0.2):
    """
    Split data into training, validation, and testing sets based on chronological order.
    The splits are created in this order: Training (earliest dates), Validation (middle dates),
    Testing (latest dates before cutoff), and True Test (after cutoff)
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input DataFrame with a 'date' column
    final_cutoff : str
        Date string for the cutoff between validation and true test sets
    train_pct : float
        Percentage of pre-cutoff data to use for training (default: 0.6)
    val_pct : float
        Percentage of pre-cutoff data to use for validation (default: 0.2)
    test_pct : float
        Percentage of pre-cutoff data to use for testing (default: 0.2)
    """
    # Input validation
    if not abs(train_pct + val_pct + test_pct - 1.0) < 1e-10:
        raise ValueError("Training, validation, and testing percentages must sum to 1.0")
    
    # Convert dates to pandas datetime
    final_cutoff = pd.to_datetime(final_cutoff)
    
    # Create mask for true test set
    true_test_mask = df['date'] > final_cutoff
    
    # Get the remaining data (everything up to final_cutoff)
    remaining_data = df[~true_test_mask].copy()
    remaining_data = remaining_data.sort_values('date')
    
    # Calculate the split points based on percentages
    n_samples = len(remaining_data)
    train_end_idx = int(n_samples * train_pct)
    val_end_idx = int(n_samples * (train_pct + val_pct))  # Changed from test_end_idx
    
    # Get the dates at these split points
    train_cutoff = remaining_data.iloc[train_end_idx]['date']
    val_cutoff = remaining_data.iloc[val_end_idx]['date']  # Changed from test_cutoff
    
    # Create masks for each period in chronological order
    train_mask = df['date'] <= train_cutoff
    val_mask = (df['date'] > train_cutoff) & (df['date'] <= val_cutoff)  # Middle period
    test_mask = (df['date'] > val_cutoff) & (df['date'] <= final_cutoff)  # Latest period before final cutoff
    
    # Split the data
    # Exclude observation_datetime, year_index, and date from features
    X_cols = [col for col in df.columns 
              if 'MITC' not in col 
              and col not in ['observation_datetime', 'year_index', 'date']]
    y_cols = [col for col in df.columns if 'MITC' in col]
    
    # Create the splits in chronological order
    X_train = df.loc[train_mask, X_cols]
    y_train = df.loc[train_mask, y_cols]
    
    X_val = df.loc[val_mask, X_cols]
    y_val = df.loc[val_mask, y_cols]
    
    X_test = df.loc[test_mask, X_cols]
    y_test = df.loc[test_mask, y_cols]
    
    X_true_test = df.loc[true_test_mask, X_cols]
    y_true_test = df.loc[true_test_mask, y_cols]
    
    # Print summary statistics in chronological order
    print("Data split summary:")
    print(f"Training period: {df.loc[train_mask, 'date'].min()} to {df.loc[train_mask, 'date'].max()}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(remaining_data):.1%} of pre-cutoff data)")
    
    print(f"\nValidation period: {df.loc[val_mask, 'date'].min()} to {df.loc[val_mask, 'date'].max()}")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(remaining_data):.1%} of pre-cutoff data)")
    
    print(f"\nTesting period: {df.loc[test_mask, 'date'].min()} to {df.loc[test_mask, 'date'].max()}")
    print(f"Testing samples: {len(X_test)} ({len(X_test)/len(remaining_data):.1%} of pre-cutoff data)")
    
    print(f"\nTrue test period: {df.loc[true_test_mask, 'date'].min()} to {df.loc[true_test_mask, 'date'].max()}")
    print(f"True test samples: {len(X_true_test)}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, X_true_test, y_true_test)

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

def plot_weather_comparison(df, y_pred, transition_date, vars_config=WEATHER_VARS, title='MITC Weather Variables 2024'):
    """Plot historical vs predicted weather variables."""
    fig, axs = plt.subplots(len(vars_config), 1, figsize=(15, 12), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    pred_dates = pd.date_range(start=transition_date, periods=len(y_pred), freq='h')
    pred_dates_num = mdates.date2num(pred_dates)

    for i, ((label, var), ax) in enumerate(zip(vars_config, axs)):
        var_name = f'MITC_{var}'
        mask = df[var_name].notna()
        historical_dates = mdates.date2num(np.array(df.loc[mask, 'observation_datetime']))
        historical_values = df.loc[mask, var_name].values
        
        ax.plot(historical_dates, historical_values, color='#77aadd', alpha=1.0, label='Historical')
        ax.plot(pred_dates_num, y_pred[:, i], color='#ee8866', alpha=0.7, label='Predicted')
        ax.axvline(x=mdates.date2num(transition_date), color='red', linestyle='-', alpha=0.8,
                  label='Day MITC went Offline' if i == 0 else "")
        
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')

    for ax in axs:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=-1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

    plt.setp(axs[-1].xaxis.get_minorticklabels(), rotation=0)
    plt.xlim(mdates.date2num(pd.Timestamp('2024-01-01')), 
            mdates.date2num(pd.Timestamp('2024-12-31')))
    plt.suptitle(title, y=1, fontsize=16)
    plt.tight_layout()
    
    return fig, axs