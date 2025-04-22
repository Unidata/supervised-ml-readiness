# Data yearanipulation and analysis
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Callable

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# Machine learning
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error
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


def display_blackhawk_gauge_dashboard(hydrograph_data):
    """Creates and displays interactive dashboard for Blackhawk Gauge hydrograph data."""
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
                ax.hist(hydrograph_data['daily_mean_discharge_N_BLKHAWK'].dropna(), bins=30, 
                       color='skyblue', edgecolor='black')
                ax.set_title("Histogram of Daily Mean Discharge at Blackhawk Gauge", fontsize=14)
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
                    ax.plot(year_data['day_of_year'], year_data['daily_mean_discharge_N_BLKHAWK'], 
                           label=str(year))
                
                ax.set_title("Annual Comparison of Daily Mean Discharge at Blackhawk Gauge", fontsize=14)
                ax.set_xlabel("Day of Year")
                ax.set_ylabel("Daily Mean Discharge")
                ax.legend(title="Year")
                plt.show()
                
            elif plot_dropdown.value == 'Yearly Comparison':
                # Calculate yearly averages
                yearly_avg = hydrograph_data.groupby('year')['daily_mean_discharge_N_BLKHAWK'].mean().reset_index()
                
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                ax.bar(yearly_avg['year'].astype(str), yearly_avg['daily_mean_discharge_N_BLKHAWK'], 
                      color='green', edgecolor='black')
                ax.set_title("Yearly Average Discharge at Blackhawk Gauge", fontsize=14)
                ax.set_xlabel("Year")
                ax.set_ylabel("Average Daily Mean Discharge")
                plt.xticks(rotation=45)
                plt.show()
    
    # Connect button to visualization update function
    plot_button.on_click(on_plot_button_click)
    
    # Display dashboard elements
    display(widgets.HTML(value="<h3>Blackhawk Gauge Hydrograph Dashboard</h3>"), 
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

def discharge_pairplot(df, prefix='daily_mean_discharge', sample_size=1000, 
                             random_state=42, figsize=(9, 9), dpi=350):
    """
    Create a pairplot of columns that start with a specific prefix.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data.
    prefix : str, optional
        The prefix to filter columns by. Default is 'daily_mean_discharge'.
    sample_size : int, optional
        Number of random rows to sample. Default is 1000.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    figsize : tuple, optional
        Size of the figure in inches (width, height). Default is (5, 5).
    dpi : int, optional
        Resolution of the figure in dots per inch. Default is 400.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The pairplot figure object.
    """
    # Sample random rows
    sample_df = df.sample(n=sample_size, random_state=random_state)
    
    # Select columns that start with the prefix
    columns_to_plot = [col for col in sample_df.columns if col.startswith(prefix)]
    
    # Create a copy with renamed columns
    plot_df = sample_df[columns_to_plot].copy()
    
    # Rename columns to remove the prefix
    plot_df.columns = [col.replace(f'{prefix}_', '') for col in plot_df.columns]
    
    # Close any existing figures to prevent the empty figure
    plt.close('all')
    
    # Create the pairplot
    g = sns.pairplot(plot_df, kind="scatter", diag_kind="kde", 
                    plot_kws={'alpha': 0.3}, height=figsize[0]/len(columns_to_plot))
    plt.tight_layout()
    
    # Return the plot without displaying it
    return g


def algorithm_selection():
    """Creates widget for algorithm selection and returns selected value via callback."""
    algorithm_options = {
        "Linear Regression": "linear_regression",
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
    'daily_mean_discharge_N_BLKHAWK'
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
                    # Use regular XGBRegressor instead of MultiXGBRegressor
                    selected_model = XGBRegressor(
                        n_estimators=100,
                        tree_method='hist',
                        random_state=42
                    )
                    selected_model.fit(X_train_filtered, y_train)
                    print("XGBoost model training completed!")
                    print(f"Model object created and trained: {selected_model is not None}")
                    status_label.value = 'Model trained successfully!'
                    
                elif selected_algo == "linear_regression":
                    print("Running Linear Regression model...")
                    # For single target, use regular LinearRegression
                    selected_model = LinearRegression()
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
    Filter DataFrame to keep only columns with specified prefixes plus year, day of year, 
    and hour of day columns if they exist.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    prefix_values (list): List of prefix values to match
    
    Returns:
    pandas.DataFrame: Filtered DataFrame with only the specified columns
    """
    # Print original column count
    print(f"Original DataFrame: {len(df.columns)} columns")
    
    # Start with time-related columns if they exist
    time_columns = ['year', 'day_of_year', 'hour_of_day']
    columns_to_keep = [col for col in time_columns if col in df.columns]
    
    # Add any column that starts with our prefix values
    for prefix in prefix_values:
        matching_columns = [col for col in df.columns if col.startswith(prefix)]
        columns_to_keep.extend(matching_columns)
    
    # Create filtered dataframe
    filtered_df = df[columns_to_keep]
    
    # Print new column count
    print(f"Filtered DataFrame: {len(filtered_df.columns)} columns")
    
    return filtered_df

def model_eval(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_type: str = 'Validation'
) -> None:
    """
    Evaluates a trained model using test data and prints performance metrics for MITC.
    Single-target version.
    """
    if eval_type not in ['Testing', 'Validation', None]:
        raise ValueError(f"eval_type must be one of ['Testing', 'Validation', None]")
    
    # Define header first
    header = f"{eval_type} Metrics" if eval_type else "Metrics"

    y_pred = model.predict(X_test)
    
    # Handle potentially 2D output from some models
    if len(np.array(y_pred).shape) > 1:
        y_pred = y_pred.ravel()
    
    # Calculate metrics for single target
    from sklearn.metrics import mean_squared_error
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_val = r2_score(y_test, y_pred)
    
    # Get target name
    target_name = y_test.name if hasattr(y_test, 'name') and y_test.name else "target"
    
    print(header)
    print(f"\nModel Type: {type(model).__name__}")
    
    print("\nRMSE:")
    print(f" {target_name}:\t{rmse_val:.4f}")
    
    print("\nR² Score:")
    print(f" {target_name}:\t{r2_val:.4f}")
    
    print(f"\nR² Score:\t{r2_val:.2f}")


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
    """Creates widget for specifying training/validation/testing splits using SelectMultiple.
    
    Returns:
        widget_box: The widget interface
        get_selection: A function that returns the current selection as integers
    """
    # Predefined years
    years = np.array([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    total_years = len(years)
    
    # Convert years to list of strings for the options
    year_options = [str(year) for year in years]
    
    # Create section headers with larger font
    header_style = "font-size: 16px; font-weight: bold; margin-bottom: 5px;"
    
    # Instruction text
    instruction_text = widgets.HTML(
        value="<span style='font-size: 14px;'><b>Selection Instructions:</b> Hold Ctrl/Cmd key while clicking to select multiple items. Use Shift to select a range.</span>",
        layout=widgets.Layout(margin='0px 0px 15px 0px')
    )
    
    # Headers for each category
    training_header = widgets.HTML(value=f"<span style='{header_style}'>Training Years:</span>")
    validation_header = widgets.HTML(value=f"<span style='{header_style}'>Validation Years:</span>")
    testing_header = widgets.HTML(value=f"<span style='{header_style}'>Testing Years:</span>")
    
    # Create SelectMultiple widgets for each category with horizontal layout
    select_layout = widgets.Layout(width='400px', height='100px')  # Taller to show more options at once
    
    training_select = widgets.SelectMultiple(
        options=year_options,
        value=[],
        rows=5,
        description='',
        disabled=False,
        layout=select_layout
    )
    
    validation_select = widgets.SelectMultiple(
        options=year_options,
        value=[],
        rows=5,
        description='',
        disabled=False,
        layout=select_layout
    )
    
    testing_select = widgets.SelectMultiple(
        options=year_options,
        value=[],
        rows=5,
        description='',
        disabled=False,
        layout=select_layout
    )
    
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
        training_years = [int(year) for year in training_select.value]
        validation_years = [int(year) for year in validation_select.value]
        testing_years = [int(year) for year in testing_select.value]
        
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
    
    # Function to update the selection display with percentages
    def update_display(change=None):
        selected = get_selected_years()
        
        # Calculate percentages
        training_pct = len(selected['training']) / total_years * 100
        validation_pct = len(selected['validation']) / total_years * 100
        testing_pct = len(selected['testing']) / total_years * 100
        
        # Format the display content with larger font and percentages
        display_html = "<div style='font-size:15px;'>"
        display_html += "<b>Current Selection:</b><br>"
        display_html += f"<b>Training:</b> {', '.join(map(str, selected['training'])) if selected['training'] else 'None'} "
        display_html += f"({len(selected['training'])} years, {training_pct:.1f}% of total)<br>"
        
        display_html += f"<b>Validation:</b> {', '.join(map(str, selected['validation'])) if selected['validation'] else 'None'} "
        display_html += f"({len(selected['validation'])} years, {validation_pct:.1f}% of total)<br>"
        
        display_html += f"<b>Testing:</b> {', '.join(map(str, selected['testing'])) if selected['testing'] else 'None'} "
        display_html += f"({len(selected['testing'])} years, {testing_pct:.1f}% of total)"
        
        display_html += "</div>"
        
        # Update the selection display
        selection_display.value = display_html
        
        # Check for warnings
        check_selections()
    
    # Register observers for the SelectMultiple widgets
    training_select.observe(update_display, names='value')
    validation_select.observe(update_display, names='value')
    testing_select.observe(update_display, names='value')
    
    def on_submit_clicked(b):
        valid = check_selections()
        selected = get_selected_years()
        
        with result_output:
            result_output.clear_output()
            if valid:
                # Store the selection
                selection_state['submitted'] = True
                selection_state['values'] = selected
                
                # Calculate percentages for the final output
                training_pct = len(selected['training']) / total_years * 100
                validation_pct = len(selected['validation']) / total_years * 100
                testing_pct = len(selected['testing']) / total_years * 100
                
                print(f"✓ Submitted successfully!")
                print(f"- Training years: {', '.join(map(str, selected['training']))} ({training_pct:.1f}%)")
                print(f"- Validation years: {', '.join(map(str, selected['validation']))} ({validation_pct:.1f}%)")
                print(f"- Testing years: {', '.join(map(str, selected['testing']))} ({testing_pct:.1f}%)")
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
    
    # Create horizontal layout for the select widgets
    selects_hbox = widgets.HBox([
        widgets.VBox([training_header, training_select]),
        widgets.VBox([validation_header, validation_select]),
        widgets.VBox([testing_header, testing_select])
    ], layout=widgets.Layout(margin='0px 0px 15px 0px'))
    
    # Create layout for the whole widget
    widget_box = widgets.VBox([
        instruction_text,
        selects_hbox,
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