# Data manipulation and analysis
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Callable

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Machine learning
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

### Columns ###

column_list = ['TEMP_C_0_m', 'TEMP_C_1000_m', 'TEMP_C_5000_m', 
               'T_DEWPOINT_C_0_m', 'T_DEWPOINT_C_1000_m', 'T_DEWPOINT_C_5000_m', 
               'UGRD_m/s_0_m', 'UGRD_m/s_1000_m', 'UGRD_m/s_5000_m', 
               'VGRD_m/s_0_m', 'VGRD_m/s_1000_m', 'VGRD_m/s_5000_m', 
               'PRES_Pa_0_m', 'PRES_Pa_1000_m', 'PRES_Pa_5000_m']

def create_column_filter_widget(columns=column_list):
    """
    Create a simple grid of checkboxes for column selection.
    
    Parameters:
    columns (list): List of column names to include as selectable options.
    
    Returns:
    tuple: (widget, get_selected_columns) where get_selected_columns is a function
           that returns the current list of selected column names.
    """
    # Group columns by altitude level for section headers
    columns_by_altitude = {
        '0_m': [col for col in columns if '_0_m' in col],
        '1000_m': [col for col in columns if '_1000_m' in col],
        '5000_m': [col for col in columns if '_5000_m' in col]
    }
    
    # Create widgets for each section
    sections = []
    all_checkboxes = {}
    
    # Function to create a section of checkboxes
    def create_section(title, cols):
        # Create checkboxes for this section
        section_checkboxes = {
            col: widgets.Checkbox(
                value=False,
                description=col.replace('_0_m', '').replace('_1000_m', '').replace('_5000_m', ''),
                disabled=False,
                indent=False
            ) for col in cols
        }
        
        # Add these checkboxes to the global dictionary
        all_checkboxes.update(section_checkboxes)
        
        # Create a grid for this section
        grid = widgets.GridBox(
            children=list(section_checkboxes.values()),
            layout=widgets.Layout(
                grid_template_columns='repeat(3, auto)',
                grid_gap='5px',
                width='100%',
                padding='2px'
            )
        )
        
        # Return section widget
        return widgets.VBox([
            widgets.HTML(value=f"<b>{title}</b>"),
            grid
        ])
    
    # Create each section
    if columns_by_altitude['0_m']:
        sections.append(create_section("0 m Level", columns_by_altitude['0_m']))
    
    if columns_by_altitude['1000_m']:
        sections.append(create_section("1000 m Level", columns_by_altitude['1000_m']))
    
    if columns_by_altitude['5000_m']:
        sections.append(create_section("5000 m Level", columns_by_altitude['5000_m']))
    
    # Create output widget for selection summary
    output = widgets.Output()
    
    # Update function
    def update_display(change=None):
        selected_columns = [col for col, checkbox in all_checkboxes.items() if checkbox.value]
        
        with output:
            output.clear_output()
            if selected_columns:
                print(f"Selected {len(selected_columns)} of {len(columns)} columns:")
                
                # Group selected columns by altitude for display
                selected_by_altitude = {
                    "0m": [c for c in selected_columns if "_0_m" in c],
                    "1000m": [c for c in selected_columns if "_1000_m" in c],
                    "5000m": [c for c in selected_columns if "_5000_m" in c]
                }
                
                for level, cols in selected_by_altitude.items():
                    if cols:
                        print(f"  - {level}: {len(cols)} columns selected")
            else:
                print("No columns selected")
    
    # Connect event handlers
    for checkbox in all_checkboxes.values():
        checkbox.observe(update_display, names='value')
    
    # Create dividers between sections
    dividers = [widgets.HTML(value="<hr style='margin: 10px 0'>") for _ in range(len(sections)-1)]
    
    # Interleave sections and dividers
    container_items = []
    for i, section in enumerate(sections):
        container_items.append(section)
        if i < len(dividers):
            container_items.append(dividers[i])
    
    # Add output at the end
    container_items.append(widgets.HTML(value="<hr style='margin: 10px 0'>"))
    container_items.append(output)
    
    # Create main container
    main_widget = widgets.VBox(container_items)
    
    # Function to get selected columns
    def get_selected_columns():
        return [col for col, checkbox in all_checkboxes.items() if checkbox.value]
    
    # Initial display update
    update_display()
    
    return main_widget, get_selected_columns

class HistogramWidget:
    def __init__(self, df, bins=21, figsize=(7, 4)):
        """
        Initialize the histogram widget for plotting precipitation data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataframe containing the data to plot
        bins : int, optional
            Number of bins for the histogram (default: 21)
        figsize : tuple, optional
            Figure size as (width, height) in inches (default: (7, 4))
        """
        self.df = df
        self.bins = bins
        self.figsize = figsize
        
        # Verify the dataframe has the 'ptype' column
        if 'ptype' not in df.columns:
            raise ValueError("Dataframe must contain a 'ptype' column with 'rain' and 'snow' values")
        
        # Get all columns except 'ptype' for dropdown
        self.numeric_columns = [col for col in df.columns if col != 'ptype' and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric columns found in dataframe besides 'ptype'")
        
        # Initialize widget components
        self.value_dropdown = widgets.Dropdown(
            options=self.numeric_columns,
            value=self.numeric_columns[0],
            description='Input Feature:',
            disabled=False
        )
        
        # Separate opacity sliders for rain and snow
        self.rain_alpha_slider = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Rain opacity:',
            disabled=False,
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        self.snow_alpha_slider = widgets.FloatSlider(
            value=0.7,
            min=0.1,
            max=1.0,
            step=0.1,
            description='Snow opacity:',
            disabled=False,
            continuous_update=False,
            style={'description_width': 'initial'}
        )
        
        self.plot_button = widgets.Button(
            description='Update Plot',
            disabled=False,
            button_style='', 
            tooltip='Click to update the plot'
        )
        
        # Set up the layout
        self.plot_output = widgets.Output()
        self.plot_button.on_click(self.update_plot)
        
        # Display the widget
        self.widget = widgets.VBox([
            self.value_dropdown,
            self.rain_alpha_slider,
            self.snow_alpha_slider,
            self.plot_button,
            self.plot_output
        ])
    
    def update_plot(self, b):
        """Update the histogram plot based on current widget values."""
        with self.plot_output:
            self.plot_output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Get current values from widgets
            value_column = self.value_dropdown.value
            rain_alpha = self.rain_alpha_slider.value
            snow_alpha = self.snow_alpha_slider.value
            
            # Split data by ptype and plot
            rain_data = self.df[self.df['ptype'] == 'rain'][value_column].dropna()
            snow_data = self.df[self.df['ptype'] == 'snow'][value_column].dropna()
            
            # Determine bin edges based on the full dataset
            all_data = self.df[value_column].dropna()
            bin_edges = np.histogram_bin_edges(all_data, bins=self.bins)
            
            # Plot snow histogram FIRST (on the bottom) with specific color #ee8866 and its own opacity
            if not snow_data.empty:
                ax.hist(snow_data, bins=bin_edges, alpha=snow_alpha, color='#ee8866', label='Snow')
            
            # Plot rain histogram SECOND (on top) with specific color #77aadd and its own opacity
            if not rain_data.empty:
                ax.hist(rain_data, bins=bin_edges, alpha=rain_alpha, color='#77aadd', label='Rain')
            
            # Add labels and legend
            ax.set_xlabel(value_column)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {value_column} by Precipitation Type')
            ax.legend()
            
            plt.tight_layout()
            plt.show()
    
    def display(self):
        """Display the widget."""
        display(widgets.HTML(value="<h3>Input Features</h3>"),
                self.widget)
        # Generate initial plot
        self.update_plot(None)

def create_correlation_plot_controls():
    """Creates control widgets for correlation plots."""
    # Add title label with larger size and bold styling
    title = widgets.HTML(value='<h3 style="font-weight: bold; margin: 0; padding: 0;">Comparison Plot</h3>')
    
    var_dropdown = widgets.Dropdown(
        options=[
            ('Variables at 0 m', '0m'),
            ('Variables at 1000 m', '1000m'),
            ('Variables at 5000 m', '5000m')
        ],
        description='Level:',
        disabled=False
    )

    plot_button = widgets.Button(description="Plot")
    output = widgets.Output()
    
    return title, var_dropdown, plot_button, output


def display_correlation_plot_dashboard(base_url="https://elearning.unidata.ucar.edu/dataeLearning/Cybertraining/applications/media/pairplot_"):
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
                f'<img src="{image_url}" width="800px"></a></center>'
            ))
    
    plot_button.on_click(update_image)
    display(title, var_dropdown, plot_button, output)

"""
# Code used to make the pairplots
colors = {'rain': '#77aadd', 'snow': '#ee8866'}
markers = {'rain': '^', 'snow': 'o'} 
title = "Comparison Plots at 5000 m AGL"
f = sns.pairplot(df[['TEMP_C_5000_m', 
                 'T_DEWPOINT_C_5000_m', 
                 'PRES_Pa_5000_m',
                 'UGRD_m/s_5000_m', 
                 'VGRD_m/s_5000_m', 
                 'ptype']].sample(2000), hue='ptype', palette=colors, markers=markers)
f.fig.suptitle(title,y=1.005, fontsize=16)
"""

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
        "LogisticRegression (Linear)": "logistic_regression",
        "Random Forest Classifier": "random_forest"
    }
    # Use a list to store the selection (mutable)
    selection = [None]
    output = widgets.Output()

    buttons = [
        widgets.Button(
            description=name,
            layout=widgets.Layout(width='300px', height='125px', margin='10px')
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

def train_button(algorithm, X_train_filtered, y_train):
    """
    Creates a training button that uses either Random Forest or Logistic Regression.
    
    Parameters:
    algorithm (str): Either "random_forest" or "logistic_regression"
    X_train_filtered (array): Feature matrix for training
    y_train (array): Target values for training
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    
    # Define global model variable
    global trained_model_obj
    trained_model_obj = None
    
    # Create output widgets
    output = widgets.Output()
    status_label = widgets.Label(value='Click button to train model')
    
    # Create button with generic label and 50% wider
    train_button = widgets.Button(
        description='Train Model', 
        layout=widgets.Layout(width='300px', height='80px')
    )

    def train_model(b):
        global trained_model_obj
        with output:
            clear_output()
            
            # Print dataset information
            unique_classes = np.unique(y_train)
            class_counts = {cls: np.sum(y_train == cls) for cls in unique_classes}
            class_distribution = {cls: f"{count} ({count/len(y_train)*100:.1f}%)" for cls, count in class_counts.items()}
            
            print(f"Dataset information:")
            print(f"- Features: {X_train_filtered.shape[1]} columns, {X_train_filtered.shape[0]} samples")
            print(f"- Class names: {unique_classes}")
            print(f"- Class distribution: {class_distribution}")
            print(f"- Selected algorithm: {algorithm.replace('_', ' ').title()}")
            
            try:
                if algorithm == "random_forest":
                    print("\nTraining Random Forest model...")
                    
                    # Create and train Random Forest
                    trained_model_obj = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                    trained_model_obj.fit(X_train_filtered, y_train)
                    
                elif algorithm == "logistic_regression":
                    print("\nTraining Logistic Regression model...")
                    
                    # Create and train Logistic Regression
                    trained_model_obj = LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                    trained_model_obj.fit(X_train_filtered, y_train)
                
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                print(f"\n{algorithm.replace('_', ' ').title()} training completed!")
                print(f"Classes: {trained_model_obj.classes_}")
                status_label.value = 'Model trained successfully!'
                
            except Exception as e:
                print(f"Error during training: {str(e)}")
                trained_model_obj = None
                status_label.value = 'Error during training'
                
            train_button.description = 'Train Model'
            train_button.disabled = False

    train_button.on_click(train_model)

    display(widgets.VBox([train_button, status_label]))
    display(output)

    def get_model():
        """Returns the trained model"""
        return trained_model_obj
            
    return get_model

def train_val_test_split(df, 
                         y_col='ptype', 
                         train_size=0.7, 
                         val_size=0.15, 
                         test_size=0.15, 
                         random_state=None):
    """
    Split a pandas DataFrame into features and target sets for training, validation, and testing.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to split.
    y_col : str, default='ptype'
        Column name to use as the target variable.
    train_size : float, default=0.7
        Proportion of the dataset to include in the training split.
    val_size : float, default=0.15
        Proportion of the dataset to include in the validation split.
    test_size : float, default=0.15
        Proportion of the dataset to include in the test split.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Features for training.
    y_train : pandas.Series
        Target for training.
    X_val : pandas.DataFrame
        Features for validation.
    y_val : pandas.Series
        Target for validation.
    X_test : pandas.DataFrame
        Features for testing.
    y_test : pandas.Series
        Target for testing.
    """
    from sklearn.model_selection import train_test_split
    
    # Verify that the proportions sum to 1
    if abs(train_size + val_size + test_size - 1.0) > 1e-10:
        raise ValueError("train_size, val_size, and test_size should sum to 1.0")
    
    total_samples = len(df)
    
    # Calculate exact sample counts that won't exceed the total
    train_count = int(total_samples * train_size)
    val_count = int(total_samples * val_size)
    test_count = total_samples - train_count - val_count
    
    # Ensure we don't get rounding errors that exceed total
    if train_count + val_count + test_count != total_samples:
        # We prioritize keeping training set size as requested
        test_count = total_samples - train_count - val_count
        if abs(train_count + val_count + test_count - total_samples) == 1:
            print(f"Off-by-one adjustment made: Train={train_count}, Val={val_count}, Test={test_count}")
    
    # First, split off the training set (using absolute count)
    from sklearn.utils import shuffle
    
    # Shuffle the dataset
    shuffled_df = shuffle(df, random_state=random_state)
    
    # Split into train, val, and test
    df_train = shuffled_df.iloc[:train_count]
    df_val = shuffled_df.iloc[train_count:train_count+val_count]
    df_test = shuffled_df.iloc[train_count+val_count:]
    
    # Split each dataframe into X and y
    X_train = df_train.drop(columns=[y_col])
    y_train = df_train[y_col]
    
    X_val = df_val.drop(columns=[y_col])
    y_val = df_val[y_col]
    
    X_test = df_test.drop(columns=[y_col])
    y_test = df_test[y_col]
    
    # Will output scaled data
    X_train_scaled, X_val_scaled, X_test_scaled = standardize_data(X_train, X_val, X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


def plot_confusion_matrix(class_labels, y_true, y_pred, title='Confusion Matrix', figsize=(4.5, 3.5), cmap='Blues'):
    """
    Creates and plots a confusion matrix as a heatmap.
    
    Parameters:
    -----------
    class_labels : list or array
        Labels for the classes in the classification task.
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str, optional (default='Confusion Matrix')
        Title of the plot.
    figsize : tuple, optional (default=(4.5, 3.5))
        Figure size (width, height) in inches.
    cmap : str, optional (default='Blues')
        Colormap for the heatmap.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the confusion matrix plot.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap with labels
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_labels,
                yticklabels=class_labels,
                cbar=False, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True Observation')
    ax.set_title(title)
    
    # Return the figure (useful if you want to save it or modify it further)
    return fig

def standardize_data(X_train, X_val, X_test):
    """
    Standardize multiple datasets using a StandardScaler fit only on the training data.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training data to fit the scaler and transform
    X_val : pandas.DataFrame
        Validation data to transform
    X_test : pandas.DataFrame
        Test data to transform
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_val_scaled, X_test_scaled)
        All as pandas DataFrames with original column names and indices preserved
    """
    from sklearn.preprocessing import StandardScaler
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler to the training data and transform it
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform the validation and test data using the same scaler
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled

def classification_model_eval(model, X_test, y_test, title='Model Evaluation'):
    """
    Evaluate a trained classification model with confusion matrix and metrics.
    
    Parameters:
    model: The trained classifier model
    X_test: Test features
    y_test: True test labels
    title: Title for the confusion matrix plot
    
    Returns:
    dict: Dictionary containing all evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get class labels
    class_labels = model.classes_
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(class_labels, 
                               y_true=y_test, 
                               y_pred=y_pred, 
                               title=title)
    plt.show()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, 
                                                             average=None, 
                                                             labels=['rain', 'snow'])
    accuracy = (accuracy_score(y_test, y_pred)) * 100
    
    # Print results
    print(f"{title} Metrics")
    print("Used this algo + these input features")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Rain Precision: {precision[0]:.3f}")
    print(f"Snow Precision: {precision[1]:.3f}")
    print(f"Rain Recall: {recall[0]:.3f}")
    print(f"Snow Recall: {recall[1]:.3f}")
    print(f"Rain F1 Score: {f1[0]:.3f}")
    print(f"Snow F1 Score: {f1[1]:.3f}")
    
    # Return metrics as a dictionary for further use if needed
    metrics = {
        'accuracy': accuracy,
        'rain_precision': precision[0],
        'snow_precision': precision[1],
        'rain_recall': recall[0],
        'snow_recall': recall[1],
        'rain_f1': f1[0],
        'snow_f1': f1[1]
    }
    
    return metrics