import pandas as pd

def formatted_error_dataframe(df, per_id=False, thresholds=None, formats=None):
    default_thresholds = {
        'mae_co2_outdoor__ppm': (25, 75),
        'rmae_valve_frac__0': (10/100, 20/100),
        'mae_occupancy__p': (1.0, 2.0),
        'rmse_co2_outdoor__ppm': None,
        'rmse_valve_frac__0': None,
        'rmse_occupancy__p': None
    }
    
    default_formats = {
        '__ppm': '{:.0f}',
        '__0': '{:.0%}',
        '__p': '{:.1f}'
    }

    if per_id:
        # Calculate essential statistics for the error values, per id
        df_stats = df.groupby('id').describe().stack().filter(regex='^mae_|^rmae_|^rmse_')
        df_stats = df_stats.loc[df_stats.index.get_level_values(1).isin(['mean', 'min', 'max'])]
    else:
        # Calculate essential statistics for all errors
        df_stats = df.describe().filter(regex='^mae_|^rmae_|^rmse_')
        df_stats = df_stats.loc[df_stats.index.get_level_values(0).isin(['mean', 'min', 'max'])]
    
    if thresholds is None:
        thresholds = default_thresholds
    else:
        thresholds = {**default_thresholds, **thresholds}

    if formats is None:
        formats = default_formats
        
    def color_cells(column):
        if column.name == 'rmse_valve_frac__0':
            threshold_key = 'rmae_valve_frac__0'
        elif (column.name.startswith('mae_') or column.name.startswith('rmae_')):
            threshold_key = column.name
        else:
            threshold_key = column.name.replace('rmse_', 'mae_')

        lower_threshold, upper_threshold = thresholds[threshold_key]
        style = pd.Series('', index=column.index)
        mask_green = column <= lower_threshold
        mask_orange = (column > lower_threshold) & (column <= upper_threshold)
        style[mask_green] = 'background-color: green'
        style[mask_orange] = 'background-color: orange'
        style[column > upper_threshold] = 'background-color: red'
        return style

    columns_to_style = [col for col in df_stats.columns if any(col.endswith(suffix) for suffix in formats)]
    styled_df = df_stats.style.apply(color_cells, axis=0, subset=columns_to_style)
        
    # Create dictionary of column formatting
    column_formatting = {}
    for col in df_stats.columns:
        suffix = col.split('__')[-1]
        fmt = formats.get(f'__{suffix}', None)
        if fmt is not None:
            column_formatting[col] = fmt
            
    return styled_df.format(column_formatting)
