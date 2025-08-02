import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def create_output_directory():
    """Create a directory for storing outputs with timestamp"""
    base_dir = 'spi_zonal_statistics_output'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'spi_analysis_{timestamp}')
    os.makedirs(output_dir)

    return output_dir


def calculate_spi_zonal_statistics(nc_file, shp_file, start_year=2013):
    """
    Calculate zonal statistics for cleaned SPI data from 2013 onwards.

    Parameters:
    nc_file (str): Path to cleaned SPI NetCDF file
    shp_file (str): Path to ADM2 shapefile
    start_year (int): Starting year for analysis (default: 2013)

    Returns:
    tuple: (results_dataframe, output_directory_path)
    """
    # Create output directory
    output_dir = create_output_directory()
    output_csv = os.path.join(output_dir, f'spi_zonal_stats_{start_year}onwards.csv')

    # Load shapefile
    print(f"Loading shapefile: {shp_file}")
    shp_df = gpd.read_file(shp_file)
    print(f"Found {len(shp_df)} administrative units")
    print("Shapefile columns:", shp_df.columns.tolist())

    # Verify ADM2_PCODE exists
    if 'ADM2_PCODE' not in shp_df.columns:
        raise KeyError("ADM2_PCODE field is not found in the shapefile.")

    # Load cleaned NetCDF data
    print(f"Loading cleaned SPI NetCDF file: {nc_file}")
    nc_ds = xr.open_dataset(nc_file)

    # Get SPI variable (assuming it's SPI3)
    spi_var = 'SPI3'
    if spi_var not in nc_ds.data_vars:
        # Try to find any SPI variable
        spi_vars = [var for var in nc_ds.data_vars if 'spi' in var.lower()]
        if spi_vars:
            spi_var = spi_vars[0]
            print(f"Using SPI variable: {spi_var}")
        else:
            raise KeyError("No SPI variable found in the dataset")

    # Print time range information
    time_range = pd.to_datetime(nc_ds.time.values)
    print(f"Full dataset time range: {time_range.min()} to {time_range.max()}")

    # Select data from start_year onwards
    spi_data = nc_ds[spi_var].sel(time=slice(f"{start_year}-01-01", None))

    # Verify we have data after filtering
    times = pd.to_datetime(spi_data.time.values)
    if len(times) == 0:
        raise ValueError(
            f"No data found for period from {start_year} onwards. Please check the date range in your NetCDF file.")

    print(f"Processing {len(times)} months from {start_year} onwards...")
    print(f"Selected time range: {times.min()} to {times.max()}")

    # Extract coordinate information
    lat_values = nc_ds['lat'].values
    lon_values = nc_ds['lon'].values

    # Compute spatial bounds
    lat_min, lat_max = lat_values.min(), lat_values.max()
    lon_min, lon_max = lon_values.min(), lon_values.max()

    print(f"Spatial bounds - Lat: {lat_min:.2f} to {lat_max:.2f}, Lon: {lon_min:.2f} to {lon_max:.2f}")

    # Initialize results
    results = []
    missing_data = []  # Track missing data

    # Process each timestep
    total_times = len(times)
    print(f"\nProcessing {total_times} time steps...")

    for i, time in enumerate(times, 1):
        year = time.year
        month = time.month
        print(f"Processing: {time.strftime('%Y-%m')} ({i}/{total_times})")

        # Get the SPI data for this time
        time_data = spi_data.sel(time=time)

        # Check for valid data
        valid_data_count = (~np.isnan(time_data)).sum().values
        total_data_count = time_data.size

        if valid_data_count == 0:
            print(f"  Warning: No valid data for {time.strftime('%Y-%m')}")
            # Record all districts for this time step as missing
            for j in range(len(shp_df)):
                missing_data.append({
                    'year': year,
                    'month': month,
                    'date': time.strftime('%Y-%m-%d'),
                    'adm2_pcode': shp_df.iloc[j]['ADM2_PCODE'],
                    'adm2_name': shp_df.iloc[j]['ADM2_EN'] if 'ADM2_EN' in shp_df.columns else None,
                    'adm1_name': shp_df.iloc[j]['ADM1_EN'] if 'ADM1_EN' in shp_df.columns else None,
                })
            continue

        print(
            f"  Valid data points: {valid_data_count}/{total_data_count} ({valid_data_count / total_data_count * 100:.1f}%)")

        # Create transform for this slice
        transform = from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            time_data.shape[1], time_data.shape[0]
        )

        # Calculate zonal statistics
        try:
            stats = zonal_stats(
                shp_df,
                time_data.values,
                affine=transform,
                stats=["mean", "min", "max", "std", "count"],
                nodata=np.nan
            )

            # Add metadata to results
            for j, stat in enumerate(stats):
                # Handle None values that can occur with zonal_stats
                if stat is None:
                    stat = {'mean': np.nan, 'min': np.nan, 'max': np.nan, 'std': np.nan, 'count': 0}

                stat['year'] = year
                stat['month'] = month
                stat['date'] = time.strftime('%Y-%m-%d')
                stat['adm2_pcode'] = shp_df.iloc[j]['ADM2_PCODE']  # <-- Use ADM2_PCODE

                # Add administrative names if available
                for col in ['ADM2_EN', 'ADM1_EN']:
                    if col in shp_df.columns:
                        stat[col.lower()] = shp_df.iloc[j][col]

                results.append(stat)

        except Exception as e:
            print(f"  Error processing {time.strftime('%Y-%m')}: {str(e)}")
            continue

    # Convert to DataFrame
    print("\nCreating final DataFrame...")
    df = pd.DataFrame(results)

    if len(df) == 0:
        raise ValueError(
            "No results generated. Please check your input data and ensure the shapefile overlaps with the NetCDF data.")

    # Add SPI interpretation columns
    print("Adding SPI interpretation categories...")
    df['spi_category'] = pd.cut(
        df['mean'].fillna(0),
        bins=[-np.inf, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, np.inf],
        labels=['Extremely Dry', 'Severely Dry', 'Moderately Dry', 'Mildly Dry',
                'Normal', 'Mildly Wet', 'Moderately Wet', 'Severely Wet', 'Extremely Wet']
    )

    # Add drought severity based on SPI thresholds
    df['drought_severity'] = pd.cut(
        df['mean'].fillna(0),
        bins=[-np.inf, -2, -1.5, -1, np.inf],
        labels=['Extreme Drought', 'Severe Drought', 'Moderate Drought', 'No Drought']
    )

    # Sort by administrative unit and date
    df = df.sort_values(['adm2_pcode', 'year', 'month']).reset_index(drop=True)

    # Save results
    print(f"Saving results to: {output_csv}")
    df.to_csv(output_csv, index=False)

    # Save missing data info if any
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        missing_file = os.path.join(output_dir, 'missing_zonal_stats.csv')
        missing_df.to_csv(missing_file, index=False)
        print(f"\nSaved list of missing zonal stats to: {missing_file}")
    else:
        missing_file = None
        print("\nNo missing zonal stats detected.")

    # Create summary statistics
    create_summary_statistics(df, output_dir)

    # Create visualizations
    create_summary_plots(df, output_dir)

    # Save processing information (now pass missing_file)
    save_processing_info(nc_file, shp_file, times, shp_df, output_dir, df, missing_file)

    print(f"\nAnalysis complete!")
    print(f"Results saved to directory: {output_dir}")
    print(f"Main results file: {output_csv}")
    print(f"Total records processed: {len(df):,}")

    return df, output_dir


def create_summary_statistics(df, output_dir):
    """Create summary statistics for the SPI analysis"""
    print("Creating summary statistics...")

    summary_file = os.path.join(output_dir, 'summary_statistics.txt')

    with open(summary_file, 'w') as f:
        f.write("SPI ZONAL STATISTICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Number of administrative units: {df['adm2_pcode'].nunique()}\n")
        f.write(
            f"Time period: {df['year'].min()}-{df['month'].min():02d} to {df['year'].max()}-{df['month'].max():02d}\n")
        f.write(f"Number of months: {len(df.groupby(['year', 'month']))}\n\n")

        # SPI value statistics
        f.write("SPI VALUE STATISTICS:\n")
        f.write(f"Mean SPI: {df['mean'].mean():.3f}\n")
        f.write(f"Std SPI: {df['mean'].std():.3f}\n")
        f.write(f"Min SPI: {df['min'].min():.3f}\n")
        f.write(f"Max SPI: {df['max'].max():.3f}\n\n")

        # Category distribution
        f.write("SPI CATEGORY DISTRIBUTION:\n")
        category_counts = df['spi_category'].value_counts().sort_index()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{category}: {count:,} ({percentage:.1f}%)\n")
        f.write("\n")

        # Drought severity distribution
        f.write("DROUGHT SEVERITY DISTRIBUTION:\n")
        drought_counts = df['drought_severity'].value_counts().sort_index()
        for severity, count in drought_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"{severity}: {count:,} ({percentage:.1f}%)\n")


def create_summary_plots(df, output_dir):
    """Create summary visualization plots"""
    print("Creating summary plots...")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SPI Zonal Statistics Summary', fontsize=16, fontweight='bold')

    # 1. Distribution of mean SPI values
    axes[0, 0].hist(df['mean'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Normal (SPI=0)')
    axes[0, 0].axvline(-1, color='orange', linestyle='--', alpha=0.7, label='Moderate Drought')
    axes[0, 0].axvline(-2, color='darkred', linestyle='--', alpha=0.7, label='Severe Drought')
    axes[0, 0].set_xlabel('Mean SPI Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Mean SPI Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Time series of overall mean SPI
    monthly_mean = df.groupby(['year', 'month'])['mean'].mean().reset_index()
    monthly_mean['date'] = pd.to_datetime(monthly_mean[['year', 'month']].assign(day=1))

    axes[0, 1].plot(monthly_mean['date'], monthly_mean['mean'], linewidth=2, color='darkblue')
    axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(-1, color='orange', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(-2, color='darkred', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Mean SPI')
    axes[0, 1].set_title('Time Series of Regional Mean SPI')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. SPI category distribution
    category_counts = df['spi_category'].value_counts()
    colors = ['darkred', 'red', 'orange', 'yellow', 'lightblue', 'blue', 'darkblue', 'purple', 'darkmagenta']
    axes[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                   colors=colors[:len(category_counts)])
    axes[1, 0].set_title('SPI Category Distribution')

    # 4. Drought severity over time
    drought_monthly = df[df['drought_severity'] != 'No Drought'].groupby(['year', 'month']).size().reset_index(
        name='drought_count')
    if len(drought_monthly) > 0:
        drought_monthly['date'] = pd.to_datetime(drought_monthly[['year', 'month']].assign(day=1))
        axes[1, 1].plot(drought_monthly['date'], drought_monthly['drought_count'],
                        linewidth=2, color='darkred', marker='o', markersize=4)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Number of Areas in Drought')
        axes[1, 1].set_title('Drought Affected Areas Over Time')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No drought conditions detected',
                        transform=axes[1, 1].transAxes, ha='center', va='center')
        axes[1, 1].set_title('Drought Affected Areas Over Time')

    plt.tight_layout()

    plot_file = os.path.join(output_dir, 'spi_summary_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary plots saved to: {plot_file}")


def save_processing_info(nc_file, shp_file, times, shp_df, output_dir, results_df, missing_file=None):
    """Save detailed processing information"""
    info_file = os.path.join(output_dir, 'processing_info.txt')

    with open(info_file, 'w') as f:
        f.write("SPI ZONAL STATISTICS PROCESSING INFO\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input NetCDF file: {nc_file}\n")
        f.write(f"Input shapefile: {shp_file}\n")
        f.write(f"Time period processed: {times.min()} to {times.max()}\n")
        f.write(f"Number of time steps: {len(times)}\n")
        f.write(f"Number of administrative units: {len(shp_df)}\n")
        f.write(f"Total output records: {len(results_df):,}\n\n")

        f.write("DATA QUALITY NOTES:\n")
        f.write("- SPI data was cleaned to remove fill values (-9999) and extreme outliers\n")
        f.write("- Problematic equatorial boundary cells were removed\n")
        f.write("- SPI values were capped at ±4 for realistic drought/wet conditions\n")
        f.write("- This cleaning was necessary due to data quality issues in the original dataset\n\n")

        if missing_file and os.path.exists(missing_file):
            missing_df = pd.read_csv(missing_file)
            f.write("MISSING ZONAL STATISTICS (no data):\n")
            f.write(f"Total missing records: {len(missing_df)}\n")
            f.write("First 10 missing entries:\n")
            f.write(missing_df.head(10).to_string(index=False))
            f.write("\nFull list saved to: " + missing_file + "\n\n")
        else:
            f.write("No missing zonal statistics detected.\n\n")

        f.write("SPI INTERPRETATION:\n")
        f.write("SPI >= 2.0:  Extremely wet conditions\n")
        f.write("1.5 <= SPI < 2.0: Severely wet conditions\n")
        f.write("1.0 <= SPI < 1.5: Moderately wet conditions\n")
        f.write("0.5 <= SPI < 1.0: Mildly wet conditions\n")
        f.write("-0.5 < SPI < 0.5: Normal conditions\n")
        f.write("-1.0 < SPI <= -0.5: Mildly dry conditions\n")
        f.write("-1.5 < SPI <= -1.0: Moderately dry conditions\n")
        f.write("-2.0 < SPI <= -1.5: Severely dry conditions\n")
        f.write("SPI <= -2.0: Extremely dry conditions\n")


#  usage
if __name__ == "__main__":
    nc_file = 'spi3_cleaned.nc'
    shp_file = 'gha_admbnda_gss_20210308_shp/gha_admbnda_gss_20210308_SHP/gha_admbnda_adm2_gss_20210308.shp'  # <-- Use correct shapefile

    try:
        print("Starting SPI zonal statistics calculation...")
        results_df, output_dir = calculate_spi_zonal_statistics(nc_file, shp_file, start_year=2013)

        print(f"\nAnalysis completed successfully!")
        print(f"Check the output directory: {output_dir}")

        # Display sample results
        print(f"\nSample results (first 5 rows):")
        print(results_df.head())

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your input files and make sure they contain the expected data.")