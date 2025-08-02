import xarray as xr
import glob
import os
from pathlib import Path
import pandas as pd


def combine_spi3_files(input_directory, output_filename):
    """
    Combine multiple monthly SPI3 NetCDF files into a single file.

    Parameters:
    input_directory (str): Directory containing the individual NetCDF files
    output_filename (str): Name for the combined output file
    """

    # Find all NetCDF files in the directory
    nc_files = glob.glob(os.path.join(input_directory, "*.nc"))

    if not nc_files:
        print(f"No NetCDF files found in {input_directory}")
        return

    print(f"Found {len(nc_files)} NetCDF files")

    # Sort files by date (extracting YYYYMM from filename)
    def extract_date_from_filename(filename):
        """Extract YYYYMM from filename like 'SPI3_gamma_global_era5_moda_ref1991to2020_202203.area-subset.15.48.0.32.nc'"""
        basename = os.path.basename(filename)
        # Look for 6-digit date pattern (YYYYMM)
        import re
        date_match = re.search(r'(\d{6})', basename)
        if date_match:
            return date_match.group(1)
        return None

    # Create list of (date, filename) tuples and sort by date
    file_date_pairs = []
    for file in nc_files:
        date_str = extract_date_from_filename(file)
        if date_str and len(date_str) == 6:
            # Filter for 2018-2023 range
            year = int(date_str[:4])
            if 2013 <= year <= 2017:
                file_date_pairs.append((date_str, file))

    # Sort by date
    file_date_pairs.sort(key=lambda x: x[0])
    sorted_files = [pair[1] for pair in file_date_pairs]

    print(f"Processing {len(sorted_files)} files from 2018-2023")

    # Open all datasets
    datasets = []
    for i, file in enumerate(sorted_files):
        try:
            ds = xr.open_dataset(file)
            print(f"Processing file {i + 1}/{len(sorted_files)}: {os.path.basename(file)}")
            datasets.append(ds)
        except Exception as e:
            print(f"Error opening {file}: {e}")
            continue

    if not datasets:
        print("No valid datasets found")
        return

    # Combine along time dimension
    print("Combining datasets along time dimension...")
    combined_ds = xr.concat(datasets, dim='time')

    # Sort by time to ensure proper chronological order
    combined_ds = combined_ds.sortby('time')

    # Update global attributes
    combined_ds.attrs.update({
        'title': 'SPI3 Monthly Time Series',
        'description': 'Combined monthly Standardized Drought Index (SPI3) data from 2018-2023',
        'history': f'Combined from {len(datasets)} monthly files using xarray.concat',
        'time_coverage_start': str(combined_ds.time.min().values),
        'time_coverage_end': str(combined_ds.time.max().values),
        'number_of_time_steps': len(combined_ds.time)
    })

    # Save the combined dataset
    print(f"Saving combined dataset to {output_filename}...")
    combined_ds.to_netcdf(output_filename)

    # Print summary information
    print(f"\nCombined dataset summary:")
    print(f"Shape: {combined_ds.SPI3.shape}")
    print(f"Time range: {combined_ds.time.min().values} to {combined_ds.time.max().values}")
    print(f"Number of time steps: {len(combined_ds.time)}")
    print(f"Spatial dimensions: {len(combined_ds.lat)} lat x {len(combined_ds.lon)} lon")

    # Close all datasets to free memory
    for ds in datasets:
        ds.close()
    combined_ds.close()

    print(f"\nSuccessfully created combined file: {output_filename}")


def verify_combined_file(filename):
    """
    Verify the combined NetCDF file and print summary statistics.
    """
    print(f"\nVerifying combined file: {filename}")

    with xr.open_dataset(filename) as ds:
        print(f"Dataset dimensions: {dict(ds.dims)}")
        print(f"Variables: {list(ds.data_vars)}")
        print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")

        # Check for any missing data
        spi_data = ds.SPI3
        total_points = spi_data.size
        valid_points = spi_data.count().values
        missing_points = total_points - valid_points

        print(f"Data statistics:")
        print(f"  Total data points: {total_points:,}")
        print(f"  Valid data points: {valid_points:,}")
        print(f"  Missing data points: {missing_points:,}")
        print(f"  Missing data percentage: {(missing_points / total_points) * 100:.2f}%")

        # Print some basic statistics
        print(f"SPI3 value statistics:")
        print(f"  Min: {float(spi_data.min()):.3f}")
        print(f"  Max: {float(spi_data.max()):.3f}")
        print(f"  Mean: {float(spi_data.mean()):.3f}")
        print(f"  Std: {float(spi_data.std()):.3f}")


# Example usage
if __name__ == "__main__":
    # Set your input directory and output filename
    input_dir = r"C:\Users\oluwa\PycharmProjects\ghana\ghN"  # Update this path
    output_file = "SPI3_GH_combined_2013_2017.nc"

    # Combine the files
    combine_spi3_files(input_dir, output_file)

    # Verify the result
    if os.path.exists(output_file):
        verify_combined_file(output_file)