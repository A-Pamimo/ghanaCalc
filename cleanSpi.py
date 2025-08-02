import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def clean_spi_data(file_path, output_path=None):
    """
    Clean SPI data by removing fill values and problematic extreme values

    Parameters:
    file_path (str): Path to input NetCDF file
    output_path (str): Path for cleaned output file (optional)

    Returns:
    xarray.Dataset: Cleaned dataset
    """

    print("Loading SPI data...")
    ds = xr.open_dataset(file_path)

    print("\n=== ORIGINAL DATA STATISTICS ===")
    print(f"Dataset dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

    # Original statistics
    total_points = ds['SPI3'].size
    valid_points_orig = (~np.isnan(ds['SPI3'])).sum().values

    print(f"\nOriginal data statistics:")
    print(f"  Total data points: {total_points:,}")
    print(f"  Valid data points: {valid_points_orig:,}")
    print(f"  Missing data points: {total_points - valid_points_orig:,}")
    print(f"  Missing data percentage: {((total_points - valid_points_orig) / total_points * 100):.2f}%")

    if valid_points_orig > 0:
        print(f"\nSPI3 value statistics (original):")
        print(f"  Min: {ds['SPI3'].min().values:.3f}")
        print(f"  Max: {ds['SPI3'].max().values:.3f}")
        print(f"  Mean: {ds['SPI3'].mean().values:.3f}")
        print(f"  Std: {ds['SPI3'].std().values:.3f}")

    print("\n=== CLEANING DATA ===")

    # Step 1: Remove obvious fill values
    print("Step 1: Removing fill values (-9999 and similar)")
    fill_values = [-9999.0, -999.0, -99.0]

    for fill_val in fill_values:
        n_fill = (ds['SPI3'] == fill_val).sum().values
        if n_fill > 0:
            print(f"  Found {n_fill} instances of fill value {fill_val}")
        ds['SPI3'] = ds['SPI3'].where(ds['SPI3'] != fill_val)

    # Also remove any values less than -10 (likely other fill values)
    extreme_low_fill = (ds['SPI3'] < -10).sum().values
    if extreme_low_fill > 0:
        print(f"  Found {extreme_low_fill} values < -10 (likely fill values)")
    ds['SPI3'] = ds['SPI3'].where(ds['SPI3'] >= -10)

    # Step 2: Identify extreme values
    print("\nStep 2: Identifying extreme values")
    extreme_high = (ds['SPI3'] > 6).sum().values
    extreme_low = (ds['SPI3'] < -6).sum().values

    print(f"  Values > 6: {extreme_high}")
    print(f"  Values < -6: {extreme_low}")

    # Step 3: Find locations of extreme values
    if extreme_high > 0 or extreme_low > 0:
        print("\nStep 3: Analyzing extreme value locations")
        extreme_coords = ds['SPI3'].where(np.abs(ds['SPI3']) > 6).stack(points=['lat', 'lon', 'time'])
        extreme_coords_clean = extreme_coords.dropna('points')

        print(f"Found {len(extreme_coords_clean)} extreme values")

        if len(extreme_coords_clean) > 0:
            print("Sample extreme values:")
            # Show first 5 extreme values
            for i in range(min(5, len(extreme_coords_clean))):
                val = extreme_coords_clean.isel(points=i).values
                lat = extreme_coords_clean.lat.isel(points=i).values
                lon = extreme_coords_clean.lon.isel(points=i).values
                time = extreme_coords_clean.time.isel(points=i).values
                print(f"  Value: {val:.3f}, Lat: {lat}, Lon: {lon}, Time: {str(time)[:10]}")

    # Step 4: Remove problematic equatorial grid cells (if they exist)
    print("\nStep 4: Checking for problematic equatorial boundary cells")
    equatorial_mask = (ds['lat'] == 0.0) & (ds['lon'] >= 42.0) & (ds['lon'] <= 43.0)
    problematic_cells = equatorial_mask.sum().values

    if problematic_cells > 0:
        print(f"  Found {problematic_cells} potentially problematic equatorial cells")
        print("  Removing these cells from analysis")
        ds['SPI3'] = ds['SPI3'].where(~equatorial_mask)

    # Step 5: Cap remaining extreme values at reasonable thresholds
    print("\nStep 5: Capping extreme values at ±4 threshold")
    values_capped_high = (ds['SPI3'] > 4).sum().values
    values_capped_low = (ds['SPI3'] < -4).sum().values

    if values_capped_high > 0:
        print(f"  Capping {values_capped_high} values > 4")
    if values_capped_low > 0:
        print(f"  Capping {values_capped_low} values < -4")

    ds['SPI3'] = ds['SPI3'].clip(-4, 4)

    print("\n=== CLEANED DATA STATISTICS ===")
    valid_points_clean = (~np.isnan(ds['SPI3'])).sum().values

    print(f"Cleaned data statistics:")
    print(f"  Total data points: {total_points:,}")
    print(f"  Valid data points: {valid_points_clean:,}")
    print(f"  Missing data points: {total_points - valid_points_clean:,}")
    print(f"  Missing data percentage: {((total_points - valid_points_clean) / total_points * 100):.2f}%")

    if valid_points_clean > 0:
        print(f"\nSPI3 value statistics (cleaned):")
        print(f"  Min: {ds['SPI3'].min().values:.3f}")
        print(f"  Max: {ds['SPI3'].max().values:.3f}")
        print(f"  Mean: {ds['SPI3'].mean().values:.3f}")
        print(f"  Std: {ds['SPI3'].std().values:.3f}")

    # Save cleaned data if output path provided
    if output_path:
        print(f"\nSaving cleaned data to: {output_path}")
        ds.to_netcdf(output_path)
        print("Data saved successfully!")

    return ds


def plot_spi_summary(ds, save_path=None):
    """
    Create summary plots of the cleaned SPI data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SPI3 Data Summary - Ethiopia', fontsize=16)

    # 1. Histogram of SPI values
    spi_values = ds['SPI3'].values.flatten()
    spi_values = spi_values[~np.isnan(spi_values)]

    axes[0, 0].hist(spi_values, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('SPI3 Values')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of SPI3 Values')
    axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Mean')
    axes[0, 0].legend()

    # 2. Time series of spatial mean
    spatial_mean = ds['SPI3'].mean(dim=['lat', 'lon'])
    axes[0, 1].plot(ds.time, spatial_mean, linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Spatial Mean SPI3')
    axes[0, 1].set_title('Time Series of Spatial Mean SPI3')
    axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Spatial pattern - mean over time
    temporal_mean = ds['SPI3'].mean(dim='time')
    im1 = axes[1, 0].contourf(ds.lon, ds.lat, temporal_mean, levels=20, cmap='RdBu_r')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    axes[1, 0].set_title('Temporal Mean SPI3')
    plt.colorbar(im1, ax=axes[1, 0])

    # 4. Data availability map
    data_availability = (~np.isnan(ds['SPI3'])).sum(dim='time') / len(ds.time) * 100
    im2 = axes[1, 1].contourf(ds.lon, ds.lat, data_availability, levels=20, cmap='viridis')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].set_title('Data Availability (%)')
    plt.colorbar(im2, ax=axes[1, 1])

    # Annotate "n/a" for cells with 0% availability (out of bounds or no data)
    lons, lats = np.meshgrid(ds.lon.values, ds.lat.values)
    na_mask = data_availability.values == 0
    for i in range(lats.shape[0]):
        for j in range(lons.shape[1]):
            if na_mask[i, j]:
                axes[1, 1].text(
                    ds.lon.values[j], ds.lat.values[i], "n/a",
                    color="red", fontsize=8, ha="center", va="center"
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plot saved to: {save_path}")

    plt.show()


# Main execution
if __name__ == "__main__":
    # USAGE INSTRUCTIONS:
    # 1. Replace 'your_spi_file.nc' with your actual file path
    # 2. Optionally specify output paths for cleaned data and plots
    # 3. Run the script

    # File paths (UPDATE THESE)
    input_file = "SPI3_GH_combined_2013_2017.nc"  # Replace with your file path
    output_file = "spi3_cleaned3_1.nc"  # Output file for cleaned data
    plot_file = "spi_summary_1.png"  # Output file for summary plots

    try:
        # Clean the data
        print("Starting SPI data cleaning process...")
        cleaned_ds = clean_spi_data(input_file, output_file)

        # Create summary plots
        print("\nCreating summary plots...")
        plot_spi_summary(cleaned_ds, plot_file)

        print("\n=== CLEANING COMPLETE ===")
        print("Your data is now ready for analysis!")
        print(f"Cleaned data saved as: {output_file}")
        print(f"Summary plots saved as: {plot_file}")

    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please update the 'input_file' variable with the correct path to your SPI data file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your file path and data format.")