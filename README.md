# ghanaCalc

Tools and data for processing the Standardized Precipitation Index (SPI) for Ghana. The project
focuses on combining monthly ERA5 SPI3 NetCDF files, cleaning the resulting dataset, and
aggregating values to administrative districts.

## Repository structure

- `ghN/` and `ghana_spei3/` – folders of monthly SPI3 and SPEI3 NetCDF files.
- `SPI3_GH_combined_2013_2017.nc` – example combined SPI3 dataset.
- `cleanSpi.py` – remove fill values, cap extreme SPI3 values and produce summary plots.
- `ZStatSPI3Era5.py` – compute zonal statistics for each ADM2 district using cleaned SPI data.
- `spi merger.py` – merge individual monthly NetCDF files into a single time series.
- `maptoADM2.py` – example of extracting SPI3 for July 2013 & July 2017 and mapping to districts.
- `readDat.py` & `checkADM.py` – small utilities for inspecting shapefiles and district tables.
- `gha_adm2/` and `gha_admbnda_gss_20210308_shp/` – Ghana administrative boundary shapefiles.
- `distFixed.dta` – district code table used for merging survey data with shapefiles.

## Requirements

Python 3 with the following packages:
`xarray`, `geopandas`, `pandas`, `numpy`, `rasterio`, `rasterstats`, `matplotlib`, `seaborn`.

A conda environment is recommended:

```bash
conda create -n ghanaCalc python=3.10 geopandas xarray rasterio rasterstats matplotlib seaborn pandas
conda activate ghanaCalc
```

## Workflow

1. **Combine monthly SPI files**
   ```bash
   python "spi merger.py"
   ```
   Adjust the `input_dir` variable inside the script to point to the directory with monthly NetCDF
   files. The output is `SPI3_GH_combined_2013_2017.nc`.

2. **Clean the combined dataset and generate plots**
   ```bash
   python cleanSpi.py
   ```
   Update `input_file`, `output_file` and `plot_file` at the bottom of the script as needed. The
   script removes fill values, caps extremes at ±4, and produces a summary graphic.

3. **Calculate zonal statistics for districts**
   ```bash
   python ZStatSPI3Era5.py
   ```
   Set `nc_file` to your cleaned SPI NetCDF and `shp_file` to a Ghana ADM2 shapefile. Results and
   plots are written to a timestamped folder in `spi_zonal_statistics_output/`.

4. **Map SPI values to ADM2 for specific months**
   ```bash
   python maptoADM2.py
   ```
   This example extracts July 2013 and July 2017 SPI3 values and writes
   `adm2_spi3_july_2013_2017.csv`.

## Data

The repository includes Ghana administrative boundaries from geoBoundaries and a sample cleaned
SPI3 NetCDF (`spi3_cleaned.nc`). Raw monthly SPI/SPEI files are stored in `ghN/` and
`ghana_spei3/`.

## License

No license specified.
