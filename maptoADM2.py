import geopandas as gpd
import pandas as pd
import xarray as xr
from rasterstats import zonal_stats

# 1. Load ADM2 shapefile and district list
adm2_gdf = gpd.read_file("gha_admbnda_adm2_gss_20210308.shp")
districts_df = pd.read_stata("distFixed.dta")

# Merge to keep only districts in the survey
adm2_gdf = adm2_gdf.merge(districts_df, left_on="ADM2_PCODE", right_on="ADM2_Pcode")

# 2. Load SPI3 NetCDF and extract July 2013 and July 2017
ds = xr.open_dataset("SPI3_GH_combined_2013_2017.nc")
# Find time indices for July 2013 and July 2017
july_2013 = ds.sel(time="2013-07")
july_2017 = ds.sel(time="2017-07")

# 3. Convert SPI3 slices to raster (2D numpy arrays)
for year, data in zip([2013, 2017], [july_2013, july_2017]):
    # Save temporary GeoTIFF for rasterstats (or use in-memory if advanced)
    data['SPI3'].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    data['SPI3'].rio.write_crs("EPSG:4326", inplace=True)
    tiff_path = f"spi3_{year}_07.tif"
    data['SPI3'].rio.to_raster(tiff_path)

    # 4. Zonal statistics: mean SPI3 per district
    stats = zonal_stats(
        adm2_gdf,
        tiff_path,
        stats=["mean"],
        geojson_out=True,
        nodata=None
    )
    # Attach results to GeoDataFrame
    adm2_gdf[f"SPI3_July_{year}"] = [feat['properties']['mean'] for feat in stats]

# 5. Output results
adm2_gdf[["ADM2_PCODE", "SPI3_July_2013", "SPI3_July_2017"]].to_csv("adm2_spi3_july_2013_2017.csv", index=False)
print("Saved zonal SPI3 means for July 2013 and 2017 to adm2_spi3_july_2013_2017.csv")