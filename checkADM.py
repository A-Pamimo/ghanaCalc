import geopandas as gpd

shp_path = r"Ghana district shape file 2017\Ghana_SHapefiles_2017.shp"
gdf = gpd.read_file(shp_path)
print(gdf.columns)
print(gdf.head())