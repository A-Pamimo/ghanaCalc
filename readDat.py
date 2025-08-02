import pandas as pd
import geopandas as gpd

dta_file = 'distFixed.dta'
df = pd.read_stata(dta_file)

shp_file = 'gha_admbnda_gss_20210308_shp/gha_admbnda_gss_20210308_SHP/gha_admbnda_adm2_gss_20210308.shp'
shp_df = gpd.read_file(shp_file)

print("First 10 values for each shapefile column:")
for col in shp_df.columns:
    print(f"{col}: {shp_df[col].head(10).tolist()}")