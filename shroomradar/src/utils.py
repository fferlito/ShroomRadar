import xarray as xr
import rioxarray

def nc_to_geotiff(nc_file: str, tif_file: str) -> None:
    """
    Converts a NetCDF file to a GeoTIFF file.

    This function opens a NetCDF dataset, selects the first data variable,
    and if a time dimension exists, it selects the first time step.
    It then sets the spatial dimensions and Coordinate Reference System (CRS)
    before exporting the data to a GeoTIFF raster file.

    Args:
        nc_file (str): The path to the input NetCDF file.
        tif_file (str): The path for the output GeoTIFF file.
    """
    # Open the NetCDF dataset
    ds = xr.open_dataset(nc_file)

    # Select the first data variable from the dataset
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    # If a 'time' dimension is present, select the first time step
    if "time" in da.dims:
        da = da.isel(time=0)

    # Set the spatial dimensions for rioxarray
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # Assign the Coordinate Reference System (CRS)
    da = da.rio.write_crs("EPSG:4326")

    # Export the DataArray to a GeoTIFF file
    da.rio.to_raster(tif_file)

    print(f"✅ Saved GeoTIFF: {tif_file}")
