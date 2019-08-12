"""
High-level functions to generate Conventions-1.4 compliant netCDF files of 
projected datasets using xarray.

Focussed on writing information relating to data projection to the xarray
Dataset.

This is probably realistically a temporary solution - was going to sort out
into a proper generic library but geoxarray is now starting to lead on this,
just isn't yet mature enough to be used.

Andrew Tedstone, July/August 2019

"""

import xarray as xr
from osgeo import osr

import georaster

def create_grid_mapping(crs):

    srs = osr.SpatialReference()
    srs.ImportFromProj4(crs) 
    gm = xr.DataArray(0, encoding={'dtype': np.dtype('int8')})
    gm.attrs['projected_crs_name'] = srs.GetAttrValue('projcs')
    gm.attrs['grid_mapping_name'] = 'universal_transverse_mercator'
    gm.attrs['scale_factor_at_central_origin'] = srs.GetProjParm('scale_factor')
    gm.attrs['standard_parallel'] = srs.GetProjParm('latitude_of_origin')
    gm.attrs['straight_vertical_longitude_from_pole'] = srs.GetProjParm('central_meridian')
    gm.attrs['false_easting'] = srs.GetProjParm('false_easting')
    gm.attrs['false_northing'] = srs.GetProjParm('false_northing')
    gm.attrs['latitude_of_projection_origin'] = srs.GetProjParm('latitude_of_origin')

    return gm



def add_grid_mapping(gm,ds):
    """
    Add the grid mapping attribute to the Dataset.
    """
    raise NotImplementedError



def create_latlon_da(geotiff_fn, x_name, y_name, encoding='default'):

    gtiff = georaster.SingleBandRaster(geotiff_fn, load_data=False)
    lon, lat = gtiff.coordinates(latlon=True)
    gtiff = None

    coords_geo = {y_name: S2vals[y_name], x_name: S2vals[x_name]}

    if encoding == 'default':
        encoding = {'_FillValue': -9999., 
                    'dtype': 'int16', 
                    'scale_factor': 0.000000001}

    lon_array = xr.DataArray(lon, coords=coords_geo, dims=['y', 'x'],
                             encoding=encoding)
    lon_array.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
    lon_array.attrs['units'] = 'degrees'
    lon_array.attrs['standard_name'] = 'longitude'

    lat_array = xr.DataArray(lat, coords=coords_geo, dims=['y', 'x'],
                             encoding=encoding)
    lat_array.attrs['grid_mapping'] = proj_info.attrs['grid_mapping_name']
    lat_array.attrs['units'] = 'degrees'
    lat_array.attrs['standard_name'] = 'latitude'

    return (lon_array, lat_array)



def add_geo_info(ds, x_name, y_name, author, title):

	# add metadata for dataset
    ds.attrs['Conventions'] = 'CF-1.4'
    ds.attrs['author'] = netcdf_metadata['author']
    ds.attrs['title'] = netcdf_metadata['title']

    # Additional geo-referencing
    ds.attrs['nx'] = len(ds[x_name])
    ds.attrs['ny'] = len(ds[y_name])
    ds.attrs['xmin'] = float(ds[x_name].min())
    ds.attrs['ymax'] = float(ds[y_name].max())
    ds.attrs['spacing'] = ds[x_name].isel() #needs work

    # NC conventions metadata for dimensions variables
    ds[x_name].attrs['units'] = 'meters' # this needs to be setable rather than hard-coded
    ds[x_name].attrs['standard_name'] = 'projection_x_coordinate'
    ds[x_name].attrs['point_spacing'] = 'even'
    ds[x_name].attrs['axis'] = 'x'

    ds[y_name].attrs['units'] = 'meters'
    ds[y_name].attrs['standard_name'] = 'projection_y_coordinate'
    ds[y_name].attrs['point_spacing'] = 'even'
    ds[y_name].attrs['axis'] = 'y'

    return ds


