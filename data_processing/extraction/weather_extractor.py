import pathlib
import numpy as np
import xarray as xr

class NetCDFWeatherExtractor:
    def __init__(self, time_name):
        self.dataset_cache = {}
        self.grid_cache = {}
        self.time_name = time_name

    def get_nc_file_path(self, parameter, date):
        raise NotImplemented()
    
    def load_and_cache_dataset(self, nc_file_path):
        path = pathlib.Path(nc_file_path)
        if nc_file_path is None or not path.exists():
            return None

        cache_key = nc_file_path

        if cache_key not in self.dataset_cache:
            try:
                print(f"Loading dataset: {path.name}")
                ds = xr.open_dataset(nc_file_path)
                self.dataset_cache[cache_key] = ds

                lon = ds.longitude.values
                lat = ds.latitude.values

                self.grid_cache[cache_key] = {
                    'lat_len': len(lat),
                    'lon_len': len(lon),
                    'min_lon': lon[0],
                    'min_lat': lat[0],
                    'lat_delta': lat[1] - lat[0],
                    'lon_delta': lon[1] - lon[0]
                }

            except Exception as e:
                print(f"Error loading dataset {nc_file_path}: {e}")
                return None

        return self.dataset_cache[cache_key]
    
    def extract_batch_daytime_hours_mean_by_parameter(self, parameter, date, coords_array):
        nc_file_path = self.get_nc_file_path(parameter, date)
        return self.extract_batch_daytime_hours_mean(nc_file_path, date, coords_array, parameter)

    def extract_batch_daytime_hours_mean(self, nc_file_path, date, coords_array, param_name):
        ds = self.load_and_cache_dataset(nc_file_path)
        if ds is None:
            raise Exception(f"Could not find or load dataset for parameter {param_name} and date {date.strftime('%Y-%m-%d')}")

        try:
            date_str = date.strftime('%Y-%m-%d')
            start_time = f"{date_str}T09:00:00"
            end_time = f"{date_str}T14:59:59"

            ds_subset = ds.sel(**{self.time_name: slice(start_time, end_time)})

            if len(ds_subset[self.time_name]) == 0:
                return np.full(len(coords_array), np.nan)

            cache_key = nc_file_path
            grid_info = self.grid_cache[cache_key]

            results = []
            for lon, lat in coords_array:
                closest_i = int(np.round((lat - grid_info['min_lat']) / grid_info['lat_delta']))
                closest_j = int(np.round((lon - grid_info['min_lon']) / grid_info['lon_delta']))
                candidates = []
                for d_j, d_i in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj_i = closest_i + d_i
                    adj_j = closest_j + d_j
                    if adj_i < 0 or adj_i >= grid_info['lat_len'] or adj_j < 0 or adj_j >= grid_info['lon_len']:
                        continue
                    cand_lon = grid_info['min_lon'] + adj_j * grid_info['lon_delta']
                    cand_lat = grid_info['min_lat'] + adj_i * grid_info['lat_delta']
                    dist_sq = (lon - cand_lon) ** 2 + (lat - cand_lat) ** 2
                    candidates.append((dist_sq, adj_i, adj_j))
                candidates.sort()

                for k, (_, adj_i, adj_j) in enumerate(candidates):
                    mean_value = ds_subset[param_name].isel(
                        latitude=adj_i, longitude=adj_j
                    ).mean().item()

                    if not np.isnan(mean_value) or k == len(candidates) - 1:
                        results.append(mean_value)
                        break

            return np.array(results)

        except Exception as e:
            print(f"Error processing batch for {date_str}: {e}")
            return np.full(len(coords_array), np.nan)

    def cleanup_cache(self):
        for ds in self.dataset_cache.values():
            try:
                ds.close()
            except:
                pass
        self.dataset_cache.clear()
        self.grid_cache.clear()

class ERA5Extractor(NetCDFWeatherExtractor):
    def __init__(self, data_base_path):
        self.data_base_path = pathlib.Path(data_base_path)

        super().__init__("valid_time")
    
    def get_nc_file_path(self, parameter, date):
        year = date.strftime('%Y')

        if parameter == "swh":
            path = self.data_base_path / f"{year}_significant_height_of_combined_wind_waves_and_swell.nc"
        elif parameter == "u10":
            path = self.data_base_path / f"{year}_10m_u_component_of_wind.nc"
        elif parameter == "v10":
            path = self.data_base_path / f"{year}_10m_v_component_of_wind.nc"
        elif parameter == "mwd":
            path = self.data_base_path / f"{year}_mean_wave_direction.nc"
        elif parameter == "mwp":
            path = self.data_base_path / f"{year}_mean_wave_period.nc"
        elif parameter == "avg_tprate":
            path = self.data_base_path / f"{year}_mean_total_precipitation_rate.nc"
        else:
            raise NotImplemented()
        
        if not path.exists():
            print(path)
            return None
        
        return str(path)