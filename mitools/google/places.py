import math
from os import PathLike
from pathlib import Path
from typing import Iterable, List, NewType, Optional, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shapely
from shapely.geometry import Point, Polygon
from shapely.ops import transform

CircleType = NewType('CircleType', Polygon)

class CityGeojson:
    def __init__(self, geojson_path: PathLike):
        self.geojson_path = Path(geojson_path)
        self.data = gpd.read_file(geojson_path)
        self.polygons = self.data['geometry']
        self.merged_polygon = self.polygons.unary_union
        self.bounds = self.polygons.bounds.iloc[0].values
    
    def plot_unary_polygon(self):
        gpd.GeoSeries(self.merged_polygon).plot(facecolor='none',
                                edgecolor=sns.color_palette('Paired')[0])
        plt.show()

    def plot_polygons(self):
        gpd.GeoSeries(self.polygons).plot(facecolor='none',
                                edgecolor=sns.color_palette('Paired')[0])
        plt.show()

def meters_to_degree(distance_in_meters: float, reference_latitude:float) -> float:
    meters_per_degree_latitude = 111_132.95
    meters_per_degree_longitude = 111_132.95 * \
        math.cos(math.radians(reference_latitude))

    lat_degrees = distance_in_meters / meters_per_degree_latitude
    lon_degrees = distance_in_meters / meters_per_degree_longitude

    return max(lat_degrees, lon_degrees)

def sample_polygon_with_circle(polygon: Polygon, radius_in_meters: float, step_in_degrees: float) -> List[CircleType]:
    if not polygon.is_valid:
        raise ValueError('Invalid Polygon')
    minx, miny, maxx, maxy = polygon.bounds
    circles = []
    for lat in np.arange(miny, maxy, step_in_degrees):
        for lon in np.arange(minx, maxx, step_in_degrees):
            point = Point(lon, lat)
            deg_radius = meters_to_degree(distance_in_meters=radius_in_meters, reference_latitude=lat)
            circle = point.buffer(deg_radius)
            if polygon.contains(point) or polygon.intersects(circle):
                circles.append(circle)
    return circles

def sample_polygons_with_circles(polygons: Iterable[Polygon], radius_in_meters: float, step_in_degrees: float) -> List[CircleType]:
    circles = []
    for polygon in polygons:
        circles.extend(sample_polygon_with_circle(polygon=polygon, 
                                                  radius_in_meters=radius_in_meters, 
                                                  step_in_degrees=step_in_degrees))
    return circles

def polygons_folium_map(polygons: Union[Iterable[Polygon], Polygon], output_file_path: Optional[PathLike]=None) -> folium.Map:
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    reversed_polygons = [transform(lambda lat, lon: (
        lon, lat), polygon) for polygon in polygons]
    centroids = [polygon.centroid.coords[0] for polygon in reversed_polygons]
    lons, lats = zip(*centroids)
    centroid_lon = sum(lons) / len(lons)
    centroid_lat = sum(lats) / len(lats)
    centroid = (centroid_lon, centroid_lat)
    folium_map = folium.Map(location=centroid, zoom_start=15)
    for polygon in polygons:
        folium.GeoJson(
            polygon,
            style_function=lambda feature: {
                'fillColor': 'blue',  # Color the polygon
                'color': 'black',     # Color for the outline
                'weight': 2,          # Width of the outline
                'fillOpacity': 0.3,   # Opacity of the fill color
            }
        ).add_to(folium_map)
    if output_file_path:
        folium_map.save(output_file_path)
    return folium_map

def polygons_folium_map_with_pois(polygons: Union[Iterable[Polygon], Polygon], 
                                  pois: Iterable[Tuple[str, Point]], 
                                  output_file_path: Optional[PathLike]=None) -> folium.Map:
    folium_map = polygons_folium_map(polygons=polygons, output_file_path=None)
    for poi_name, poi_point in pois:
        folium.Marker(
            location=[poi_point.x, poi_point.y],
            popup=folium.Popup(poi_name, max_width=250),
            icon=folium.Icon(color='orange', icon='cutlery',
                             prefix='fa')
        ).add_to(folium_map)
    if output_file_path:
        folium_map.save(output_file_path)
    return folium_map


if __name__ == '__main__':
    
    dehli_geojson = 'delhi_1997-2012_district.json'
    dehli = CityGeojson(dehli_geojson)

    dehli.plot_unary_polygon()
    dehli.plot_polygons()

    polygons_folium_map(dehli.polygons, 'test.html')

