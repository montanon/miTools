import os
from os import PathLike
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple, Type, Union

import folium
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from tqdm import tqdm

from mitools.exceptions import ArgumentTypeError, ArgumentValueError

from .places_objects import (
    NewPlace,
    Place,
)

CircleType = NewType("CircleType", Polygon)

DPI = 500
WIDTH = 14
ASPECT_RATIO = 16 / 9
HEIGHT = WIDTH / ASPECT_RATIO
PLACE_CLASSES = Union[Place, NewPlace]


def polygons_folium_map(
    polygons: Union[Iterable[Polygon], Polygon],
    output_file_path: Optional[PathLike] = None,
) -> folium.Map:
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    reversed_polygons = [
        transform(lambda lat, lon: (lon, lat), polygon) for polygon in polygons
    ]
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
                "fillColor": "blue",  # Color the polygon
                "color": "black",  # Color for the outline
                "weight": 2,  # Width of the outline
                "fillOpacity": 0.3,  # Opacity of the fill color
            },
        ).add_to(folium_map)
    if output_file_path:
        folium_map.save(output_file_path)
    return folium_map


def polygons_folium_map_with_pois(
    polygons: Union[Iterable[Polygon], Polygon],
    pois: Iterable[Tuple[str, Point]],
    output_file_path: Optional[PathLike] = None,
) -> folium.Map:
    folium_map = polygons_folium_map(polygons=polygons, output_file_path=None)
    for poi_name, poi_point in pois:
        folium.Marker(
            location=[poi_point.x, poi_point.y],
            popup=folium.Popup(poi_name, max_width=250),
            icon=folium.Icon(color="orange", icon="cutlery", prefix="fa"),
        ).add_to(folium_map)
    if output_file_path:
        folium_map.save(output_file_path)
    return folium_map


def polygon_plot_with_sampling_circles(
    polygon: Polygon,
    circles: List[CircleType],
    point_of_interest: Optional[Point] = None,
    zoom_level: Optional[float] = 1.0,
    output_file_path: Optional[PathLike] = None,
) -> Axes:
    minx, miny, maxx, maxy = polygon.bounds
    out_circles, in_circles = [], []
    for circle in tqdm(circles):
        if polygon.contains(circle) or polygon.intersects(circle):
            in_circles.append(circle)
        else:
            out_circles.append(circle)
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
        facecolor=sns.color_palette("Paired")[0],
        edgecolor="none",
        alpha=0.5,
        figsize=(WIDTH, HEIGHT),
    )
    ax = polygon.plot(
        facecolor="none", edgecolor=sns.color_palette("Paired")[0], linewidth=3, ax=ax
    )
    point1 = (minx, miny)
    point2 = (maxx, maxy)
    rectangle = patches.Rectangle(
        (point1[0], point1[1]),  # (x,y)
        point2[0] - point1[0],  # width
        point2[1] - point1[1],  # height
        edgecolor="k",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rectangle)
    if out_circles:
        ax = gpd.GeoSeries(out_circles).plot(
            facecolor="none", edgecolor="r", ax=ax, alpha=0.5, label="Out Circles"
        )
        out_circle_proxy = mlines.Line2D(
            [],
            [],
            color="r",
            marker="o",
            markersize=10,
            label="Out Circles",
            linestyle="None",
        )
    ax = gpd.GeoSeries(in_circles).plot(
        facecolor="none", edgecolor="g", ax=ax, alpha=0.5, label="In Circles"
    )
    in_circle_proxy = mlines.Line2D(
        [],
        [],
        color="g",
        marker="o",
        markersize=10,
        label="In Circles",
        linestyle="None",
    )
    ax.legend(
        handles=[out_circle_proxy, in_circle_proxy]
        if out_circles
        else [in_circle_proxy],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
    )
    ax.set_title("Circles to Sample")
    ax.set_xticks([])
    ax.set_yticks([])
    if point_of_interest:
        ax.set_xlim(
            [
                point_of_interest.centroid.x - zoom_level,
                point_of_interest.centroid.x + zoom_level,
            ]
        )
        ax.set_ylim(
            [
                point_of_interest.centroid.y - zoom_level,
                point_of_interest.centroid.y + zoom_level,
            ]
        )
    if output_file_path:
        plt.savefig(output_file_path, dpi=DPI)
    return ax


def polygon_plot_with_circles_and_points(
    polygon,
    circles,
    points,
    point_of_interest=None,
    zoom_level=None,
    output_file_path=None,
):
    ax = polygon_plot_with_sampling_circles(
        polygon=polygon,
        circles=circles,
        point_of_interest=point_of_interest,
        zoom_level=zoom_level,
        output_file_path=output_file_path,
    )
    for point in points:
        ax.plot(point[0], point[1], "ro", markersize=0.25)
    if output_file_path:
        plt.savefig(output_file_path, dpi=DPI)
    return ax


def polygon_plot_with_points(
    polygon,
    points,
    point_of_interest: Optional[Point] = None,
    zoom_level: Optional[float] = 1.0,
    output_file_path=None,
):
    minx, miny, maxx, maxy = polygon.bounds
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
        facecolor=sns.color_palette("Paired")[0],
        edgecolor="none",
        alpha=0.5,
        figsize=(WIDTH, HEIGHT),
    )
    ax = polygon.plot(
        facecolor="none", edgecolor=sns.color_palette("Paired")[0], linewidth=3, ax=ax
    )
    point1 = (minx, miny)
    point2 = (maxx, maxy)
    rectangle = patches.Rectangle(
        (point1[0], point1[1]),  # (x,y)
        point2[0] - point1[0],  # width
        point2[1] - point1[1],  # height
        edgecolor="k",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(rectangle)
    for point in points:
        ax.plot(point[0], point[1], "ro", markersize=0.25)
    places_proxy = mlines.Line2D(
        [],
        [],
        color="g",
        marker="o",
        markersize=10,
        label="In Circles",
        linestyle="None",
    )
    ax.legend(
        handles=[places_proxy], loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2
    )
    ax.set_title("Sampled Points")
    ax.set_xticks([])
    ax.set_yticks([])
    if point_of_interest:
        ax.set_xlim(
            [
                point_of_interest.centroid.x - zoom_level,
                point_of_interest.centroid.x + zoom_level,
            ]
        )
        ax.set_ylim(
            [
                point_of_interest.centroid.y - zoom_level,
                point_of_interest.centroid.y + zoom_level,
            ]
        )
    if output_file_path:
        plt.savefig(output_file_path, dpi=DPI)
    return ax


if __name__ == "__main__":
    pass
