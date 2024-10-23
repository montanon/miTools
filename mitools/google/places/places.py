import math
import os
import random
import time
from datetime import datetime
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Iterable, List, NewType, Optional, Tuple, Union

import folium
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.axes import Axes
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import transform
from tqdm import tqdm

from mitools.exceptions import ArgumentTypeError, ArgumentValueError

from .places_objects import (
    CityGeojson,
    DummyResponse,
    NewNearbySearchRequest,
    NewPlace,
    intersection_condition_factory,
)

CircleType = NewType("CircleType", Polygon)

# https://mapsplatform.google.com/pricing/#pricing-grid
# https://developers.google.com/maps/documentation/places/web-service/search-nearby
# https://developers.google.com/maps/documentation/places/web-service/usage-and-billing#nearby-search
# https://developers.google.com/maps/documentation/places/web-service/nearby-search#fieldmask
# https://developers.google.com/maps/documentation/places/web-service/search-nearby#PlaceSearchPaging
# https://developers.google.com/maps/documentation/places/web-service/place-types
NEW_NEARBY_SEARCH_URL = "https://places.googleapis.com/v1/places:searchNearby"
NEARBY_SEARCH_URL = (
    "https://maps.googleapis.com/maps/api/place/nearbysearch/json?parameters"
)
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY")
RESTAURANT_TYPES = [
    "american_restaurant",
    "bakery",
    "bar",
    "barbecue_restaurant",
    "brazilian_restaurant",
    "breakfast_restaurant",
    "brunch_restaurant",
    "cafe",
    "chinese_restaurant",
    "coffee_shop",
    "fast_food_restaurant",
    "french_restaurant",
    "greek_restaurant",
    "hamburger_restaurant",
    "ice_cream_shop",
    "indian_restaurant",
    "indonesian_restaurant",
    "italian_restaurant",
    "japanese_restaurant",
    "korean_restaurant",
    "lebanese_restaurant",
    "meal_delivery",
    "meal_takeaway",
    "mediterranean_restaurant",
    "mexican_restaurant",
    "middle_eastern_restaurant",
    "pizza_restaurant",
    "ramen_restaurant",
    "restaurant",
    "sandwich_shop",
    "seafood_restaurant",
    "spanish_restaurant",
    "steak_house",
    "sushi_restaurant",
    "thai_restaurant",
    "turkish_restaurant",
    "vegan_restaurant",
    "vegetarian_restaurant",
    "vietnamese_restaurant",
]
FIELD_MASK = (
    "places.accessibilityOptions,places.addressComponents,places.adrFormatAddress,places.businessStatus,"
    + "places.displayName,places.formattedAddress,places.googleMapsUri,places.iconBackgroundColor,"
    + "places.iconMaskBaseUri,places.id,places.location,places.name,places.primaryType,places.primaryTypeDisplayName,places.plusCode,"
    + "places.shortFormattedAddress,places.subDestinations,places.types,places.utcOffsetMinutes,places.viewport,"
    + "places.currentOpeningHours,places.currentSecondaryOpeningHours,places.internationalPhoneNumber,places.nationalPhoneNumber,"
    + "places.priceLevel,places.rating,places.regularOpeningHours,places.regularSecondaryOpeningHours,places.userRatingCount,places.websiteUri"
)
QUERY_HEADERS = {
    "Content-Type": "aplication/json",
    "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
    "X-Goog-FieldMask": FIELD_MASK,
}
DPI = 500
WIDTH = 14
ASPECT_RATIO = 16 / 9
HEIGHT = WIDTH / ASPECT_RATIO


def meters_to_degree(distance_in_meters: float, reference_latitude: float) -> float:
    if not isinstance(distance_in_meters, (int, float)) or distance_in_meters < 0:
        raise ArgumentValueError("Invalid Distance, must be a positive number")
    if reference_latitude >= 90 or reference_latitude <= -90:
        raise ArgumentValueError("Invalid Latitude, must be between -90° and 90°")
    meters_per_degree_latitude = 111_132.95
    meters_per_degree_longitude = 111_132.95 * math.cos(
        math.radians(reference_latitude)
    )
    lat_degrees = distance_in_meters / meters_per_degree_latitude
    lon_degrees = distance_in_meters / meters_per_degree_longitude
    return max(lat_degrees, lon_degrees)


def sample_polygon_with_circles(
    polygon: Polygon,
    radius_in_meters: float,
    step_in_degrees: float,
    condition_rule: str = "center",
) -> List[CircleType]:
    if not isinstance(polygon, Polygon):
        raise ArgumentTypeError("Invalid 'polygon' is not of type Polygon.")
    if not polygon.is_valid:
        raise ArgumentValueError("Invalid Polygon")
    if polygon.is_empty:
        raise ArgumentValueError("Empty Polygon")
    if step_in_degrees <= 0:
        raise ArgumentValueError("Invalid Step, must be a positive number")
    condition = intersection_condition_factory(condition_rule)
    minx, miny, maxx, maxy = polygon.bounds
    latitudes = np.arange(miny, maxy, step_in_degrees)
    longitudes = np.arange(minx, maxx, step_in_degrees)
    circles = []
    for lat, lon in product(latitudes, longitudes):
        deg_radius = meters_to_degree(
            distance_in_meters=radius_in_meters, reference_latitude=lat
        )
        circle = Point(lon, lat).buffer(deg_radius)
        if condition.check(polygon=polygon, circle=circle):
            circles.append(circle)
    return circles


def sample_polygons_with_circles(
    polygons: Union[Iterable[Polygon], Polygon],
    radius_in_meters: float,
    step_in_degrees: float,
    condition_rule: Optional[str] = "center",
) -> List[CircleType]:
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    elif isinstance(polygons, MultiPolygon):
        polygons = list(polygons.geoms)
    else:
        raise ArgumentTypeError(
            "Invalid 'polygons' is not of type Polygon or MultiPolygon."
        )
    circles = []
    for polygon in polygons:
        circles.extend(
            sample_polygon_with_circles(
                polygon=polygon,
                radius_in_meters=radius_in_meters,
                step_in_degrees=step_in_degrees,
                condition_rule=condition_rule,
            )
        )
    return circles


def get_circles_search(
    circles_path,
    polygon,
    radius_in_meters,
    step_in_degrees,
    condition_rule="center",
    recalculate=False,
):
    if not circles_path.exists() or recalculate:
        circles = sample_polygons_with_circles(
            polygons=polygon,
            radius_in_meters=radius_in_meters,
            step_in_degrees=step_in_degrees,
            condition_rule=condition_rule,
        )
        circles = gpd.GeoDataFrame(geometry=circles).reset_index(drop=True)
        circles["searched"] = False
        circles.to_file(circles_path, driver="GeoJSON")
    else:
        circles = gpd.read_file(circles_path)
    return circles


def create_subsampled_circles(
    large_circle_center: Point,
    large_radius: float,
    small_radius: float,
    radial_samples: int,
    factor: float = 1.0,
) -> List[Polygon]:
    if large_radius <= 0 or small_radius <= 0:
        raise ValueError("Radius values must be positive.")
    if radial_samples <= 0:
        raise ValueError("radial_samples must be a positive integer.")
    large_radius_deg = meters_to_degree(large_radius, large_circle_center.y)
    small_radius_deg = meters_to_degree(small_radius, large_circle_center.y)
    large_circle = large_circle_center.buffer(large_radius_deg)
    subsampled_circles = [large_circle_center.buffer(small_radius_deg)]
    angle_step = 2 * np.pi / radial_samples
    for i in range(radial_samples):
        angle = i * angle_step
        dx = factor * small_radius_deg * np.cos(angle)
        dy = factor * small_radius_deg * np.sin(angle)
        new_center = Point(large_circle_center.x + dx, large_circle_center.y + dy)
        if large_circle.contains(new_center):
            subsampled_circles.append(new_center.buffer(small_radius_deg))
    return subsampled_circles


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


def read_or_initialize_places(file_path, recalculate=False):
    if file_path.exists() and not recalculate:
        return pd.read_parquet(file_path)
    else:
        return pd.DataFrame(columns=["circle", *list(NewPlace.__annotations__.keys())])


def generate_unique_place_id():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def create_dummy_place(query):
    latitude = query["locationRestriction"]["circle"]["center"]["latitude"]
    longitude = query["locationRestriction"]["circle"]["center"]["longitude"]
    radius = query["locationRestriction"]["circle"]["radius"]
    distance_in_deg = meters_to_degree(radius, latitude)
    random_types = random.sample(
        RESTAURANT_TYPES,
        random.randint(1, min(len(RESTAURANT_TYPES), random.randint(1, 5))),
    )
    unique_id = generate_unique_place_id()
    random_latitude = random.uniform(
        latitude - distance_in_deg, latitude + distance_in_deg
    )
    random_longitude = random.uniform(
        longitude - distance_in_deg, longitude + distance_in_deg
    )
    place_json = {
        "id": unique_id,
        "types": random_types,
        "location": {
            "latitude": random_latitude,
            "longitude": random_longitude,
        },
        "displayName": {"text": f"Name {unique_id}"},
        "primaryType": random.choice(random_types),
    }
    return place_json


def create_dummy_response(query):
    dummy_response = DummyResponse()
    has_places = random.choice([True, False, False])
    if has_places:
        places_n = random.randint(1, 21)
        dummy_response["places"] = [create_dummy_place(query) for _ in range(places_n)]
    return dummy_response


def nearby_search_request(
    circle, radius_in_meters, query_headers=None, restaurants=False
):
    query = NewNearbySearchRequest(
        circle.geometry,
        distance_in_meters=radius_in_meters,
        included_types=RESTAURANT_TYPES if restaurants else None,
    ).json_query()
    if query_headers is not None:
        QUERY_HEADERS = query_headers
    if QUERY_HEADERS["X-Goog-Api-Key"] != "":
        return requests.post(NEW_NEARBY_SEARCH_URL, headers=QUERY_HEADERS, json=query)
    else:
        return create_dummy_response(query)


def get_response_places(response_id, response):
    for n, place in enumerate(response.json()["places"]):
        place_series = NewPlace.from_json(place).to_series()
        place_series["circle"] = response_id
        if n == 0:
            places_df = pd.DataFrame(place_series).T
        else:
            places_df = pd.concat(
                [places_df, pd.DataFrame(place_series).T], axis=0, ignore_index=True
            )
    return places_df


def search_and_update_places(
    circle, radius_in_meters, response_id, query_headers=None, restaurants=False
):
    response = nearby_search_request(
        circle, radius_in_meters, query_headers=query_headers, restaurants=restaurants
    )
    places_df = None
    if response.reason == "OK":
        if "places" in response.json():
            places_df = get_response_places(response_id, response)
        searched = True
    else:
        print(response.status_code, response.reason, response.text)
        searched = False
        time.sleep(30)
    return searched, places_df


def process_circles(
    circles,
    radius_in_meters,
    file_path,
    circles_path,
    global_requests_counter=None,
    global_requests_counter_limit=None,
    query_headers=None,
    restaurants=False,
    recalculate=False,
):
    if global_requests_counter is None and global_requests_counter_limit is None:
        global GLOBAL_REQUESTS_COUNTER, GLOBAL_REQUESTS_COUNTER_LIMIT
    else:
        GLOBAL_REQUESTS_COUNTER = global_requests_counter[0]
        GLOBAL_REQUESTS_COUNTER_LIMIT = global_requests_counter_limit[0]
    if (
        (~circles["searched"]).any() or recalculate
    ) and GLOBAL_REQUESTS_COUNTER <= GLOBAL_REQUESTS_COUNTER_LIMIT:
        circles_search = circles[~circles["searched"]]
        found_places = read_or_initialize_places(file_path, recalculate)
        with tqdm(total=len(circles_search), desc="Processing circles") as pbar:
            for response_id, circle in circles_search.iterrows():
                searched, places_df = search_and_update_places(
                    circle,
                    radius_in_meters,
                    response_id,
                    query_headers=query_headers,
                    restaurants=restaurants,
                )
                if places_df is not None:
                    found_places = pd.concat(
                        [found_places, places_df], axis=0, ignore_index=True
                    )
                GLOBAL_REQUESTS_COUNTER += 1
                update_progress_and_save(
                    searched,
                    circles,
                    response_id,
                    found_places,
                    file_path,
                    circles_path,
                    pbar,
                )
                global_requests_counter[0] = GLOBAL_REQUESTS_COUNTER
                if GLOBAL_REQUESTS_COUNTER >= GLOBAL_REQUESTS_COUNTER_LIMIT:
                    break
    else:
        found_places = pd.read_parquet(file_path)
    return found_places


def update_progress_and_save(
    searched, circles, index, found_places, file_path, circles_path, pbar
):
    circles.loc[index, "searched"] = searched
    if (
        (index % 200 == 0)
        or (index == circles.shape[0] - 1)
        or (GLOBAL_REQUESTS_COUNTER >= GLOBAL_REQUESTS_COUNTER_LIMIT - 1)
    ):
        found_places.to_parquet(file_path)
        circles.to_file(circles_path, driver="GeoJSON")
    pbar.update()
    pbar.set_postfix(
        {
            "Remaining Circles": circles["searched"].value_counts()[False]
            if False in circles["searched"].value_counts()
            else 0,
            "Found Places": found_places["id"].nunique(),
            "Searched Circles": circles["searched"].sum(),
        }
    )


def search_places_in_polygon(
    root_folder,
    plot_folder,
    tag,
    polygon,
    radius_in_meters,
    step_in_degrees,
    condition_rule,
    global_requests_counter=None,
    global_requests_counter_limit=None,
    query_headers=None,
    restaurants=False,
    recalculate=False,
    show=False,
):
    circles_path = root_folder / Path(
        f"{tag}_{radius_in_meters}_radius_{step_in_degrees}_step_circles.geojson"
    )
    places_path = root_folder / Path(
        f"{tag}_{radius_in_meters}_radius_{step_in_degrees}_step_places.parquet"
    )
    polygon_with_circles_plot_path = plot_folder / Path(
        f"{tag}_polygon_with_circles_plot.png"
    )
    polygon_with_circles_zoom_plot_path = plot_folder / Path(
        f"{tag}_polygon_with_circles_zoom_plot.png"
    )
    polygon_with_circles_and_points_plot_path = plot_folder / Path(
        f"{tag}_polygon_with_circles_and_places_plot.png"
    )
    polygon_with_circles_and_points_zoom_plot_path = plot_folder / Path(
        f"{tag}_polygon_with_circles_and_places_zoom_plot.png"
    )
    circles = get_circles_search(
        circles_path,
        polygon,
        radius_in_meters,
        step_in_degrees,
        condition_rule=condition_rule,
        recalculate=recalculate,
    )
    if show or recalculate:
        _ = polygon_plot_with_sampling_circles(
            polygon=polygon,
            circles=circles.geometry.tolist(),
            output_file_path=polygon_with_circles_plot_path,
        )
        if show:
            plt.show()
        random_circle = random.choice(circles.geometry.tolist())
        _ = polygon_plot_with_sampling_circles(
            polygon=polygon,
            circles=circles.geometry.tolist(),
            point_of_interest=random_circle,
            zoom_level=5 * meters_to_degree(radius_in_meters, random_circle.centroid.y),
            output_file_path=polygon_with_circles_zoom_plot_path,
        )
        if show:
            plt.show()
    found_places = process_circles(
        circles,
        radius_in_meters,
        places_path,
        circles_path,
        global_requests_counter=global_requests_counter,
        global_requests_counter_limit=global_requests_counter_limit,
        query_headers=query_headers,
        restaurants=restaurants,
        recalculate=recalculate,
    )
    if show or recalculate:
        _ = polygon_plot_with_circles_and_points(
            polygon=polygon,
            circles=circles.geometry.tolist(),
            points=found_places[["longitude", "latitude"]].values.tolist(),
            output_file_path=polygon_with_circles_and_points_plot_path,
        )
        if show:
            plt.show()
        random_circle = random.choice(circles.geometry.tolist())
        _ = polygon_plot_with_circles_and_points(
            polygon=polygon,
            circles=circles.geometry.tolist(),
            point_of_interest=random_circle,
            zoom_level=5 * meters_to_degree(radius_in_meters, random_circle.centroid.y),
            points=found_places[["longitude", "latitude"]].values.tolist(),
            output_file_path=polygon_with_circles_and_points_zoom_plot_path,
        )
        if show:
            plt.show()
    return circles, found_places


def get_saturated_circles(
    polygon, found_places, circles, threshold, show=False, output_file_path=None
):
    places_by_circle = (
        found_places.groupby("circle")["id"].nunique().sort_values(ascending=False)
    )
    saturated_circles = places_by_circle[places_by_circle >= threshold].index
    saturated_circles = circles.loc[saturated_circles, :]
    _ = polygon_plot_with_circles_and_points(
        polygon=polygon,
        circles=saturated_circles.geometry.tolist(),
        points=found_places.loc[
            found_places["circle"].isin(saturated_circles.index),
            ["longitude", "latitude"],
        ].values.tolist(),
        output_file_path=output_file_path,
    )
    if show:
        plt.show()
    return saturated_circles


def get_saturated_area(polygon, saturated_circles, show=False, output_path=None):
    saturated_area = saturated_circles.geometry.unary_union
    polygon = gpd.GeoSeries(polygon)
    ax = polygon.plot(
        facecolor=sns.color_palette("Paired")[0],
        edgecolor="none",
        alpha=0.5,
        figsize=(WIDTH, HEIGHT),
    )
    gpd.GeoSeries(saturated_area).plot(
        ax=ax, facecolor="none", edgecolor=sns.color_palette("Paired")[0]
    )
    ax.set_title("Saturated Sampled Areas")
    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")
    if output_path:
        plt.savefig(output_path, dpi=DPI)
    if show:
        plt.show()
    return saturated_area


def places_search_step(
    project_folder,
    plots_folder,
    tag,
    polygon,
    radius_in_meters,
    step_in_degrees,
    global_requests_counter=None,
    global_requests_counter_limit=None,
    query_headers=None,
    restaurants=False,
    show=False,
    recalculate=False,
):
    circles, found_places = search_places_in_polygon(
        project_folder,
        plots_folder,
        tag,
        polygon,
        radius_in_meters,
        step_in_degrees,
        condition_rule="center",
        global_requests_counter=global_requests_counter,
        global_requests_counter_limit=global_requests_counter_limit,
        query_headers=query_headers,
        restaurants=restaurants,
        recalculate=recalculate,
        show=show,
    )
    saturated_circles_plot_path = plots_folder / f"{tag}_saturated_circles_plot.png"
    saturated_area_plot_path = plots_folder / f"{tag}_saturated_area_plot.png"
    saturated_circles = get_saturated_circles(
        polygon,
        found_places,
        circles,
        threshold=20,
        show=show,
        output_file_path=saturated_circles_plot_path,
    )
    saturated_area = get_saturated_area(
        polygon, saturated_circles, show=show, output_path=saturated_area_plot_path
    )
    plt.close("all")

    return found_places, circles, saturated_area, saturated_circles


def calculate_degree_steps(meter_radiuses, step_in_degrees=0.00375):
    degree_steps = []
    for i, radius in enumerate(meter_radiuses):
        if i == 0:
            step = step_in_degrees
        else:
            step *= radius / meter_radiuses[i - 1]
        degree_steps.append(step)
    return degree_steps


if __name__ == "__main__":
    cities_geojsons = {
        "delhi": "/Users/sebastian/Desktop/MontagnaInc/Projects/India_shapefiles/city/delhi/district/delhi_1997-2012_district.json",
        "tokyo": "/Users/sebastian/Desktop/MontagnaInc/Research/Cities_Restaurants/translated_tokyo_wards.geojson",
    }

    PROJECT_FOLDER = Path(
        "/Users/sebastian/Desktop/MontagnaInc/Research/Cities_Restaurants/Tokyo_Places_with_Price"
    )
    PROJECT_FOLDER.mkdir(exist_ok=True)
    PLOTS_FOLDER = PROJECT_FOLDER / "plots"
    PLOTS_FOLDER.mkdir(exist_ok=True)
    CITY = "tokyo"
    SHOW = True
    RECALCULATE = False

    city = CityGeojson(cities_geojsons[CITY], CITY)
    city_wards_plot_path = PLOTS_FOLDER / f"{city.name}_wards_polygons_plot.png"
    city_plot_path = PLOTS_FOLDER / f"{city.name}_polygon_plot.png"
    if SHOW or False:
        ax = city.plot_polygons()
        if not city_wards_plot_path.exists() or RECALCULATE:
            ax.get_figure().savefig(city_wards_plot_path, dpi=DPI)
        plt.show()
        ax = city.plot_unary_polygon()
        if not city_plot_path.exists() or RECALCULATE:
            ax.get_figure().savefig(city_plot_path, dpi=DPI)
        plt.show()

    STEP_IN_DEGREES = 0.00375
    meter_radiuses = [250, 100, 50, 25, 12.5, 5, 2.5, 1]
    degree_steps = calculate_degree_steps(
        meter_radiuses, step_in_degrees=STEP_IN_DEGREES
    )

    area_polygon = city.merged_polygon

    all_places_parquet_path = PROJECT_FOLDER / f"{city.name}_all_found_places.parquet"
    all_places_excel_path = PROJECT_FOLDER / f"{city.name}_all_found_places.xlsx"
    unique_places_parquet_path = (
        PROJECT_FOLDER / f"{city.name}_unique_found_places.parquet"
    )
    unique_places_excel_path = PROJECT_FOLDER / f"{city.name}_unique_found_places.xlsx"
    all_places = pd.DataFrame(
        columns=["circle", *list(NewPlace.__annotations__.keys())]
    )
    total_sampled_circles = 0
    for i, (radius, step) in enumerate(zip(meter_radiuses, degree_steps)):
        TAG = f"Step-{i+1}_{city.name}"
        print(TAG)
        found_places, circles, area_polygon, saturated_circles = places_search_step(
            PROJECT_FOLDER,
            PLOTS_FOLDER,
            TAG,
            area_polygon,
            radius,
            step,
            global_requests_counter=None,
            global_requests_counter_limit=None,
            restaurants=True,
            show=SHOW,
            recalculate=RECALCULATE,
        )
        sampled_circles = circles.shape[0]
        total_sampled_circles += sampled_circles
        print(
            f"Found Places: {found_places.shape[0]}, Sampled Circles: {sampled_circles}, Saturated Circles: {saturated_circles.shape[0]}"
        )
        all_places = pd.concat([all_places, found_places], axis=0, ignore_index=True)

        print(f"Total Sampled Circles: {total_sampled_circles}")

    if True:
        all_places = all_places[
            [
                c
                for c in all_places.columns
                if c not in ["iconMaskBaseUri", "googleMapsUri", "websiteUri"]
            ]
        ].reset_index(drop=True)
        all_places.to_parquet(all_places_parquet_path)
        all_places.to_excel(all_places_excel_path, index=False)

        unique_places = all_places.drop_duplicates(subset=["id"]).reset_index(drop=True)
        unique_places.to_parquet(unique_places_parquet_path)
        unique_places.to_excel(unique_places_excel_path, index=False)

        print(f"Total Unique Found Places: {unique_places.shape[0]}")
