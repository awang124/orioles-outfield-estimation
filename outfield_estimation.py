#!/usr/bin/env python3
import matplotlib
matplotlib.use("TkAgg")

import argparse
import os
import csv
import math
import requests

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def parse_arguments():
    """
    Read user-specified command-line arguments. See
    README.md for full descriptions of all arguments.

    Returns
    -------
    argparse.Namespace
        Command-line arguments (each accessed as an attribute)
    """
    parser = argparse.ArgumentParser(
        description="Estimate baseball outfield dimensions from satellite imagery"
    )
    parser.add_argument(
        "--latitude", type=float, help="Map latitude coordinate"
    )
    parser.add_argument(
        "--longitude", type=float, help="Map longitude coordinate"
    )
    parser.add_argument(
        "--zoom", type=int, default=19, help="Map image zoom"
    )
    parser.add_argument(
        "--size", type=int, default=600, help="Map image size"
    )
    parser.add_argument(
        "--api-key", help="Google Maps API key"
    )
    parser.add_argument(
        "--image-path", help="Image path"
    )
    parser.add_argument(
        "--csv-path", default="distances.csv", help="CSV output path"
    )
    parser.add_argument(
        "--vis-path", default="distances_vis.png", help="Visualization output path"
    )
    parser.add_argument(
        "--graph-path", default="distances_graph.png", help="Graph output path"
    )
    parser.add_argument(
        "--field-type", choices=("MLB", "MiLB", "Olympic"), default="MLB", help="Field type"
    )
    parser.add_argument(
        "--min-distance-feet", type=int, default=275, help="Min distance (ft) to consider"
    )
    parser.add_argument(
        "--smooth-level", type=int, default=5, help="Max consecutive distance difference (ft)"
    )
    return parser.parse_args()

def download_image(image_path, latitude, longitude, zoom, size, api_key):
    """
    Download ballpark image at specified coordinates from Google Maps API.

    Parameters
    ----------
    image_path: string
        Filename to which to save downloaded image
    latitude: float
        Ballpark latitude
    longitude: float
        Ballpark longitude
    zoom: int
        Google Maps zoom
    size: int
        Google Maps image size
    api_key: string
        User's Google Maps API Key
    """
    url = f"https://maps.googleapis.com/maps/api/staticmap?center="
    url += f"{latitude},{longitude}&zoom={zoom}"
    url += f"&size={size}x{size}maptype=satellite&key={api_key}"
    with open(image_path, "wb") as f:
        f.write(requests.get(url).content)

def select_bases(image):
    """
    Prompt user to select home plate, first base, and third base
    from field image, and return their (pixel) coordinates.

    Parameters
    ----------
    image: numpy.ndarray
        Field image

    Returns
    -------
    home_px: Tuple[int, int]
        Home plate coordinates
    first_px: Tuple[int, int]
        First base coordinates
    third_px: Tuple[int, int]
        Third base coordinates
    """
    fig, axs = plt.subplots()
    axs.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Please click on: (1) Home Plate, (2) First Base, (3) Third Base")
    
    plt.draw()
    plt.pause(0.1)
    pts = plt.ginput(3, timeout=0)
    plt.close()
    
    home_px = (int(pts[0][0]), int(pts[0][1]))
    first_px = (int(pts[1][0]), int(pts[1][1]))
    third_px = (int(pts[2][0]), int(pts[2][1]))
    return home_px, first_px, third_px
    
def compute_grassmask(image):
    """
    Compute grass mask (binary mask indicating whether each pixel
    is grass (falls within a certain range of green in color)).

    Parameters
    ----------
    image: numpy.ndarray
        Field image

    Returns
    -------
    numpy.ndarray
        Mask
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (30, 40, 40), (90, 256, 256))

    # FILL IN SMALL GAPS IN GRASS AREAS
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask

def compute_dirtmask(image):
    """
    Compute dirt mask (binary mask indicating whether each pixel is dirt
    (falls within a certain range of reddish-brown or light-blue in color)).

    Parameters
    ----------
    image: numpy.ndarray
        Field image

    Returns
    -------
    numpy.ndarray
        Mask
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, (0, 30, 30), (30, 256, 256))
    mask2 = cv.inRange(hsv, (150, 30, 30), (180, 256, 256))
    mask = cv.bitwise_or(mask1, mask2)

    # FILL IN SMALL GAPS IN DIRT AREAS
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask

def compute_edges(image, window=(9, 9)):
    """
    Compute edges (binary mask indicating whether
    each pixel is an edge of some image object).

    Parameters
    ----------
    image: numpy.ndarray
        Field image
    window: Tuple[int, int]
        Gaussian blur window

    Returns
    -------
    numpy.ndarray
        Edges
    """
    blurred = cv.GaussianBlur(image, window, -1)
    return cv.Canny(blurred, 30, 100, apertureSize=3)

def pixel_angle_from_home(home_px, target_px):
    """
    Compute angle of target pixel w.r.t home plate.

    Parameters
    ----------
    home_px: Tuple[int, int]
        Home plate coordinates
    target_px: Tuple[int, int]
        Target pixel coordinates

    Returns
    -------
    float
        Angle (in degrees)
    """
    (xh, yh), (xt, yt) = home_px, target_px
    return math.degrees(math.atan2(-(yt - yh), xt - xh))

def shortest_angle_interp(a_from, a_to, t):
    """
    Given two angles, compute angle 100t% between them. For example:
    t = 0   --> res = a_from
    t = 0.5 --> res = (a_from + a_to) / 2
    t = 1   --> res = a_to
    
    Parameters
    ----------
    a_from: float
        First angle
    a_to: float
        Second angle
    t: float
        Where between them

    Returns
    -------
    float
        Angle between
    """
    norm = lambda a: (a + 180) % 360 - 180
    a_from_n, a_to_n = norm(a_from), norm(a_to)
    if (diff := a_to_n - a_from_n) > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return a_from_n + diff * t

def compute_wallmask(image):
    """
    Compute wall mask (binary mask indicating whether each pixel
    belongs to outfield wall). Computes edges of largest grass
    contour, so really computes whether grass meets warning track.
    
    Parameters
    ----------
    image: numpy.ndarray
        Field image

    Returns
    -------
    numpy.ndarray
        Mask
    """
    grass_mask = compute_grassmask(image)
    if cont := cv.findContours(grass_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
        wall_mask = np.zeros_like(grass_mask)
        cv.drawContours(wall_mask, [max(cont, key=cv.contourArea)], -1, 255, 2)
    else:
        wall_mask = compute_edges(grass_mask)
    return wall_mask

def distance_to_wall(home_px, angle_deg, wall_mask, min_distance_px):
    """
    Compute distance from home plate to outfield wall along a single ray (determined
    by a single angle) by trying each distance until outfield wall is hit.
    
    Parameters
    ----------
    home_px: Tuple[int, int]
        Home plate coordinates
    angle_deg: float
        Angle of ray (relative from home plate) in degrees
    wall_mask: numpy.ndarray
        Outfield wall location (see compute_wallmask)
    min_distance_px: int
        Minimum / starting distance value (in pixels) to consider

    Returns
    -------
    int
        Distance
    Tuple[int, int]
        Pixel coordinates of point where hit occurs
    """
    MAX_DISTANCE_PX = 3000
    angle_rad = math.radians(angle_deg)
    dx, dy = math.cos(angle_rad), -math.sin(angle_rad)

    x, y = home_px

    # FIND SMALLEST DISTANCE VALUE WHICH REACHES OUTFIELD WALL
    for i in range(1, MAX_DISTANCE_PX):
        xi = int(x + i * dx)
        yi = int(y + i * dy)
        if not 0 <= xi < wall_mask.shape[1]:
            break
        if not 0 <= yi < wall_mask.shape[0]:
            break
        if wall_mask[yi, xi] > 0 and i > min_distance_px:
            return i, (xi, yi)
    return None, None

def find_wall_after_offset(
    start_point, angle_deg, dirt_edges,
    offset_feet, feet_per_pixel, max_extra_feet=100
):
    """
    Updates initial computed distance (see distance_to_wall) by computing
    number of extra feet along a given ray is required for the point at
    which it hits the warning track to actually hit the outfield wall.
    
    Parameters
    ----------
    start_point: Tuple[int, int]
        Pixel coordinates of point where ray hits warning track
    angle_deg: float
        Angle of ray (relative from home plate) in degrees
    dirt_edges: numpy.ndarray
        Edges of dirt regions (see compute_dirtmask, compute_mask_edge)
    offset_feet: int
        Warning track length
    feet_per_pixel: float
        Number of feet per one image pixel
    max_extra_feet: int
        Maximum / ending extra feet to try

    Returns
    -------
    Tuple[int, int]
        Pixel coordinates of point where hit occurs
    int
        Extra feet
    """
    angle_rad = math.radians(angle_deg)
    dx, dy = math.cos(angle_rad), -math.sin(angle_rad)
    max_extra_px = int(max_extra_feet / feet_per_pixel)

    # FIND SMALLEST DISTANCE VALUE WHICH REACHES WARNING TRACK EDGE
    x, y = start_point
    for i in range(1, max_extra_px):
        xi = int(x + i * dx)
        yi = int(y + i * dy)
        if not 0 <= xi < dirt_edges.shape[1]:
            break
        if not 0 <= yi < dirt_edges.shape[0]:
            break
        if dirt_edges[yi, xi] > 0:
            return (xi, yi), offset_feet + (i * feet_per_pixel)
    return None, None

def compute_distances(image, home_px, first_px, third_px, field_type, min_distance_feet):
    """
    Updates initial computed distance (see distance_to_wall) by computing
    number of extra feet along a given ray is required for the point at
    which it hits the warning track to actually hit the outfield wall.
    
    Parameters
    ----------
    image: numpy.ndarray
        Field image
    home_px: Tuple[int, int]
        Home plate coordinates
    first_px: Tuple[int, int]
        First base coordinates
    third_px: Tuple[int, int]
        Third base coordinates
    field_type: str
        Field type, determines warning track length estimate
    min_distance_feet: int
        Minimum / starting distance value (in feet) to consider

    Returns
    -------
    List[Tuple[float, Tuple[int, int], int]]
        Each list entry contains an angle (which determines a ray from home base),
        the point (pixel coordinates) at which its ray hits the outfield wall,
        and the distance (number of pixels from home plate to that point)
    """
    # GET INITIAL WARNING TRACK LENGTH ESTIMATE
    track_len = {
        "MLB": 15,
        "MiLB": 8,
        "Olympic": 20
    }[field_type]
    
    distances = []
    
    # EXTRACT OUTFIELD WALL & EDGES OF DIRT REGIONS FROM IMAGE
    wall_mask = compute_wallmask(image)
    dirt_mask = compute_dirtmask(image)
    dirt_edges = compute_edges(dirt_mask)
    
    # COMPUTE ANGLE OF FIRST / THIRD RELATIVE TO HOME IN IMAGE
    angle_first_img = pixel_angle_from_home(home_px, first_px)
    angle_third_img = pixel_angle_from_home(home_px, third_px)

    feet_per_pixel = 90 / np.linalg.norm(np.array(home_px) - np.array(first_px))
    feet_per_pixel /= (FEET_PER_PIXEL_CALIBRATION_TWEAK := 1.018347812)
    min_distance_px = int(min_distance_feet / feet_per_pixel)

    # COMPUTE DISTANCE TO OUTFIELD WALL FOR EACH ANGLE
    for angle_logical in range(-45, 46):

        # MAP LOGICAL ANGLE (0 STRAIGHT DOWN CENTER FIELD) TO ACTUAL IMAGE ANGLE
        t = (angle_logical + 45) / 90
        angle_image = shortest_angle_interp(
            angle_third_img, angle_first_img, t
        )

        # COMPUTE INITIAL ESTIMATE (TO WARNING TRACK, PLUS WARNING TRACK LENGTH)
        d_pix, hit_point = distance_to_wall(
            home_px, angle_image, wall_mask, min_distance_px
        )
        if d_pix is None:
            distances.append((angle_logical, None, np.nan))
            continue

        d_feet = d_pix * feet_per_pixel
        distances.append((angle_logical, hit_point, d_feet + track_len))
        
        # UPDATE ESTIMATE TO DISTANCE FROM WARNING TRACK POINT TO ACTUAL WALL
        angle_rad = math.radians(angle_image)
        extra_px = int(track_len / feet_per_pixel)
        offset_end = (
            int(hit_point[0] + extra_px * math.cos(angle_rad)),
            int(hit_point[1] - extra_px * math.sin(angle_rad))
        )
        # UPDATE DISTANCE VALUE BASED ON WALL POINT
        wall_point, extra_feet = find_wall_after_offset(
            offset_end, angle_image, dirt_edges, track_len, feet_per_pixel
        )
        if wall_point is None:
            continue
        distances[-1] = (angle_logical, hit_point, d_feet + extra_feet)
        
    return distances

def smooth_distances(distances, max_jump_ft):
    """
    Smooth distances using weighted averaging. For each point,
    consider its two neighbors: if a difference between distances
    of that point and either neighbor exceeds a maximum tolerance,
    that point's distance is replaced with a weighted average, where
    the smaller difference is weighed by 3/4, and the larger by 1/4.

    Parameters
    ----------
    distances: List[float, Tuple[int, int], int]
        Angles, hit points, distances (see compute_distances)
    max_jump_ft: int
        Maximum tolerance for distance differences

    Returns
    -------
    List[float, Tuple[int, int], int]
        Same angles / hit points, with smoothed distances
    """
    smoothed = distances.copy()
    
    # SMOOTH EACH DISTANCE BASED ON ITS TWO IMMEDIATE NEIGHBORS
    for i in range(1, len(distances) - 1):
        if not -38 <= distances[i][0] <= 38:
            continue
        prev_d = distances[i - 1][-1]
        curr_d = distances[i    ][-1]
        next_d = distances[i + 1][-1]
        if np.isnan((prev_d, curr_d, next_d)).sum():
            continue
        
        # IF DIFFERENCE EXCEEDS TOLERANCE, REPLACE WITH WEIGHTED AVERAGE
        if min(abs(curr_d - prev_d), abs(curr_d - next_d)) > max_jump_ft:
            smoothed[i] = (
                distances[i][0], distances[i][1],
                (3 * min(prev_d, next_d) + max(prev_d, next_d)) / 4
            )
    return smoothed

def create_visualization(image, distances, home_px, first_px, third_px):
    """
    Create visualization (annotate field image, highlighting rays,
    displaying computed distances, and marking home plate).

    Parameters
    ----------
    image: numpy.ndarray
        Field image
    distances: List[float, Tuple[int, int], int]
        Angles, hit points, distances (see compute_distances)
    home_px: Tuple[int, int]
        Home plate coordinates
    first_px: Tuple[int, int]
        First base coordinates
    third_px: Tuple[int, int]
        Third base coordinates

    Returns
    -------
    numpy.ndarray
        Visualization
    """
    vis = image.copy()

    # MARK HOME PLATE
    cv.circle(vis, home_px, 5, (255, 0, 0), -1)

    angle_first_img = pixel_angle_from_home(home_px, first_px)
    angle_third_img = pixel_angle_from_home(home_px, third_px)
    feet_per_pixel = 90 / np.linalg.norm(np.array(home_px) - np.array(first_px))
    feet_per_pixel /= (FEET_PER_PIXEL_CALIBRATION_TWEAK := 1.018347812)
    extra_px = int(21 / feet_per_pixel)
    
    # MARK EACH RAY (ANGLE) AND ITS CORRESPONDING DISTANCE
    for angle_logical, hit_point, d_feet in distances:
        t = (angle_logical + 45) / 90
        angle_image = shortest_angle_interp(
            angle_third_img, angle_first_img, t
        )
        angle_rad = math.radians(angle_image)
        
        # IF COMPUTATION FAILED FOR THIS ANGLE, MARK RED RAY
        if np.isnan(d_feet):
            end = (
                int(home_px[0] + 1500 * math.cos(angle_rad)),
                int(home_px[1] - 1500 * math.sin(angle_rad))
            )
            cv.line(vis, home_px, end, (0, 0, 255), 1)
            continue

        # EXTEND RAY SLIGHTLY TO DECLUTTER VISUALIZATION
        extended_end = (
            int(hit_point[0] + extra_px * math.cos(angle_rad)),
            int(hit_point[1] - extra_px * math.sin(angle_rad))
        )
        # MARK RAY GREEN, PRINT DISTANCE NEXT TO RAY TERMINUS
        cv.line(vis, home_px, extended_end, (0, 255, 0), 1)
        cv.circle(vis, extended_end, 3, (0, 255, 0), -1)
        cv.putText(
            vis, f"{int(round(d_feet))}ft",
            (extended_end[0] + 4, extended_end[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA
        )
        
    return vis

def analyze_results(graph_path, distances):
    """
    Analyze computed distances:

    Parameters
    ----------
    graph_path: str
        Filename to which to save angle/distance graph (None if no save desired)
    distances: List[float, Tuple[int, int], int]
        Angles, hit points, distances (see compute_distances)
    """
    summary = {
        "green_hits": [],
        "yellow_partial": [],
        "red_fail": [],
        "distances_ft": [],
        "angles": [],
    }

    # SORT DISTANCES INTO FAILURES (DUE TO NOT FINDING OUTFIELD
    # WALL, ETC.), SUSPICIOUS ESTIMATES, AND SUCCESSES
    for angle, _, dist in distances:
        summary["angles"].append(angle)
        summary["distances_ft"].append(dist)
        if np.isnan(dist):
            summary["red_fail"].append(angle)
        elif dist < 300:
            summary["yellow_partial"].append(angle)
        else:
            summary["green_hits"].append(angle)

    dist_list = [d for d in summary["distances_ft"] if not np.isnan(d)]
    min_dist, max_dist, avg_dist, std_dist = np.nan, np.nan, np.nan, np.nan

    # COMPUTE DISTANCE SUMMARY STATISTICS (MIN, MAX, MEAN, STD)
    if dist_list:
        min_dist = float(np.min (dist_list))
        max_dist = float(np.max (dist_list))
        avg_dist = float(np.mean(dist_list))
        std_dist = float(np.std (dist_list))

    # COMPUTE ERROR (DIFFERENCE BETWEEN EACH DISTANCE AND ITS
    # TWO IMMEDIATE NEIGHBORS' AVERAGE) SUMMARY STATISTICS
    errors = []
    for i in range(1, len(distances) - 1):
        a, _, d = distances[i]
        _, _, prev_d = distances[i - 1]
        _, _, next_d = distances[i + 1]
        if d and prev_d and next_d:
            neighbor_avg = (prev_d + next_d) / 2
            errors.append(abs(d - neighbor_avg))
    avg_error, std_error = np.nan, np.nan
    if errors:
        avg_error = float(np.nanmean(errors))
        std_error = float(np.nanstd(errors))

    # COMPUTE DISTANCE SUMMARY STATISTICS FOR EACH
    # REGION (LEFT / CENTER / RIGHT FIELD)
    angle_groups = {
        "LF": (-45, -14),
        "CF": (-14, 15),
        "RF": (15, 46)
    }
    group_stats = {}
    summary["angles"] = np.array(summary["angles"])
    summary["distances_ft"] = np.array(summary["distances_ft"])
    for name, (lower, upper) in angle_groups.items():
        cond = (lower <= summary["angles"]) & (summary["angles"] < upper)
        if (~np.isnan(group_vals := summary["distances_ft"][cond])).sum():
            group_stats[name] = {
                "count": group_vals.size,
                "min": float(np.nanmin (group_vals)),
                "max": float(np.nanmax (group_vals)),
                "avg": float(np.nanmean(group_vals)),
                "std": float(np.nanstd (group_vals))
            }
        else:
            group_stats[name] = None

    # CREATE DISTANCE VS ANGLE LINE PLOT
    if graph_path != "NONE":
        plt.figure(figsize=(5, 4))
        plt.plot(summary["angles"], summary["distances_ft"], linewidth=2)
        plt.title("Angle vs Outfield Distance")
        plt.xlabel("Angle (deg)")
        plt.ylabel("Distances (ft)")
        plt.grid(True, alpha=0.3)
        plt.savefig(graph_path, dpi=150)

    # PRINT ALL COMPUTED STATISTICS
    print("\n============= FIELD SUMMARY =============")
    print(f"Successful detections (green): {len(summary["green_hits"])}")
    print(f"Partial detections (yellow): {len(summary["yellow_partial"])}")
    print(f"Failed detections (red): {len(summary["red_fail"])}")

    print("\nDISTANCE  STATISTICS:")
    print(f"Min distance: {min_dist}")
    print(f"Max distance: {max_dist}")
    print(f"Avg distance: {avg_dist}")
    print(f"Distance Std: {std_dist}")

    print("\nERROR ANALYSIS:")
    print("Absolute differences between each distance and its immediate neighbors' average")
    print(f"Avg error: {avg_error}")
    print(f"Error Std: {std_error}")

    print("\nREGIONAL BREAKDOWN (LEFT / CENTER / RIGHT FIELD):")
    for name, stats in group_stats.items():
        if stats:
            print(f"{name}:", end=" ")
            print(f"min = {stats["min"]:.1f} ft,", end=" ")
            print(f"max = {stats["max"]:.1f} ft,", end=" ")
            print(f"avg = {stats["avg"]:.1f} ft,", end=" ")
            print(f"std = {stats["std"]:.1f} ft")
        else:
            print(f"{name}: no valid values")

    # PRINT WARNINGS (POTENTIAL CAUSES OF ERRONEOUS COMPUTATION, IF PRESENT)
    if len(summary["red_fail"]) > 10:
        print("WARNING: Many angles failed, field dirt may be atypically colored.")
    if avg_error and avg_error > 10:
        print("WARNING: Large spikes detected, may need increased smoothing.")
    if min_dist and min_dist < 250:
        print("WARNING: Some distances very short (<250ft).")

def write_outputs(csv_path, vis_path, distances, visualization):
    """
    Save distances as CSV, and visualization as image.

    Parameters
    ----------
    csv_path: str
        Filename to which to save distances (None if no save desired)
    vis_path: str
        Filename to which to save visualization (None if no save desired)
    distances: List[float, Tuple[int, int], int]
        Angles, hit points, distances (see compute_distances)
    visualization: numpy.ndarray
        Distance visualization (annotated image)
    """
    if csv_path != "NONE":
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Angle (deg)", "Distance (ft)"])
            for angle, _, dist in distances:
                writer.writerow([angle, dist if dist is not None else "NA"])
    if vis_path != "NONE":
        cv.imwrite(vis_path, visualization)

def main():

    # READ COMMAND-LINE ARGUMENTS, DOWNLOAD IMAGE (IF NECESSARY)
    args = parse_arguments()
    map_info = (args.latitude, args.longitude, args.zoom, args.size, args.api_key)
    if all(arg for arg in map_info):
        download_image(args.image_path, *map_info)

    # READ IMAGE, OBTAIN HOME / FIRST / THIRD LOCATIONS, COMPUTE DISTANCES
    image = cv.imread(args.image_path)
    home_px, first_px, third_px = select_bases(image)
    distances = compute_distances(
        image, home_px, first_px, third_px,
        args.field_type, args.min_distance_feet
    )
    
    # SMOOTH DISTANCES (IF DESIRED)
    if args.smooth_level > 0:
        distances = smooth_distances(distances, args.smooth_level)
    
    # CREATE VISUALIZATION, PRINT SUMMARY, SAVE RESULTS
    visualization = create_visualization(image, distances, home_px, first_px, third_px)
    analyze_results(args.graph_path, distances)
    write_outputs(args.csv_path, args.vis_path, distances, visualization)

if __name__ == "__main__":
    main()

