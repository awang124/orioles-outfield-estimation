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
        "--zoom", type=int, default=19, help="Map zoom"
    )
    parser.add_argument(
        "--api-key", help="Google Maps API Key"
    )
    parser.add_argument(
        "--image-path", default="field.png", help="Image path"
    )
    parser.add_argument(
        "--csv-path", default="distances.csv", help="CSV output path"
    )
    parser.add_argument(
        "--vis-path", default="distances_vis.png", help="Visualization output path"
    )
    return parser.parse_args()

def download_image(args):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center="
    url += f"{args.latitude},{args.longitude}&zoom={args.zoom}"
    url += f"&size=600x600&maptype=satellite&key={args.api_key}"
    with open(args.image_path, "wb") as f:
        f.write(requests.get(url).content)

def select_bases(image):
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

def compute_dirtmask(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask1 = cv.inRange(hsv, (0, 30, 30), (30, 255, 255))
    mask2 = cv.inRange(hsv, (150, 30, 30), (180, 255, 255))
    mask = cv.bitwise_or(mask1, mask2)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask

def pixel_angle_from_home(home_px, target_px):
    (xh, yh), (xt, yt) = home_px, target_px
    return math.degrees(math.atan2(-(yt - yh), xt - xh))

def shortest_angle_interp(a_from, a_to, t):
    norm = lambda a: (a + 180) % 360 - 180
    a_from_n, a_to_n = norm(a_from), norm(a_to)
    if (diff := a_to_n - a_from_n) > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return a_from_n + diff * t

def distance_to_outer_dirt(home_px, angle_deg, dirt_mask, feet_per_pixel):
    angle_rad = math.radians(angle_deg)
    dx, dy = math.cos(angle_rad), -math.sin(angle_rad)
    MIN_DISTANCE_FEET, MAX_DISTANCE_PX = 200, 3000
    min_distance_px = int(MIN_DISTANCE_FEET / feet_per_pixel)

    x, y = home_px
    prev_transition = (None, None)
    prev_val = dirt_mask[int(y), int(x)]
    for i in range(1, MAX_DISTANCE_PX):
        xi = int(x + i * dx)
        yi = int(y + i * dy)
        if not 0 <= xi < dirt_mask.shape[1]:
            break
        if not 0 <= yi < dirt_mask.shape[0]:
            break
        val = dirt_mask[yi, xi]
        if prev_val > 127 and val <= 127 and i > min_distance_px:
            prev_transition = (i, (xi, yi))
        prev_val = val
    return prev_transition

def compute_distances(args, image, home_px, first_px, third_px):
    distances = []
    dirt_mask = compute_dirtmask(image)
    angle_first_img = pixel_angle_from_home(home_px, first_px)
    angle_third_img = pixel_angle_from_home(home_px, third_px)
    feet_per_pixel = 90 / np.linalg.norm(np.array(home_px) - np.array(first_px))

    for angle_logical in range(-45, 46):
        t = (angle_logical + 45) / 90
        angle_image = shortest_angle_interp(
            angle_third_img, angle_first_img, t
        )
        d_pix, hit_point = distance_to_outer_dirt(
            home_px, angle_image, dirt_mask, feet_per_pixel
        )
        d_feet = d_pix * feet_per_pixel if d_pix is not None else None
        distances.append((angle_logical, d_feet))
    return distances

def smooth_distances(distances, max_jump_ft=4):
    smoothed = distances.copy()
    for i in range(1, len(distances) - 1):
        if not -38 <= distances[i][0] <= 38:
            continue
        prev_d = distances[i - 1][1]
        curr_d = distances[i   ][1]
        next_d = distances[i + 1][1]
        if curr_d is None or prev_d is None or next_d is None:
            continue
        if max(abs(curr_d - prev_d), abs(curr_d - next_d)) > max_jump_ft:
            smoothed[i] = (
                distances[i][0],
                (3 * min(prev_d, next_d) + max(prev_d, next_d)) / 4
            )
    return smoothed

def create_visualization(image, distances, home_px, first_px, third_px):
    vis = image.copy()
    cv.circle(vis, home_px, 5, (255, 0, 0), -1)
    angle_first_img = pixel_angle_from_home(home_px, first_px)
    angle_third_img = pixel_angle_from_home(home_px, third_px)
    feet_per_pixel = 90 / np.linalg.norm(np.array(home_px) - np.array(first_px))
    
    for angle_logical, d_feet in distances:
        if d_feet is None or np.isnan(d_feet):
            continue
        t = (angle_logical + 45) / 90
        angle_image = shortest_angle_interp(
            angle_third_img, angle_first_img, t
        )
        angle_rad = math.radians(angle_image)
        d_pix = d_feet / feet_per_pixel
        end_pt = (
            int(home_px[0] + d_pix * math.cos(angle_rad)),
            int(home_px[1] - d_pix * math.sin(angle_rad))
        )
        cv.line(vis, home_px, end_pt, (0, 255, 0), 1)
        cv.circle(vis, end_pt, 3, (0, 255, 0), -1)
        cv.putText(
            vis, f"{int(round(d_feet))}ft", (end_pt[0] + 4, end_pt[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv.LINE_AA
        )
    return vis

def write_outputs(args, distances, visualization):
    cv.imwrite(args.vis_path, visualization)
    with open(args.csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Angle (deg)", "Distance (ft)"])
        for angle, dist in distances:
            writer.writerow([angle, dist if dist is not None else "NA"])

def main():
    args = parse_arguments()
    if all(a for a in (args.latitude, args.longitude, args.zoom, args.api_key)):
        download_image(args)
    # currdir = os.path.dirname(os.path.abspath(__file__))
    # image = cv.imread(os.path.join(currdir, args.image_path))
    image = cv.imread(args.image_path)
    home_px, first_px, third_px = select_bases(image)
    distances = compute_distances(args, image, home_px, first_px, third_px)
    for max_jump_ft in (8, 8, 7, 7, 6, 6, 5, 5, 4, 4):
        distances = smooth_distances(distances)
    visualization = create_visualization(image, distances, home_px, first_px, third_px)
    write_outputs(args, distances, visualization)

if __name__ == "__main__":
    main()