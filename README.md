# Baseball Outfield Distance Estimation

This Jupyter Notebook (`Four Points.ipynb`) analyzes satellite imagery of baseball fields to calculate distances from home plate to the outfield wall at various angles. It uses the Google Maps Static API to retrieve field images and computer vision techniques to detect field boundaries.

## Getting Images
How to correctly collect images from Google Earth Pro. To get historical images, between step 2 and 3, select the clock icon at the top to select the year you want the image from and follow the steps in the video. 

Link: [https://youtu.be/DZWq781ROYY](url)

## Overview

The script processes a satellite image of a baseball field to:
- Detect the outfield wall boundaries using color segmentation and edge detection
- Calculate distances from home plate to the wall at angles ranging from -45° to +45°
- Generate a visualization image and CSV file with distance measurements
- Account for warning track offsets (typically 16 feet)

## Prerequisites

- Python 3.x
- Google Maps Static API key

### Required Python Packages
```bash
pip install opencv-python numpy matplotlib requests
```

## Configuration

Before running the script, update these variables in the code if you are using the Google Maps API instead of an uploaded image. 

1. **API Key**: Replace `api_key` with your Google Maps Static API key
2. **Field Coordinates**: Update the latitude/longitude values:
   - `lat, lng`: Center point of the map
   - `home_latlng`: Home plate coordinates
   - `first_latlng`: First base coordinates  
   - `third_latlng`: Third base coordinates
3. **Image Settings**: Adjust `size` and `zoom` for desired resolution

## Functions

### `grassmask(image)`
- **Purpose**: Creates a mask identifying grass areas using HSV color filtering
- **Parameters**: `image` - input BGR image

### `dirtmask(image)`
- **Purpose**: Creates a mask for identifying dirt areas using HSV color filtering
- **Parameters**: `image` - input BGR image

### `dirtedge(dirtmask, window)`
- **Purpose**: Detects dirt mask edges using Canny edge detection
- **Parameters**: `dirtmask` - binary dirt mask, `window` - blur kernel size for noise reduction

### `lines(edges, lengap)`
- **Purpose**: Finds line segments in edge images using Hough Transform
- **Parameters**: `edges` - edge image, `lengap` - tuple controlling minimum line length and maximum gap between segments

### `pixel_angle_from_home(home_px, target_px)`
- **Purpose**: Calculates angle from home plate to any pixel coordinates
- **Parameters**: `home_px` - home plate pixel coordinates, `target_px` - target pixel coordinates

### `shortest_angle_interp(a_from, a_to, t)`
- **Purpose**: Interpolates between two angles using shortest path around circle
- **Parameters**: `a_from` - starting angle, `a_to` - ending angle, `t` - interpolation factor (0-1)

### `latlng_to_pixel(lat, lng, center_lat, center_lng, zoom, size)`
- **Purpose**: Converts geographic coordinates to pixel coordinates in map image
- **Parameters**: `lat, lng` - target coordinates, `center_lat, center_lng` - map center coordinates, `zoom` - map zoom level, `size` - output image dimensions

### `distance_to_wall(home_px, angle_deg, wall_mask, min_distance_px)`
- **Purpose**: Raycasts from home plate at specified angle to find wall intersection
- **Parameters**: `home_px` - home plate position, `angle_deg` - ray angle, `wall_mask` - boundary mask, `min_distance_px` - minimum distance to ignore infield boundaries

### `find_wall_after_offset(home_px, angle_deg, start_point, offset_feet, feet_per_pixel, dirt_edges, max_extra_ft)`
- **Purpose**: Continues scanning beyond warning track to find actual wall
- **Parameters**: `home_px` - home plate position, `angle_deg` - ray angle, `start_point` - offset starting position, `offset_feet` - warning track width, `feet_per_pixel` - conversion factor, `dirt_edges` - edge detection output, `max_extra_ft` - maximum search distance beyond offset

### `smooth_distances(distances, max_jump_ft)`
- **Purpose**: Filters out measurement outliers using neighborhood averaging
- **Parameters**: `distances` - list of angle-distance tuples, `max_jump_ft` - maximum allowed distance difference between neighbors

## Usage

1. **Configure the script** with your field coordinates and API key
2. **Run the script**: `python baseball.py`
3. **Output files**:
   - `field_visualization.png`: Visual representation with distance lines
   - `field_distances.csv`: Table of angles and corresponding distances

## Output Interpretation

- **CSV Columns**:
  - `Angle (deg)`: Angle from center field (-45° to +45°)
  - `Distance (ft)`: Distance to outfield wall in feet
- **Visualization Colors**:
  - **Green**: Success: wall detection with distance measurement
  - **Yellow**: Warning: field edge detected, but wall not found
  - **Red**: Failure: no boundary detected



## Customization / Refinement

- Adjust `min_distance_ft` (currently 275) if detecting boundaries for a smaller park such as a college or high school dield
- Modify color ranges in `grassmask()` and `dirtmask()` for different field conditions
- Tweak `smooth_distances()` parameters if measurements are too jumpy. Add more to smooth more. 
- Adjust `feet_per_pixel` calculation if the distance between home and first plate isn't exactly 90 feet. 

## Troubleshooting

If measurements seem inaccurate:
- Check coordinate accuracy using Google Maps
- Verify the field is clearly visible in the generated `map.png`
- Adjust color thresholds in masking functions
- Modify the `zoom` level for better resolution

## Limitations

- Requires clear satellite imagery with visible field boundaries
- Works best on professional/well-defined baseball fields
- Accuracy depends on image resolution and zoom level
- May struggle with obscured boundaries or unusual field layouts

