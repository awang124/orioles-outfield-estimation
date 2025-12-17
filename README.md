# Baseball Outfield Distance Estimation

This program (`outfield_estimation.py`) reads a ballpark image ("image-mode") or downloads it via Google Maps API ("map-mode"), and computes distances from home plate to the outfield wall at 91 evenly spaced angles between the foul lines.

## Getting Images via Google Maps
To obtain historical images (if current Google Maps imagery for a certain ballpark is insufficient, e.g. park is roofed), see this [short instructional video](https://youtu.be/DZWq781ROYY). Between Steps 2 and 3, select the clock icon at the top to select the year you want the image from.

## Prerequisites

- Python 3
- Google Maps Static API key

### Required Packages
```bash
pip install opencv-python numpy matplotlib requests
```

## Execution

These are bare-minimum execution commands: review Optional Arguments for further customization.

**Image-Mode**:
```
python outfield_estimation.py --image-path <YOUR_IMAGE_PATH>
```

**Map-Mode**:
```
python outfield_estimation.py --latitude <BALLPARK_LATITUDE> --longitude <BALLPARK_LONGITUDE> --api-key <GOOGLE_MAPS_API_KEY> --image-path <IMAGE_SAVE_PATH>
```
(Whether program runs in image-mode or map-mode depends simply on whether coordinates (and API-Key) are provided).

### Required Arguments
**--image-path**: If (--latitude, --longitude, --api-key) specified, downloads image via API to this path; otherwise, assumes user has image already downloaded, and reads from this path

### Required Arguments (Map-Mode Only)
**--latitude**: latitude of desired ballpark  
**--longitude**: longitude of desired ballpark  
**--api-key**: Google Maps API Key

### Additional Optional Arguments
**--csv-path**: Filename to which to save computed distances (`NONE` if no save desired, default `distances.csv`)  
**--vis-path**: Filename to which to save distance visualization (`NONE` if no save desired, default `distances_vis.png`)  
**--graph-path**: Filename to which to save angle/distance graph (`NONE` if no save desired, default `distances_graph.png`)  
**--field-type**: Field type (`MLB`, `MiLB`, `Olympic`), defines warning track length (default `MLB`)  
**--min-distance-feet**: Minimum distance (in feet) to consider (default `275`)  
**--smooth-level** : Maximum allowed jump (in feet) between consecutive measurements (`0` if no smoothing desired, default `5`)

### Additional Optional Arguments (Map-Mode Only)
**--zoom**: Google Maps zoom level (default `19`)  
**--size**: Google Maps image size (default `600`)

## Output Interpretation

- **Distance CSV File**:
  - Angle (deg): Angle from center field (-45° to +45°)
  - Distance (ft): Distance to outfield wall in feet
- **Distance Visualization Rays**:
  - **Green**: Success: wall detection with distance measurement
  - **Yellow**: Warning: field edge detected, but wall not found
  - **Red**: Failure: no boundary detected

## Refinement / Troubleshooting
If measurements seem suboptimal:
- Verify ballpark field clearly visible in image
- Increase / decrease `--smooth-level` to reweigh data vs smoothness
- Modify Google Maps `--zoom` and `--size` for better resolution
- Decrease `--min-distance-feet` for small parks

## Limitations

- Requires clear satellite imagery with visible field boundaries
- Works best on professional/well-defined baseball fields
- Accuracy depends on image resolution and zoom level
- May struggle with obscured boundaries or unusual field layouts

