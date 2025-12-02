import numpy as np
import matplotlib.pyplot as plt

def analyze_and_summarize(distances, park):
    summary = {
        "green_hits": [],
        "yellow_partial": [],
        "red_fail": [],
        "distances_ft": [],
        "angles": [],
    }

    for angle, dist in distances:
        summary["angles"].append(angle)

        if dist is None:
            summary["red_fail"].append(angle)
            continue

        summary["distances_ft"].append(dist)

        if dist < 300:
            summary["yellow_partial"].append(angle)
        else:
            summary["green_hits"].append(angle)

    dist_array = np.array([d for d in summary["distances_ft"] if d is not None])

    min_dist = float(np.min(dist_array)) if dist_array.size > 0 else None
    max_dist = float(np.max(dist_array)) if dist_array.size > 0 else None
    avg_dist = float(np.mean(dist_array)) if dist_array.size > 0 else None
    std_dist = float(np.std(dist_array)) if dist_array.size > 0 else None

    errors = []
    for i in range(1, len(distances) - 1):
        a, d = distances[i]
        _, prev_d = distances[i - 1]
        _, next_d = distances[i + 1]
        if d and prev_d and next_d:
            neighbor_avg = (prev_d + next_d) / 2
            errors.append(abs(d - neighbor_avg))

    avg_error = float(np.mean(errors)) if errors else None
    std_error = float(np.std(errors)) if errors else None

    groups = {
        "LF": list (range(-45, -14)),
        "CF": list (range(-14, 15)),
        "RF": list (range(15, 46)),
    }

    group_stats = {}
    for name, angle_range in groups.items():
        vals = [dist for ang, dist in distances if ang in angle_range and dist is not None]
        if vals:
            group_stats[name] = {
                "count": len(vals),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "avg": float(np.mean(vals)),
            }
        else:
            group_stats[name] = None

    
    angles_plot = [a for a, _ in distances]
    d_plot = [d if d is not None else np.nan for _, d in distances]

    plt.figure(figsize=(8,3))
    plt.plot(angles_plot, d_plot, linewidth=2)
    plt.title(f"{park}: Angle vs Outfield Distance")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Distances (ft)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{park} Angle vs Distance.png", dpi=150)
    plt.show()
    plt.close()

    warnings = []
    if len(summary["red_fail"]) > 10:
        warnings.append("Many angles failed; the masks might need tuning")

    if avg_error and avg_error > 10:
        warnings.append("Large spikes are detected, so smoothing might need tuning")
    
    if min_dist and min_dist < 250:
        warnings.append("Some distances are very short (less than 250ft)")

    
    print("\n============= FIELD SUMMARY =============")
    print(f"Successful detections (green): {len(summary['green_hits'])}")
    print(f"Partial detections (yellow): {len(summary['yellow_partial'])}")
    print(f"Failed detections (red): {len(summary['red_fail'])}")

    print("\nDISTANCE  STATISTICS:")
    print(f"Min distance: {min_dist}")
    print(f"Max distance: {max_dist}")
    print(f"Avg distance: {avg_dist}")
    print(f"Std deviation: {std_dist}")

    print("\nERROR ANALYSIS:")
    print(f"Avg local error: {avg_error}")
    print(f"Std error: {std_error}")

    print("\nREGIONAL BREAKDOWN (LF, CF, RF):")
    for name, stats in group_stats.items():
        if stats:
            print(f"{name}: avg={stats['avg']:.1f} ft, min={stats['min']:.1f}, max={stats['max']:.1f}")
        else:
            print(f"{name}: no valid values")
    
    print("\nWARNINGS:")
    if warnings:
        for w in warnings:
            print("*", w)
    else:
        print("None")
    
    print("Mini plot saved as:", f"{park} Angle vs Distance.png")
    


