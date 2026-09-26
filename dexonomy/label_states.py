"""Index actual near-contact combinations without changing generated states."""

import argparse
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np

from dexonomy.util.file_util import load_scene_cfg
from dexonomy.util.task_region import TaskRegions


def body_tag(name, object_name="joystick_006"):
    prefix = "obj-" + object_name
    if not name.startswith(prefix):
        return None
    link = name[len(prefix):]
    if link == "stick_link":
        return "stick"
    if link == "trigger_00_link":
        return "trigger"
    if link.startswith("button_") and link.endswith("_link"):
        return link[:-5]
    return None


def combination_labels(labels):
    ordered = sorted(labels)
    return ["+".join(c) for n in range(2, len(ordered)+1) for c in combinations(ordered, n)]


def summarize_contacts(state, max_gap=0.002, max_penetration=0.0,
                       regions=None, region_tolerance=0.004):
    c = state["ho_c"]
    distances = np.asarray(c["dist"])
    mask = (distances <= max_gap + 1e-9) & (distances >= -max_penetration - 1e-9)
    groups = {}
    for i in np.flatnonzero(mask):
        tag = body_tag(c["bn2"][i])
        if tag is not None:
            groups.setdefault(tag, set()).add(c["bn1"][i])
    labels = sorted(groups)
    task_labels = regions.contact_labels(c, mask, region_tolerance) if regions else []
    region_combinations = [combo for combo in combination_labels(task_labels)
                           if len({regions.bodies[n] for n in combo.split("+")})
                           == len(combo.split("+"))]
    return {
        "body_labels": labels,
        "body_combinations": combination_labels(labels),
        "multitask_contact_candidate": len(labels) >= 2,
        "hand_bodies_by_part": {k: sorted(v) for k, v in groups.items()},
        "region_labels": task_labels,
        "region_combinations": region_combinations,
        "base_near_contact": any(mask[i] and b == "obj-joystick_006object_root"
                                 for i, b in enumerate(c["bn2"])),
        "minimum_gap_m": float(distances.min()) if len(distances) else None,
        "contacts": [{"hand_body": c["bn1"][i], "object_body": c["bn2"][i],
                      "gap_m": float(distances[i]), "position_world": np.asarray(c["pos"])[i].tolist(),
                      "normal_world": np.asarray(c["normal"])[i].tolist()}
                     for i in np.flatnonzero(mask)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--max-gap-mm", type=float, default=2)
    parser.add_argument("--max-penetration-mm", type=float, default=0)
    parser.add_argument("--region-tolerance-mm", type=float, default=4)
    args = parser.parse_args()
    if min(args.max_gap_mm, args.max_penetration_mm, args.region_tolerance_mm) < 0:
        parser.error("Distance thresholds must be nonnegative")
    paths = sorted(args.data.rglob("*.npy"))
    if not paths:
        parser.error("No generated states found")
    regions = TaskRegions(args.regions) if args.regions else None
    checked = set()
    records, body_index, region_index = [], {}, {}
    for path in paths:
        state = np.load(path, allow_pickle=True).item()
        if regions and state["scene_path"] not in checked:
            regions.check_scene(load_scene_cfg(state["scene_path"]))
            checked.add(state["scene_path"])
        record = summarize_contacts(state, args.max_gap_mm/1000, args.max_penetration_mm/1000,
                                    regions, args.region_tolerance_mm/1000)
        record.update(path=path.relative_to(args.data).as_posix(), scene_path=state["scene_path"])
        if "region_guidance" in state:
            guidance = state["region_guidance"]
            record["requested_regions"] = guidance["tasks"]
            record["initial_region_distances_m"] = np.asarray(guidance["init_distances_m"]).tolist()
            record["all_requested_regions_contacted"] = (
                set(guidance["tasks"]).issubset(record["region_labels"]) if regions else None)
        records.append(record)
        for label in record["body_labels"] + record["body_combinations"]:
            body_index.setdefault(label, []).append(record["path"])
        for label in record["region_labels"] + record["region_combinations"]:
            region_index.setdefault(label, []).append(record["path"])
    output = args.output or args.data.parent / "contact_labels.json"
    result = {
        "schema": "dexonomy.contact_candidates.v1",
        "data_dir": os.path.relpath(args.data.resolve(), output.parent.resolve()),
        "max_gap_m": args.max_gap_mm/1000, "max_penetration_m": args.max_penetration_mm/1000,
        "region_tolerance_m": args.region_tolerance_mm/1000,
        "regions_path": str(args.regions.resolve()) if args.regions else None,
        "meaning": "Near-contact candidates, not functional successes; stick includes head and mounts.",
        "body_index": body_index, "region_index": region_index, "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print(f"States: {len(records)}; multi-body: {sum(r['multitask_contact_candidate'] for r in records)}")
    print("Body counts:", {k: len(v) for k, v in body_index.items() if "+" not in k or k == "stick+trigger"})
    print("Region counts:", {k: len(v) for k, v in region_index.items() if "+" not in k})


if __name__ == "__main__":
    main()
