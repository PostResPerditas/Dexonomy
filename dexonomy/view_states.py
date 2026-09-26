"""Browse generated grasp poses and saved hand-object contacts in Viser."""

import argparse
from collections import Counter
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import viser

from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_VisEnv
from dexonomy.util.file_util import load_scene_cfg
from dexonomy.label_states import summarize_contacts
from dexonomy.util.task_region import TaskRegions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(
        "output/joystick006_margin_fixed_tianyi_right/grasp_data"))
    parser.add_argument("--hand", default="assets/hand/tianyi_right/right.xml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--regions", type=Path)
    args = parser.parse_args()
    paths = sorted(args.data.rglob("*.npy"))
    if not paths:
        parser.error(f"No states in {args.data}")
    states = [np.load(p, allow_pickle=True).item() for p in paths]
    scenes = sorted({s["scene_path"] for s in states})
    regions = TaskRegions(args.regions) if args.regions else None
    if regions:
        for scene_path in scenes:
            regions.check_scene(load_scene_cfg(scene_path))
    summaries = [summarize_contacts(s, regions=regions) for s in states]
    combinations = sorted({c for s in summaries for c in s["body_combinations"]})
    parts = sorted({str(b) for s in states for b in s["ho_c"]["bn2"]})
    palette = [(45, 170, 130), (235, 110, 65), (220, 175, 40),
               (180, 100, 195), (50, 175, 200), (220, 85, 135),
               (130, 170, 65), (150, 135, 105)]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(parts)}
    counts = Counter(b for s in states for b in set(s["ho_c"]["bn2"]))
    print(f"Loaded {len(states)} states. Recorded near-contact state counts:", flush=True)
    for name, count in counts.most_common():
        print(f"  {name}: {count}", flush=True)

    server = viser.ViserServer(host=args.host, port=args.port, label="Dexonomy states")
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", axes_length=0.08, axes_radius=0.001)
    scene_choice = server.gui.add_dropdown("Scene", scenes, initial_value=scenes[0])
    part = server.gui.add_dropdown("Contact body", ["All"] + parts)
    combination = server.gui.add_dropdown("Body combination", ["All", "Multi-body"] + combinations)
    with server.gui.add_folder("Required task regions", expand_by_default=False):
        task_checks = {name: server.gui.add_checkbox(name.replace("_", " "), False)
                       for name in (regions.names if regions else [])}
    exclude_base = server.gui.add_checkbox("Exclude base contacts", False)
    region_tolerance = server.gui.add_slider("Region distance (mm)", min=0, max=10, step=0.5, initial_value=4)
    show_regions = server.gui.add_checkbox("Task region cloud", False)
    tolerance = server.gui.add_slider("Max gap (mm)", min=0, max=2, step=0.1, initial_value=2)
    index = server.gui.add_slider("State", min=0, max=max(1, len(states)-1), step=1, initial_value=0, order=0)
    mode = server.gui.add_dropdown("Geometry", ["visual", "collision"], initial_value="visual")
    opacity = server.gui.add_slider("Hand opacity", min=0.1, max=1, step=0.1, initial_value=0.7)
    show_points = server.gui.add_checkbox("Contact points", True)
    show_normals = server.gui.add_checkbox("Contact normals", False)
    show_labels = server.gui.add_checkbox("Contact labels", False)
    show_all = server.gui.add_checkbox("All-state contact cloud", False)
    status = server.gui.add_markdown("")
    table = server.gui.add_markdown("")
    lock = threading.RLock()
    cache = {}
    meshes = {}
    current_key = None
    filtered = []

    def contact_mask(state):
        contact = state["ho_c"]
        distances = np.asarray(contact["dist"])
        mask = (distances <= tolerance.value / 1000 + 1e-9) & (distances >= -1e-9)
        if part.value != "All":
            mask &= np.asarray(contact["bn2"]) == part.value
        return mask

    def clear_group(name):
        server.scene.add_frame(name, show_axes=False).remove()

    def render(_=None):
        nonlocal current_key, meshes
        with lock, server.atomic():
            clear_group("/contacts")
            clear_group("/aggregate")
            clear_group("/regions")
            if not filtered:
                clear_group("/model")
                current_key = None
                status.content = "**0 matching states**"
                table.content = ""
                return
            state_id = filtered[min(int(index.value), len(filtered)-1)]
            state = states[state_id]
            key = (state["scene_path"], mode.value)
            if key not in cache:
                cache[key] = MuJoCo_VisEnv(
                    HandCfg(xml_path=args.hand, freejoint=True),
                    load_scene_cfg(key[0]), sim_cfg=MuJoCo_OptCfg(), vis_mode=key[1])
            env = cache[key]
            if current_key != key:
                clear_group("/model")
                meshes = {}
                for name in env.get_all_body_names():
                    mesh = env.get_body_mesh(name)
                    hand = name.startswith(tuple(env.get_prefix("hand")))
                    meshes[name] = server.scene.add_mesh_simple(
                        f"/model/{name}", mesh.vertices, mesh.faces,
                        color=(100, 155, 225) if hand else colors.get(name, (175, 180, 185)),
                        opacity=opacity.value if hand else 1.0)
                current_key = key
            rotations, positions = env.forward_kinematics(state["grasp_qpos"][0])
            for name, handle in meshes.items():
                body_id = env.get_body_id(name)
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, rotations[body_id].astype(float).reshape(-1))
                handle.wxyz = quat
                handle.position = positions[body_id]
                if name.startswith(tuple(env.get_prefix("hand"))):
                    handle.opacity = opacity.value
            contact = state["ho_c"]
            mask = contact_mask(state)
            points = np.asarray(contact["pos"])[mask].reshape(-1, 3)
            normals = np.asarray(contact["normal"])[mask].reshape(-1, 3)
            selected = np.flatnonzero(mask)
            rgb = np.array([colors[contact["bn2"][i]] for i in selected], dtype=np.uint8).reshape(-1, 3)
            if len(points):
                server.scene.add_point_cloud("/contacts/points", points, rgb,
                    point_size=0.003, point_shape="circle", visible=show_points.value)
                server.scene.add_line_segments("/contacts/normals",
                    np.stack([points, points + normals * 0.012], axis=1),
                    np.repeat(rgb[:, None, :], 2, axis=1), thickness=0.0007,
                    visible=show_normals.value)
                for j, i in enumerate(selected):
                    server.scene.add_label(f"/contacts/label_{j}",
                        f"{contact['bn2'][i]} ({contact['dist'][i]*1000:.2f} mm)",
                        position=points[j], visible=show_labels.value)
            if show_all.value:
                all_points, all_colors = [], []
                for sid in filtered:
                    c = states[sid]["ho_c"]
                    m = contact_mask(states[sid])
                    all_points.extend(np.asarray(c["pos"])[m])
                    all_colors.extend(colors[b] for b in np.asarray(c["bn2"])[m])
                if all_points:
                    server.scene.add_point_cloud("/aggregate/points", np.array(all_points),
                        np.array(all_colors, dtype=np.uint8), point_size=0.0015)
            status.content = (f"**{int(index.value)+1} / {len(filtered)} matching states** "
                              f"({len(states)} total)\n\n`{paths[state_id].relative_to(args.data)}`"
                              f"\n\nBody: {', '.join(summaries[state_id]['body_labels'])}"
                              f"\n\nRegion: {', '.join(summaries[state_id]['region_labels'])}")
            if regions and show_regions.value:
                names = [name for name, check in task_checks.items() if check.value] or regions.names
                for name in names:
                    points_world = regions.points[name] @ regions.rotation.T + regions.frame_pose[:3]
                    server.scene.add_point_cloud(f"/regions/{name}", points_world,
                        colors.get(regions.bodies[name], (245, 180, 55)), point_size=0.001)
            rows = ["| Hand body | Object body | Gap (mm) |", "|---|---|---:|"]
            rows.extend(f"| {contact['bn1'][i]} | {contact['bn2'][i]} | {contact['dist'][i]*1000:.3f} |"
                        for i in selected)
            table.content = "\n".join(rows)

    def refilter(_=None):
        nonlocal filtered, summaries
        with lock:
            summaries = [summarize_contacts(s, tolerance.value/1000, regions=regions,
                         region_tolerance=region_tolerance.value/1000) for s in states]
            filtered = [i for i, s in enumerate(states)
                        if s["scene_path"] == scene_choice.value
                        and (part.value == "All" or np.any(contact_mask(s)))
                        and (combination.value == "All"
                             or combination.value == "Multi-body" and summaries[i]["multitask_contact_candidate"]
                             or combination.value in summaries[i]["body_combinations"])
                        and all(name in summaries[i]["region_labels"]
                                for name, check in task_checks.items() if check.value)
                        and (not exclude_base.value or not summaries[i]["base_near_contact"])]
            index.value = 0
            index.max = max(1, len(filtered)-1)
            index.disabled = len(filtered) <= 1
            render()

    for control in (scene_choice, part, tolerance, combination, exclude_base, region_tolerance, *task_checks.values()):
        control.on_update(refilter)
    for control in (index, mode, opacity, show_points, show_normals, show_labels, show_all, show_regions):
        control.on_update(render)

    @server.on_client_connect
    def set_camera(client):
        client.camera.position = (0.4, -0.45, 0.42)
        client.camera.look_at = (0.0, 0.035, 0.22)
        client.camera.up_direction = (0.0, 0.0, 1.0)

    refilter()
    print(f"Viewer: http://{args.host}:{server.get_port()}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
