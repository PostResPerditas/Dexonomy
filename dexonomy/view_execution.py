"""Replay recorded execution states in MuJoCo or Viser (no re-simulation)."""

import argparse
import json
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import trimesh
from transforms3d.quaternions import quat2mat


def load_trace(path):
    with np.load(path, allow_pickle=False) as saved:
        trace = {k: saved[k].copy() for k in saved.files}
    trace["info"] = json.loads(str(trace["metadata"]))
    return trace


def set_frame(model, data, trace, frame):
    data.time = float(trace["time"][frame])
    for key in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
        getattr(data, key)[:] = trace[key][frame]
    mujoco.mj_forward(model, data)


def body_meshes(model, mode):
    result = {}
    for i in range(model.ngeom):
        collision = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
        if collision != (mode == "collision"):
            continue
        kind = model.geom_type[i]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[i]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            mesh = trimesh.Trimesh(model.mesh_vert[va:va+vn].copy(), model.mesh_face[fa:fa+fn].copy())
            if collision:
                mesh = mesh.convex_hull
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2*model.geom_size[i])
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=model.geom_size[i, 0], height=2*model.geom_size[i, 1])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(radius=model.geom_size[i, 0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(radius=model.geom_size[i, 0], height=2*model.geom_size[i, 1])
        else:
            raise ValueError(f"Unsupported replay geom type {kind}")
        mesh.vertices = mesh.vertices @ quat2mat(model.geom_quat[i]).T + model.geom_pos[i]
        result.setdefault(int(model.geom_bodyid[i]), []).append(mesh)
    return {bid: trimesh.util.concatenate(parts) for bid, parts in result.items()}


def style_mujoco(model):
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.65, 0.65, 0.65]
    model.vis.headlight.specular[:] = [0.15, 0.15, 0.15]
    model.vis.rgba.contactpoint[:] = [0.8, 0.05, 0.35, 1]
    model.vis.scale.contactwidth = 0.004
    model.vis.scale.contactheight = 0.002
    for i in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[i])).name
        color = ((44/255, 43/255, 184/255) if name.startswith("hand-") else
                 (213/255, 96/255, 89/255) if "button" in name or "trigger" in name else
                 (233/255, 233/255, 233/255))
        model.geom_rgba[i] = [*color, 1]
        model.geom_matid[i] = -1


def white_backdrop(scene, camera, extent):
    # Viewer-only plane behind the model, never part of collision or dynamics.
    azimuth, elevation = np.deg2rad([camera.azimuth, camera.elevation])
    normal = np.array([-np.cos(azimuth)*np.cos(elevation),
                       -np.sin(azimuth)*np.cos(elevation), -np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    rotation = np.column_stack([right, np.cross(normal, right), normal])
    distance = max(2*extent, camera.distance)
    mujoco.mjv_initGeom(scene.geoms[0], mujoco.mjtGeom.mjGEOM_PLANE,
                       np.array([10*distance, 10*distance, 0.001]),
                       camera.lookat - distance*normal, rotation.ravel(),
                       np.ones(4, dtype=np.float32))
    scene.geoms[0].emission = 1
    scene.geoms[0].specular = 0
    scene.geoms[0].category = mujoco.mjtCatBit.mjCAT_DECOR
    scene.ngeom = 1


def view_mujoco(model, paths, args):
    import mujoco.viewer
    style_mujoco(model)
    data = mujoco.MjData(model)
    state = dict(sample=args.sample, frame=0, play=True, reload=False)
    lock = threading.RLock()
    trace = load_trace(paths[state["sample"]])

    def key_callback(key):
        with lock:
            if key == ord(" "):
                state["play"] = not state["play"]
            elif key in (ord("N"), ord("P")):
                delta = 1 if key == ord("N") else -1
                state["sample"] = (state["sample"] + delta) % len(paths)
                state["reload"] = True
            elif key == ord("R"):
                state["frame"] = 0
            elif key in (ord(","), ord(".")):
                state["play"] = False
                state["frame"] = int(np.clip(state["frame"] + (1 if key == ord(".") else -1), 0, len(trace["time"])-1))

    set_frame(model, data, trace, 0)
    print("Replay: SPACE pause/play; N/P next/previous state; R rewind; ,/. frame step", flush=True)
    print(paths[state["sample"]], trace["info"], flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0, 0.035, 0.25]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.opt.geomgroup[3] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        while viewer.is_running():
            tick = time.perf_counter()
            with lock:
                if state["reload"]:
                    trace = load_trace(paths[state["sample"]])
                    state.update(frame=0, reload=False)
                    print(paths[state["sample"]], trace["info"], flush=True)
                frame = state["frame"]
                with viewer.lock():
                    set_frame(model, data, trace, frame)
                    white_backdrop(viewer.user_scn, viewer.cam, model.stat.extent)
                viewer.sync()
                period = float(np.median(np.diff(trace["time"]))) if len(trace["time"]) > 1 else 0.02
                stride = max(1, int(np.ceil(args.speed / (60*period))))
                if state["play"]:
                    state["frame"] = (frame + stride) % len(trace["time"])
            time.sleep(max(0.001, stride*period / args.speed - (time.perf_counter()-tick)))


def view_viser(model, paths, args):
    import viser
    data = mujoco.MjData(model)
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Dexonomy execution replay")
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", axes_length=0.06, axes_radius=0.001)
    sample = server.gui.add_slider("State", min=0, max=max(1, len(paths)-1), step=1,
                                   initial_value=args.sample, disabled=len(paths)==1)
    frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    play = server.gui.add_checkbox("Play", True)
    speed = server.gui.add_slider("Speed", min=0.1, max=2, step=0.1, initial_value=args.speed)
    mode = server.gui.add_dropdown("Geometry", ["visual", "collision"])
    contacts = server.gui.add_checkbox("Contacts", True)
    command_axes = server.gui.add_checkbox("Command frame", False)
    info = server.gui.add_markdown("")
    trace = None
    handles = {}
    current_mode = None
    lock = threading.RLock()
    axes = server.scene.add_frame("/command", axes_length=0.035, axes_radius=0.0008, visible=False)

    def render(_=None):
        nonlocal handles, current_mode
        if trace is None:
            return
        with lock, server.atomic():
            f = min(int(frame.value), len(trace["time"])-1)
            set_frame(model, data, trace, f)
            if mode.value != current_mode:
                for handle in handles.values():
                    handle.remove()
                handles = {}
                for bid, mesh in body_meshes(model, mode.value).items():
                    name = model.body(bid).name
                    color = ((100, 155, 225) if name.startswith("hand-") else
                             (235, 145, 65) if "button" in name or "trigger" in name else
                             (165, 170, 180) if "object_root" in name else (55, 160, 120))
                    handles[bid] = server.scene.add_mesh_simple(f"/model/{bid}", mesh.vertices, mesh.faces,
                        color=color, opacity=0.85 if name.startswith("hand-") else 1.0)
                current_mode = mode.value
            for bid, handle in handles.items():
                handle.position = data.xpos[bid]
                handle.wxyz = data.xquat[bid]
            start, end = trace["contact_offsets"][f:f+2]
            points = trace["contact_points"][start:end]
            server.scene.add_frame("/contacts", show_axes=False).remove()
            if len(points):
                server.scene.add_point_cloud("/contacts/points", points, (235, 70, 65),
                                            point_size=0.003, visible=contacts.value)
            axes.position = trace["command"][f, :3]
            axes.wxyz = trace["command"][f, 3:7]
            axes.visible = command_axes.value
            meta = trace["info"]
            scale = 180 / np.pi if meta["units"] == "rad" else 1000
            unit = "deg" if meta["units"] == "rad" else "mm"
            info.content = (f"**{meta['task']} / {meta['reason']}**\n\n`{paths[int(sample.value)].name}`"
                            f"\n\nPhase: **{trace['phase'][f]}**, t={trace['time'][f]:.2f} s"
                            f"\n\nProgress: **{trace['progress'][f]*scale:.3f} / {meta['threshold']*scale:.3f} {unit}**"
                            f"\n\nTarget contact: {bool(trace['target_contact'][f])}"
                            f"\n\nBase contact: {bool(trace['base_contact'][f])}")
            secondary = meta.get("secondary")
            if secondary:
                scale2 = 180 / np.pi if secondary["units"] == "rad" else 1000
                unit2 = "deg" if secondary["units"] == "rad" else "mm"
                info.content += (f"\n\n**Then: {secondary['task']}**"
                                 f"\n\nProgress: **{trace['secondary_progress'][f]*scale2:.3f} / "
                                 f"{secondary['threshold']*scale2:.3f} {unit2}**"
                                 f"\n\nTarget contact: {bool(trace['secondary_contact'][f])}")
            if "physics_valid" in meta:
                info.content += (f"\n\nPhysics valid: **{meta['physics_valid']}**"
                                 f"\n\nTask criterion: {meta['task_success']} ({meta['task_reason']})"
                                 f"\n\nMax penetration: {meta['max_penetration']*1000:.3f} mm")

    def choose(_=None):
        nonlocal trace
        with lock:
            trace = load_trace(paths[min(int(sample.value), len(paths)-1)])
            frame.value = 0
            frame.max = max(1, len(trace["time"])-1)
            render()

    sample.on_update(choose)
    for control in (frame, mode, contacts, command_axes):
        control.on_update(render)

    @server.on_client_connect
    def camera(client):
        client.camera.position = (0.45, -0.55, 0.5)
        client.camera.look_at = (0, 0.035, 0.25)
        client.camera.up_direction = (0, 0, 1)

    choose()
    print(f"Replay: http://127.0.0.1:{server.get_port()}", flush=True)
    try:
        while True:
            tick = time.perf_counter()
            with lock:
                period = (float(np.median(np.diff(trace["time"]))) if len(trace["time"]) > 1 else 0.02) / speed.value
                stride = max(1, int(np.ceil(1 / (60*period))))
                if play.value:
                    frame.value = (int(frame.value) + stride) % len(trace["time"])
            time.sleep(max(0.001, stride*period - (time.perf_counter()-tick)))
    except KeyboardInterrupt:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Execution output directory or one trajectory NPZ")
    parser.add_argument("--backend", choices=["mujoco", "viser"], default="viser")
    parser.add_argument("--result", choices=["all", "success", "failure"], default="all")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1)
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    folder = args.data if args.data.is_dir() else args.data.parent
    if args.data.is_file():
        paths = [args.data]
    else:
        summary = json.loads((folder / "summary.json").read_text())
        paths = [folder / r["trajectory"] for r in summary["results"]
                 if args.result == "all" or r["success"] == (args.result == "success")]
    if not paths or not 0 <= args.sample < len(paths) or not 0.1 <= args.speed <= 2:
        parser.error("No matching trajectory, invalid sample index or speed outside [0.1, 2]")
    model = mujoco.MjModel.from_binary_path(str(folder / "model.mjb"))
    (view_mujoco if args.backend == "mujoco" else view_viser)(model, paths, args)


if __name__ == "__main__":
    main()
