# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import os

import xml.etree.ElementTree as ET

import warp as wp
from warp.sim.model import Mesh

from typing import Union


def parse_partnet_urdf(
    urdf_filename,
    builder,
    xform=wp.transform(),
    floating=False,
    base_joint: Union[dict, str] = None,
    density=1000.0,
    stiffness=100.0,
    damping=10.0,
    armature=0.0,
    shape_ke=1.0e4,
    shape_kd=1.0e3,
    shape_kf=1.0e2,
    shape_mu=0.25,
    shape_restitution=0.5,
    shape_thickness=0.0,
    limit_ke=100.0,
    limit_kd=10.0,
    scale=1.0,
    parse_visuals_as_colliders=False,
    enable_self_collisions=True,
    ignore_inertial_definitions=True,
    ensure_nonstatic_links=True,
    static_link_mass=1e-2,
    collapse_fixed_joints=False,
    continuous_joint_type="screw",
):
    file = ET.parse(urdf_filename)
    root = file.getroot()

    def parse_origin(element):
        if element is None or element.find("origin") is None:
            return wp.transform()
        origin = element.find("origin")
        xyz = origin.get("xyz") or "0 0 0"
        rpy = origin.get("rpy") or "0 0 0"
        xyz = [float(x) * scale for x in xyz.split()]
        rpy = [float(x) for x in rpy.split()]
        return wp.transform(xyz, wp.quat_rpy(*rpy))

    def parse_shapes(link, collisions, density, incoming_xform=None):
        # add geometry
        for collision in collisions:
            geo = collision.find("geometry")
            if geo is None:
                continue

            tf = parse_origin(collision)
            if incoming_xform is not None:
                tf = incoming_xform * tf

            for box in geo.findall("box"):
                size = box.get("size") or "1 1 1"
                size = [float(x) for x in size.split()]
                builder.add_shape_box(
                    body=link,
                    pos=tf.p,
                    rot=tf.q,
                    hx=size[0] * 0.5 * scale,
                    hy=size[1] * 0.5 * scale,
                    hz=size[2] * 0.5 * scale,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    restitution=shape_restitution,
                    thickness=shape_thickness,
                )

            for sphere in geo.findall("sphere"):
                builder.add_shape_sphere(
                    body=link,
                    pos=tf.p,
                    rot=tf.q,
                    radius=float(sphere.get("radius") or "1") * scale,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    restitution=shape_restitution,
                    thickness=shape_thickness,
                )

            for cylinder in geo.findall("cylinder"):
                builder.add_shape_capsule(
                    body=link,
                    pos=tf.p,
                    rot=tf.q,
                    radius=float(cylinder.get("radius") or "1") * scale,
                    half_height=float(cylinder.get("length") or "1") * 0.5 * scale,
                    density=density,
                    ke=shape_ke,
                    kd=shape_kd,
                    kf=shape_kf,
                    mu=shape_mu,
                    up_axis=2,  # cylinders in URDF are aligned with z-axis
                    restitution=shape_restitution,
                    thickness=shape_thickness,
                )

            for mesh in geo.findall("mesh"):
                filename = mesh.get("filename")
                if filename is None:
                    continue
                if filename.startswith("http://") or filename.startswith("https://"):
                    # download mesh
                    import requests
                    import tempfile
                    import shutil

                    with tempfile.TemporaryDirectory() as tmpdir:
                        # get filename extension
                        extension = os.path.splitext(filename)[1]
                        tmpfile = os.path.join(tmpdir, "mesh" + extension)
                        with requests.get(filename, stream=True) as r:
                            with open(tmpfile, "wb") as f:
                                shutil.copyfileobj(r.raw, f)
                        filename = tmpfile
                else:
                    filename = os.path.join(os.path.dirname(urdf_filename), filename)
                if not os.path.exists(filename):
                    import warnings

                    warnings.warn(f"Warning: mesh file {filename} does not exist")
                    continue

                import trimesh

                m = trimesh.load_mesh(filename)
                scaling = mesh.get("scale") or "1 1 1"
                scaling = np.array([float(x) * scale for x in scaling.split()])
                if hasattr(m, "geometry"):
                    # multiple meshes are contained in a scene
                    for geom in m.geometry.values():
                        vertices = np.array(geom.vertices, dtype=np.float32) * scaling
                        faces = np.array(geom.faces, dtype=np.int32)
                        mesh = Mesh(vertices, faces)
                        builder.add_shape_mesh(
                            body=link,
                            pos=tf.p,
                            rot=tf.q,
                            mesh=mesh,
                            density=density,
                            ke=shape_ke,
                            kd=shape_kd,
                            kf=shape_kf,
                            mu=shape_mu,
                            restitution=shape_restitution,
                            thickness=shape_thickness,
                        )
                else:
                    # a single mesh
                    vertices = np.array(m.vertices, dtype=np.float32) * scaling
                    faces = np.array(m.faces, dtype=np.int32)
                    mesh = Mesh(vertices, faces)
                    builder.add_shape_mesh(
                        body=link,
                        pos=tf.p,
                        rot=tf.q,
                        mesh=mesh,
                        density=density,
                        ke=shape_ke,
                        kd=shape_kd,
                        kf=shape_kf,
                        mu=shape_mu,
                        restitution=shape_restitution,
                        thickness=shape_thickness,
                    )

    # maps from link name -> link index
    link_index = {}

    builder.add_articulation()

    start_shape_count = len(builder.shape_geo_type)

    # add links
    for i, urdf_link in enumerate(root.findall("link")):
        if parse_visuals_as_colliders:
            colliders = urdf_link.findall("visual")
        else:
            colliders = urdf_link.findall("collision")

        name = urdf_link.get("name")
        link = builder.add_body(origin=wp.transform_identity(), armature=armature, name=name)

        # add ourselves to the index
        link_index[name] = link

        parse_shapes(link, colliders, density=density)
        m = builder.body_mass[link]
        if not ignore_inertial_definitions and urdf_link.find("inertial") is not None:
            # overwrite inertial parameters if defined
            inertial = urdf_link.find("inertial")
            inertial_frame = parse_origin(inertial)
            com = inertial_frame.p
            I_m = np.zeros((3, 3))
            I_m[0, 0] = float(inertial.find("inertia").get("ixx") or "0") * scale**2
            I_m[1, 1] = float(inertial.find("inertia").get("iyy") or "0") * scale**2
            I_m[2, 2] = float(inertial.find("inertia").get("izz") or "0") * scale**2
            I_m[0, 1] = float(inertial.find("inertia").get("ixy") or "0") * scale**2
            I_m[0, 2] = float(inertial.find("inertia").get("ixz") or "0") * scale**2
            I_m[1, 2] = float(inertial.find("inertia").get("iyz") or "0") * scale**2
            I_m[1, 0] = I_m[0, 1]
            I_m[2, 0] = I_m[0, 2]
            I_m[2, 1] = I_m[1, 2]
            rot = wp.quat_to_matrix(inertial_frame.q)
            I_m = rot @ I_m
            m = float(inertial.find("mass").get("value") or "0")
            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m
            builder.body_com[link] = com
            builder.body_inertia[link] = I_m
            builder.body_inv_inertia[link] = np.linalg.inv(I_m)
        if m == 0.0 and ensure_nonstatic_links:
            # set the mass to something nonzero to ensure the body is dynamic
            m = static_link_mass
            # cube with side length 0.5
            I_m = np.eye(3) * m / 12.0 * (0.5 * scale) ** 2 * 2.0
            builder.body_mass[link] = m
            builder.body_inv_mass[link] = 1.0 / m
            builder.body_inertia[link] = I_m
            builder.body_inv_inertia[link] = np.linalg.inv(I_m)

    end_shape_count = len(builder.shape_geo_type)

    # find joints per body
    body_children = {name: [] for name in link_index.keys()}
    # mapping from parent, child link names to joint
    parent_child_joint = {}

    joints = []

    for joint in root.findall("joint"):
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        body_children[parent].append(child)
        joint_data = {
            "name": joint.get("name"),
            "parent": parent,
            "child": child,
            "type": joint.get("type"),
            "origin": parse_origin(joint),
            "friction": 0.0,
            "limit_lower": -1.0e6,
            "limit_upper": 1.0e6,
        }
        if joint.find("axis") is not None:
            joint_data["axis"] = joint.find("axis").get("xyz")
            joint_data["axis"] = np.array([float(x) for x in joint_data["axis"].split()])
        if joint.find("dynamics") is not None:
            dynamics = joint.find("dynamics")
            joint_data["friction"] = float(dynamics.get("friction") or "0")
        if joint.find("limit") is not None:
            limit = joint.find("limit")
            joint_data["limit_lower"] = float(limit.get("lower") or "-1e6")
            joint_data["limit_upper"] = float(limit.get("upper") or "1e6")
        if joint.find("mimic") is not None:
            mimic = joint.find("mimic")
            joint_data["mimic_joint"] = mimic.get("joint")
            joint_data["mimic_multiplier"] = float(mimic.get("multiplier") or "1")
            joint_data["mimic_offset"] = float(mimic.get("offset") or "0")

        parent_child_joint[(parent, child)] = joint_data
        joints.append(joint_data)

    # topological sorting of joints because the FK solver will resolve body transforms
    # in joint order and needs the parent link transform to be resolved before the child
    visited = {name: False for name in link_index.keys()}
    sorted_joints = []

    # depth-first search
    def dfs(joint):
        link = joint["child"]
        if visited[link]:
            return
        visited[link] = True

        for child in body_children[link]:
            if not visited[child]:
                dfs(parent_child_joint[(link, child)])

        sorted_joints.insert(0, joint)

    # start DFS from each unvisited joint
    for joint in joints:
        if not visited[joint["parent"]]:
            dfs(joint)

    # add base joint
    if len(sorted_joints) > 0:
        base_link_name = sorted_joints[0]["parent"]
    else:
        base_link_name = next(iter(link_index.keys()))
    root = link_index[base_link_name]
    if base_joint is not None:
        # in case of a given base joint, the position is applied first, the rotation only
        # after the base joint itself to not rotate its axis
        base_parent_xform = wp.transform(xform.p, wp.quat_identity())
        base_child_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_inverse(xform.q))
        if isinstance(base_joint, str):
            axes = base_joint.lower().split(",")
            axes = [ax.strip() for ax in axes]
            linear_axes = [ax[-1] for ax in axes if ax[0] in {"l", "p"}]
            angular_axes = [ax[-1] for ax in axes if ax[0] in {"a", "r"}]
            axes = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }
            builder.add_joint_d6(
                linear_axes=[wp.sim.JointAxis(axes[a]) for a in linear_axes],
                angular_axes=[wp.sim.JointAxis(axes[a]) for a in angular_axes],
                parent_xform=base_parent_xform,
                child_xform=base_child_xform,
                parent=-1,
                child=root,
                name="base_joint",
            )
        elif isinstance(base_joint, dict):
            base_joint["parent"] = -1
            base_joint["child"] = root
            base_joint["parent_xform"] = base_parent_xform
            base_joint["child_xform"] = base_child_xform
            base_joint["name"] = "base_joint"
            builder.add_joint(**base_joint)
        else:
            raise ValueError(
                "base_joint must be a comma-separated string of joint axes or a dict with joint parameters"
            )
    elif floating:
        builder.add_joint_free(root, name="floating_base")

        # set dofs to transform
        start = builder.joint_q_start[root]

        builder.joint_q[start + 0] = xform.p[0]
        builder.joint_q[start + 1] = xform.p[1]
        builder.joint_q[start + 2] = xform.p[2]

        builder.joint_q[start + 3] = xform.q[0]
        builder.joint_q[start + 4] = xform.q[1]
        builder.joint_q[start + 5] = xform.q[2]
        builder.joint_q[start + 6] = xform.q[3]
    else:
        builder.add_joint_fixed(-1, root, parent_xform=xform, name="fixed_base")

    if isinstance(stiffness, float):
        joint_stiffness = [stiffness for _ in range(len(sorted_joints))]
        joint_damping = [damping for _ in range(len(sorted_joints))]
    else:
        stiffness, joint_stiffness = stiffness[0], stiffness[1:]
        damping, joint_damping = damping[0], damping[1:]

    # add joints, in topological order starting from root body
    for joint in sorted_joints:
        parent = link_index[joint["parent"]]
        child = link_index[joint["child"]]
        if child == -1:
            # we skipped the insertion of the child body
            continue

        lower = joint["limit_lower"]
        upper = joint["limit_upper"]

        parent_xform = joint["origin"]
        child_xform = wp.transform_identity()

        joint_mode = wp.sim.JOINT_MODE_LIMIT

        if stiffness > 0.0:
            joint_mode = wp.sim.JOINT_MODE_TARGET_POSITION

        joint_params = dict(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            name=joint["name"],
        )
        if joint["type"] == "continuous" and continuous_joint_type is not "revolute":
            if continuous_joint_type == "screw":  # sets up a screw joint
                print("setting screw continuous upper/lower to +/-pi")
                upper = 2 * np.pi
                lower = -2 * np.pi
                axes = ["ry", "py"]
            elif continuous_joint_type == "prismatic":
                axes = ["py"]
            axes = [ax.strip() for ax in axes]
            linear_axes = [ax[-1] for ax in axes if ax[0] in {"l", "p"}]
            angular_axes = [ax[-1] for ax in axes if ax[0] in {"a", "r"}]
            axes = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }
            builder.add_joint_d6(
                linear_axes=[wp.sim.JointAxis(axes[a], limit_lower=0.0, limit_upper=1.0) for a in linear_axes],
                angular_axes=[wp.sim.JointAxis(axes[a], limit_lower=lower, limit_upper=upper) for a in angular_axes],
                **joint_params,
            )
        elif joint["type"] == "revolute" or (joint["type"] == "continuous" and continuous_joint_type == "revolute"):
            if joint["type"] == "continuous":
                print("setting screw continuous upper/lower to +/-pi")
                upper = 2 * np.pi
                lower = -2 * np.pi
            builder.add_joint_revolute(
                axis=joint["axis"],
                target_ke=stiffness,
                target_kd=damping,
                limit_lower=lower,
                limit_upper=upper,
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                mode=joint_mode,
                **joint_params,
            )
            if len(joint_stiffness) > 0:
                stiffness, joint_stiffness = joint_stiffness[0], joint_stiffness[1:]
                damping, joint_damping = joint_damping[0], joint_damping[1:]
        elif joint["type"] == "prismatic":
            builder.add_joint_prismatic(
                axis=joint["axis"],
                target_ke=stiffness,
                target_kd=damping,
                limit_lower=lower * scale,
                limit_upper=upper * scale,
                limit_ke=limit_ke,
                limit_kd=limit_kd,
                mode=joint_mode,
                **joint_params,
            )
            if len(joint_stiffness) > 0:
                stiffness, joint_stiffness = joint_stiffness[0], joint_stiffness[1:]
                damping, joint_damping = joint_damping[0], joint_damping[1:]
        elif joint["type"] == "fixed":
            builder.add_joint_fixed(**joint_params)
        elif joint["type"] == "floating":
            builder.add_joint_free(**joint_params)
        elif joint["type"] == "planar":
            # find plane vectors perpendicular to axis
            axis = np.array(joint["axis"])
            axis /= np.linalg.norm(axis)

            # create helper vector that is not parallel to the axis
            helper = np.array([1, 0, 0]) if np.allclose(axis, [0, 1, 0]) else np.array([0, 1, 0])

            u = np.cross(helper, axis)
            u /= np.linalg.norm(u)

            v = np.cross(axis, u)
            v /= np.linalg.norm(v)

            builder.add_joint_d6(
                linear_axes=[
                    wp.sim.JointAxis(
                        u, limit_lower=lower * scale, limit_upper=upper * scale, limit_ke=limit_ke, limit_kd=limit_kd
                    ),
                    wp.sim.JointAxis(
                        v, limit_lower=lower * scale, limit_upper=upper * scale, limit_ke=limit_ke, limit_kd=limit_kd
                    ),
                ],
                **joint_params,
            )
        else:
            raise Exception("Unsupported joint type: " + joint["type"])

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            for j in range(i + 1, end_shape_count):
                builder.shape_collision_filter_pairs.add((i, j))

    if collapse_fixed_joints:
        builder.collapse_fixed_joints()
