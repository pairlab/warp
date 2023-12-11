# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Contact
#
# Shows how to set up free rigid bodies with different shape types falling
# and colliding against each other and the ground using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

from pxr import UsdGeom, Usd

from pair_warp.utils.grad_utils import *
from pair_warp.utils.vis_utils import *
from pair_warp.utils.autograd import *
from pair_warp.utils.manipulability import *
from warp.envs import WarpEnv
from pair_warp.envs.environment import Environment, RenderMode
import cv2
import copy
import argparse

wp.init()

class Example:
    frame_dt = 1.0 / 60.0

    episode_duration = 6.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10 # 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0

    def __init__(self, stage=None, render=True):
        parser = argparse.ArgumentParser()
        parser.add_argument("--no_control", action='store_true', default=False)
        parser.add_argument("--use_ad", action='store_true', default=False)
        parser.add_argument("--move_obj", action='store_true', default=False)
        parser.add_argument("--num_balls", type=int, default=1)
        parser.add_argument("--goal_x", type=float, default=0.)
        parser.add_argument("--goal_y", type=float, default=2.)
        parser.add_argument("--goal_z", type=float, default=0.)
        self.args = parser.parse_args()

        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder._ground_params["has_ground_collision"] = False

        self.enable_rendering = render

        self.scale = 0.8

        self.ke = 5.e6 # 1.e+2 # 1.e+5
        self.kd = 1e3  # 250.0
        self.kf = 1e3 # 0.5
        # self.ke = 1.e+2
        # self.kd = 250.0
        # self.kf = 500.0
        self.mu = 1.0
        self.joint_target_ke = 1e8
        self.joint_target_kd = 1e6

        self.num_balls = self.args.num_balls
        self.obj_list = []

        # box (object that we want to manipulate)
        cube_0 = builder.add_body(origin=wp.transform((0.0, 2., 0.0), wp.quat_identity()))

        builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=0.5 * self.scale,
            hy=0.5 * self.scale,
            hz=0.5 * self.scale,
            body=cube_0,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            mu=self.mu,
            density=1.e2  # 1000.0 
        )
        self.obj_list.append('cube_0')

        self.ball_list = []
        self.ball_pos = {}

        # spheres (controllable objects)
        for i in range(self.num_balls):
            ball_pos = (wp.cos(wp.pi * 2 * i / self.num_balls)*0.9, 2., wp.sin(wp.pi * 2 * i / self.num_balls)*0.9)
            # ball_pos = (wp.sin(wp.pi * 2 * i / self.num_balls)*0.9, wp.cos(wp.pi * 2 * i / self.num_balls)*0.9 + 2., 0.0)
            index = builder.add_body(origin=wp.transform(ball_pos, wp.quat_identity()))
            self.ball_pos[index] = ball_pos

            builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.75 * self.scale,
                body=index,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                mu=self.mu,
                # density=5.e2  # 1000.0
            )

            self.ball_list.append(index)
            self.obj_list.append('ball_{}'.format(i))
            
        if self.args.no_control:
            # initial ball velocity
            builder.body_qd[1] = (0.0, 0.0, 0.0, -5.0, 0.0, 0.0)
        else: # enable dofs
            for sphere_body in self.ball_list:
                for i in range(3):
                    axis = np.zeros(3)
                    axis[i] = 1.0
                    builder.add_joint_prismatic(
                        parent=-1,  # Assuming free movement not relative to another body
                        child=sphere_body,
                        parent_xform=wp.transform_identity(),
                        child_xform=wp.transform_identity(),
                        mode=wp.sim.JOINT_MODE_TARGET_POSITION,
                        axis=axis,
                        target=self.ball_pos[sphere_body][i],
                        limit_lower=-5.0, limit_upper=5.0,
                        target_ke=self.joint_target_ke,  # Stiffness
                        target_kd=self.joint_target_kd,  # Damping
                        name="joint_{}_{}".format(sphere_body, i),
                    )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.model.integrator = self.integrator # had to add this line for some reason (Jeremy)

        # -----------------------
        # set up Usd renderer
        if (self.enable_rendering):
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=0.5)

        self.device = wp.get_device("cuda")

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        return wp.sim.Mesh(points, indices)

    # Krishnan's version
    def update(
        self,
        model,
        integrator,
        state_in,
        state_out,
        state_list=None,
        substeps=10,
        dt=1.0 / 60.0,
        ):

        # setup state_list if not provided
        if state_list is None or len(state_list) == 0:
            state_list = [model.state(requires_grad=True) for _ in range(substeps - 1)]

        # run forward simulate substeps with integrator
        for state, state_next in zip([state_in] + state_list[:-1], state_list[1:] + [state_out]):
            state.clear_forces()
            if model.ground:
                wp.sim.collide(model, state_in)
            state_next = integrator.simulate(
                model,
                state,
                state_next,
                dt,
            )

        return state_out

    def render(self, state, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(state)
        self.renderer.end_frame()

    def update_for_jacobian(self, start_state, substeps=10):
        end_state = self.model.state(requires_grad=True)
        self.update(model=self.model,
                    integrator=self.model.integrator,
                    state_in=start_state,
                    state_out=end_state,
                    state_list=None,
                    substeps=substeps,
                    dt=self.sim_dt,
                )
        return end_state
    
    @wp.kernel
    def control_body_delta(
        control_deltas: wp.array(dtype=wp.float32),
        joint_target: wp.array(dtype=wp.float32),
        action_dim: int,
    ):
        for i in range(action_dim):
            joint_target[i] = joint_target[i] + control_deltas[i]

    @wp.kernel
    def control_body_abs(
        control_target: wp.array(dtype=wp.float32),
        joint_target: wp.array(dtype=wp.float32),
        action_dim: int,
    ):
        for i in range(action_dim):
            joint_target[i] = control_target[i]

    def get_control_deltas(self, goal, curr_state, manipulability,  scale=.1, window_len=1):
        err = goal - curr_state
        print('goal error: ', err)
        action = err.T @ manipulability
        print('unnormalized action: ', action)
        if np.linalg.norm(action) > 1e-5:
            action = action / np.linalg.norm(action) * scale
        else:
            action = np.zeros_like(action)

        # averaging the last window_len actions
        self.action_list.append(action)
        if len(self.action_list) > window_len:
            self.action_list.pop(0)
        final_action = np.mean(self.action_list, axis=0)

        return wp.array(final_action, dtype=wp.float32)
    
    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        input_state = self.model.state(requires_grad = True)
        output_state = self.model.state(requires_grad = True)
        self.cube_goal = np.array([self.args.goal_x, self.args.goal_y, self.args.goal_z], dtype=np.float32)

        profiler = {}

        all_labels = get_manip_labels(obj_list=self.obj_list, modalities=['pos', 'quat', 'vel', 'force'], indices=range(len(self.obj_list)))

        input_modalities = ['pos']
        output_modalities = ['pos']
        # input_modalities = ['pos', 'force', 'vel']
        # output_modalities = ['pos', 'force', 'vel']
        
        input_indices = [i for i, obj in enumerate(self.obj_list) if 'ball' in obj]
        output_indices = [i for i, obj in enumerate(self.obj_list) if 'cube' in obj]

        manip_rows = get_manip_labels(obj_list=self.obj_list, modalities=output_modalities, indices=output_indices) # outputs (manually define these?)
        manip_cols = get_manip_labels(obj_list=self.obj_list, modalities=input_modalities, indices=[*output_indices, *input_indices]) # inputs (manually define these?)
        
        if self.args.move_obj:
            manip_rows, manip_cols = manip_cols, manip_rows

        # manip_rows, manip_cols = all_labels, all_labels

        manipulability_ad = np.zeros((len(manip_rows), len(manip_cols)), dtype=np.float32)
        manipulability_fd = np.zeros((len(manip_rows), len(manip_cols)), dtype=np.float32)
        action_dim = abs(len(manip_cols) - len(manip_rows))
        control_deltas = wp.zeros(action_dim, dtype=np.float32)
        self.action_list = []

        # simulate
        # input_state, output_state = output_state, input_state
        self.update(model=self.model,
                    integrator=self.model.integrator,
                    state_in=input_state,
                    state_out=output_state,
                    # state_list=self.state_list,
                    state_list=None,
                    substeps=self.sim_substeps,
                    dt=self.sim_dt,
                )

        with wp.ScopedTimer("simulate", detailed=False, print=False, active=False, dict=profiler):

            for f in range(0, self.episode_frames):
                self.sim_time += self.frame_dt

                if (self.enable_rendering):
                    with wp.ScopedTimer("render", active=False):
                        self.render(output_state)

                tape = wp.Tape()
                with tape:
                    if self.args.use_ad:
                        control_manipulability = manipulability_ad
                    else:
                        control_manipulability = manipulability_fd

                    if not self.args.no_control:
                        if self.args.move_obj:
                            control_deltas = self.get_control_deltas(self.cube_goal, output_state.body_q.numpy()[0][0:3], control_manipulability.T[:, 3:]) # ignoring the first 3 columns (cube position) for now TODO: simplify this
                        else:
                            control_deltas = self.get_control_deltas(self.cube_goal, output_state.body_q.numpy()[0][0:3], control_manipulability[:, 3:]) # ignoring the first 3 columns (cube position) for now TODO: simplify this

                        # control_deltas = wp.array([0,0.05,0, 0,0.05,0], dtype=wp.float32) 
                        wp.launch(self.control_body_delta, dim=1, inputs=[control_deltas, self.model.joint_target, action_dim], outputs=[], device="cuda")
                        print('control_deltas: ', control_deltas)

                    with wp.ScopedTimer("simulate", active=False):
                        input_state, output_state = output_state, input_state
                        output_state = self.update(model=self.model,
                                    integrator=self.model.integrator,
                                    state_in=input_state,
                                    state_out=output_state,
                                    # state_list=self.state_list,
                                    state_list=None,
                                    substeps=self.sim_substeps,
                                    dt=self.sim_dt,
                                )

                # manipulability_ad = get_manipulability_ad_composed(tape, input_state, output_state, input_modalities, output_modalities,
                #                                 all_labels, manip_rows, manip_cols)
                
                # setting stiffness and damping to 0 for the cube
                self.model.joint_target_ke = wp.array([0., 0., 0., 0., 0., 0.], dtype=wp.float32)
                self.model.joint_target_kd = wp.array([0., 0., 0., 0., 0., 0.], dtype=wp.float32)
                # self.model.joint_enabled = wp.zeros(action_dim, dtype=wp.int32)
                manipulability_fd = get_manipulability_fd_simplified(self.update_for_jacobian, input_state, eps=1e-3,
                                                                     input_indices=input_indices, output_indices=output_indices, all_labels=all_labels,
                                                                     manip_rows=manip_rows, manip_cols=manip_cols, input_modalities=input_modalities, output_modalities=output_modalities)
                self.model.joint_target_ke = wp.array([self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke], dtype=wp.float32)
                self.model.joint_target_kd = wp.array([self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd], dtype=wp.float32)
                # self.model.joint_enabled = wp.array(np.ones(action_dim), dtype=wp.int32)
                    
                if self.args.move_obj:
                    # visualizing the autodiff manipulability
                    print('manipulability_ad: ')
                    labeled_matrix_print(manipulability_ad.T, manip_cols, manip_rows, precision=6)
                    # print(manipulability_ad)
                    print('manipulability_fd: ')
                    labeled_matrix_print(manipulability_fd.T, manip_cols, manip_rows, precision=6)
                else:
                    # visualizing the autodiff manipulability
                    print('manipulability_ad: ')
                    labeled_matrix_print(manipulability_ad, manip_rows, manip_cols, precision=6)
                    # print(manipulability_ad)
                    print('manipulability_fd: ')
                    labeled_matrix_print(manipulability_fd, manip_rows, manip_cols, precision=6)

                # matrix_to_heatmap(manipulability_fd, manip_rows, manip_cols, title="heatmap (fd)", vis=True)
                # matrix_to_heatmap(manipulability_ad, manip_rows, manip_cols, title="heatmap (ad)", vis=True)
                # plot_eigenvalues(manipulability_ad, manip_rows, manip_cols, title="eigenvalues (ad)", vis=True)
                # visualize_unit_ball(manipulability_ad, manip_rows, manip_cols, title="unit ball (ad)", vis=True)
                # cv2.waitKey(1)  # Small delay to display images

                # print('checking:\n', (manipulability_fd - manipulability_ad) < 1e-4)

                print('cube pos: ', output_state.body_q.numpy()[0])
                print('cube vel: ', output_state.body_qd.numpy()[0])
                print('cube force: ', output_state.body_f.numpy()[0])

                print('\n\n')

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
robot = Example(stage, render=True)
robot.run()