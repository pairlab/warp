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
        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        self.num_bodies = 8
        self.scale = 0.8
        self.ke = 1.e5 # 1.e+2
        self.kd = 1000.0
        self.kf = 0.5

        # boxes
        cube1 = builder.add_body(origin=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))

        s1 = builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=0.5 * self.scale,
            hy=0.5 * self.scale,
            hz=0.5 * self.scale,
            body=cube1,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
        )

        # spheres
        ball1 = builder.add_body(origin=wp.transform((1.0, 1.0, 0.0), wp.quat_identity()))

        builder.add_shape_sphere(
            pos=(0.0, 0.0, 0.0), radius=0.75 * self.scale, body=ball1, ke=self.ke, kd=self.kd, kf=self.kf
        )


        ball2 = builder.add_body(origin=wp.transform((-0.5, 1.0, 0.866), wp.quat_identity()))

        builder.add_shape_sphere(
            pos=(0.0, 0.0, 0.0), radius=0.75 * self.scale, body=ball2, ke=self.ke, kd=self.kd, kf=self.kf
        )

        ball3 = builder.add_body(origin=wp.transform((-0.5, 1.0, -0.866), wp.quat_identity()))

        builder.add_shape_sphere(
            pos=(0.0, 0.0, 0.0), radius=0.75 * self.scale, body=ball3, ke=self.ke, kd=self.kd, kf=self.kf
        )

        # initial ball velocity
        builder.body_qd[1] = (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        # ground box
        builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=10.0*self.scale,
            hy=0.1*self.scale,
            hz=10.0*self.scale,
            body=-1,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf)
        
        for sphere_body in [ball1, ball2, ball3]:
            builder.add_joint_prismatic(
                parent=-1,  # Assuming free movement not relative to another body
                child=sphere_body,
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform_identity(),
                mode=wp.sim.JOINT_MODE_TARGET_POSITION,
                axis=(0., 1., 0.),
                limit_lower=-5.0, limit_upper=5.0,
                target_ke=1e5,  # Stiffness
                target_kd=10.0  # Damping
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

        # parser = argparse.ArgumentParser()
        # parser.add_argument("--ad", type=str, default=None, help="Autodiff variant")
        # parser.add_argument("--vis_ad", type=bool, default=False)
        # parser.add_argument("--fd", type=str, default=None, help="Finite difference variant")

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        return wp.sim.Mesh(points, indices)

    # dynamically building state_list
    def update(self):
        self.state_list = [self.state_0]

        # for state_in, state_out in self.state_pairs:
        for i in range(self.sim_substeps):
            self.state_list[-1].clear_forces()
            wp.sim.collide(self.model, self.state_list[-1])
            next_state = self.model.state(requires_grad = True)
            self.integrator.simulate(self.model, self.state_list[-1], next_state, self.sim_dt)

            if i < self.sim_substeps - 1:
                self.state_list.append(next_state)

        # self.state_0 = self.state_list[1]
        self.state_0 = self.state_list[-2]
        self.state_1 = self.state_list[-1]

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        # self.renderer.render(self.state_list[-1])
        # self.renderer.render(self.curr_state)
        self.renderer.end_frame()

    # single substep version for finite difference
    def update_for_jacobian(self, start_state):
        start_state.clear_forces()
        wp.sim.collide(self.model, start_state)

        end_state = self.model.state(requires_grad = True)
        self.integrator.simulate(model=self.model, state_in=start_state, state_out=end_state, dt=self.sim_dt, requires_grad=True)

        return end_state

    def select_rows_cols(self, arr, rows, cols):
        arr = arr[rows, :]
        arr = arr[:, cols]

        return arr
    
    @wp.kernel
    def control_balls(
        time: float,
        joint_target: wp.array(dtype=float),
    ):
        # set the target position of the revolute joint
        # tid = wp.tid()
        # joint_target[tid] = wp.sin(time*2.0 + float(tid))*0.5
        joint_target[1] = wp.sin(time*10.0)*0.5


    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state(requires_grad = True)
        self.state_1 = self.model.state(requires_grad = True)

        self.state_list = [self.model.state(requires_grad = True) for i in range(self.sim_substeps - 1)]
        self.state_list.append(self.state_1)

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_1)

        profiler = {}

        # simulate
        self.update()

        with wp.ScopedTimer("simulate", detailed=False, print=False, active=False, dict=profiler):

            for f in range(0, self.episode_frames):
                self.sim_time += self.frame_dt

                if (self.enable_rendering):
                    with wp.ScopedTimer("render", active=False):
                        self.render()

                # cube_pose_np = self.state_1.body_q.numpy()[0]
                # cube_vel_np = self.state_1.body_qd.numpy()[0]
                # ball_pose_np = self.state_1.body_q.numpy()[1]
                # ball_vel_np = self.state_1.body_qd.numpy()[1]

                # manipulability_fd = wp.zeros((14,14), dtype=wp.float32, device=wp.get_device("cuda"), requires_grad=True)
                manipulability_fd = wp.zeros((3,3), dtype=wp.float32, device=wp.get_device("cuda"), requires_grad=True)

                tape = wp.Tape()
                with tape:
                    wp.launch(self.control_balls, dim=1, inputs=[self.sim_time, self.model.joint_q], outputs=[], device="cuda")
                    # wp.launch(self.manipulability_kernel, dim=1, inputs=[self.state_0.body_q, self.state_1.body_q], outputs=[manipulability_fd], device="cuda")
                    # wp.launch(self.manipulability_fd_kernel(self.state_0.body_q, self.state_1.body_q, manipulability_fd, self.simulate_step, self.model, self.sim_dt), dim=1, inputs=[self.state_0.body_q, self.state_1.body_q], outputs=[manipulability_fd], device="cuda")
                    with wp.ScopedTimer("simulate", active=False):
                        self.update()
   
                        
                # check_backward_pass(tape=tape)

                # manipulability_ad = self.get_manipulability_ad(tape, dim=14)
                manipulability_ad = get_manipulability_ad(tape, dim=28, input_state=self.state_0, output_state=self.state_1)
                
                # # check_tape_safety(self.get_manipulability_fd_tape, inputs=[tape, 14])
                # # manipulability_fd = self.get_manipulability_fd(self.state_1, dim=7, eps=1e-5, input_index=1, output_index=0)
                # manipulability_fd_0 = get_manipulability_fd(self.update_for_jacobian, self.state_1, dim=7, eps=1e-4, input_index=0, output_index=0)
                # manipulability_fd_1 = get_manipulability_fd(self.update_for_jacobian, self.state_1, dim=7, eps=1e-4, input_index=1, output_index=0)
                # manipulability_fd_2 = get_manipulability_fd(self.update_for_jacobian, self.state_1, dim=7, eps=1e-4, input_index=0, output_index=1)
                # manipulability_fd_3 = get_manipulability_fd(self.update_for_jacobian, self.state_1, dim=7, eps=1e-4, input_index=1, output_index=1)

                # # concatenating them together like [[[0,0], [1,0]], [[0,1], [1,1]]], where indices are [input_index, output_index]
                # manipulability_fd = np.concatenate((np.concatenate((manipulability_fd_0, manipulability_fd_1), axis=1), np.concatenate((manipulability_fd_2, manipulability_fd_3), axis=1)), axis=0)

                manipulability_fd = get_manipulability_fd_composed(self.update_for_jacobian, self.state_1, eps=1e-2, input_indices=[0, 1, 2, 3], output_indices=[0, 1, 2, 3])

                all_labels = ['cube_x', 'cube_y', 'cube_z','cube_i', 'cube_j', 'cube_k', 'cube_w',
                            'ball1_x', 'ball1_y', 'ball1_z', 'ball1_i', 'ball1_j', 'ball1_k', 'ball1_w',
                            'ball2_x', 'ball2_y', 'ball2_z', 'ball2_i', 'ball2_j', 'ball2_k', 'ball2_w',
                            'ball3_x', 'ball3_y', 'ball3_z', 'ball3_i', 'ball3_j', 'ball3_k', 'ball3_w',
                            ]

                manip_rows = ['cube_x', 'cube_y', 'cube_z']
                manip_cols = ['cube_x', 'cube_y', 'cube_z', 'ball1_x', 'ball1_y', 'ball1_z', 'ball2_x', 'ball2_y', 'ball2_z', 'ball3_x', 'ball3_y', 'ball3_z']
                # manip_rows, manip_cols = all_labels, all_labels
                
                manipulability_ad = self.select_rows_cols(manipulability_ad,
                                    rows=[all_labels.index(label) for label in manip_rows],
                                    cols=[all_labels.index(label) for label in manip_cols]
                                    )
                
                manipulability_fd = self.select_rows_cols(manipulability_fd,
                                    rows=[all_labels.index(label) for label in manip_rows],
                                    cols=[all_labels.index(label) for label in manip_cols]
                                    )

                # visualizing the autodiff manipulability
                print('manipulability_ad: ')
                # print(manipulability_ad)
                labeled_matrix_print(manipulability_ad, manip_rows, manip_cols)
                print('manipulability_fd: ')
                # print(manipulability_fd)
                labeled_matrix_print(manipulability_fd, manip_rows, manip_cols)
                # matrix_to_heatmap(manipulability_ad, manip_rows, manip_cols, title="heatmap (ad)", vis=True)
                # plot_eigenvalues(manipulability_ad, manip_rows, manip_cols, title="eigenvalues (ad)", vis=True)
                # visualize_unit_ball(manipulability_ad, manip_rows, manip_cols, title="unit ball (ad)", vis=True)
                # cv2.waitKey(1)  # Small delay to display the image

                # print('determinant: ', np.linalg.det(manipulability_ad.numpy()))
                # print('trace: ', np.trace(manipulability_ad.numpy()))
                # print('eigenvalues: ', np.linalg.eigvals(manipulability_ad))
                # print('eigenvectors: ', np.linalg.eig(manipulability_ad))

                # # printing attributes of the state
                # for attr in dir(self.state_0):
                #     print("obj.%s = %r" % (attr, getattr(self.state_0, attr)))

                print('fd == ad:\n', (manipulability_fd - manipulability_ad) < 1e-2)
                print('\n')

                # self.control_balls()

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
robot = Example(stage, render=True)
robot.run()