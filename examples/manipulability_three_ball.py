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

        self.scale = 0.8
        self.ke = 2.e3 # 1.e+2 # 1.e+5
        self.kd = 10000.0
        self.kf = 1000 # 0.5
        self.num_balls = 3

        # boxes
        cube1 = builder.add_body(origin=wp.transform((0.0, 0.5, 0.0), wp.quat_identity()))

        s1 = builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=0.5 * self.scale,
            hy=0.5 * self.scale,
            hz=0.5 * self.scale,
            body=cube1,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            density=1.e3  # 1000.0 
            # density=10.  # 1000.0 TODO: change back
        )

        self.ball_list = []
        self.ball_pos = {}

        for i in range(self.num_balls):
            # index = builder.add_body(origin=wp.transform((wp.cos(wp.pi * 2 * i / self.num_balls), 0.5, wp.sin(wp.pi * 2 * i / self.num_balls)), wp.quat_identity()))
            ball_pos = (wp.cos(wp.pi * 2 * i / self.num_balls)*0.9, 0.5, wp.sin(wp.pi * 2 * i / self.num_balls)*0.9)
            index = builder.add_body(origin=wp.transform(ball_pos, wp.quat_identity()))
            self.ball_pos[index] = ball_pos

            builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0), radius=0.75 * self.scale, body=index, ke=self.ke, kd=self.kd, kf=self.kf
            )

            self.ball_list.append(index)

        # initial ball velocity
        # builder.body_qd[1] = (0.0, 0.0, 0.0, -5.0, 0.0, 0.0)


        # # adding a small immobile cube to represent the goal
        # cube_goal = builder.add_body(origin=wp.transform((4.0, 4.0, 4.0), wp.quat_identity()))

        # builder.add_shape_box(
        #     pos=(0.0, 0.0, 0.0),
        #     hx=0.5 * self.scale,
        #     hy=0.5 * self.scale,
        #     hz=0.5 * self.scale,
        #     body=cube_goal,
        #     # ke=self.ke,
        #     # kd=self.kd,
        #     # kf=self.kf,
        #     ke=0.0,
        #     kd=0.0,
        #     kf=0.0,
        #     # is_solid=False,
        # )
            
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

        # for sphere_body in [ball1, ball2, ball3]:
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
                    target_ke=1e8,  # Stiffness
                    target_kd=1e6  # Damping
                )

        # axes = {
        #         "x": [1.0, 0.0, 0.0],
        #         "y": [0.0, 1.0, 0.0],
        #         "z": [0.0, 0.0, 1.0],
        #     }
        
        # builder.add_joint_d6(
        #         linear_axes=[wp.sim.JointAxis(axes[a]) for a in axes],
        #         angular_axes=[wp.sim.JointAxis(axes[a]) for a in axes],
        #         parent_xform=wp.transform_identity(),
        #         child_xform=wp.transform_identity(),
        #         parent=-1,
        #         child=ball1,
        #         name="ball1",

        #     )

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

    # # Krishnan's version
    # def update(
    #     self,
    #     model,
    #     integrator,
    #     state_in,
    #     state_out,
    #     state_list=None,
    #     substeps=10,
    #     dt=1.0 / 60.0,
    #     body_f=None,
    #     joint_q=None,
    #     joint_qd=None,
    #     act_params: dict = None,
    #     record_forward=False,
    #     ):
    #     # if in graph capture mode, only use state_in and state_out
    #     if record_forward:
    #         state_list = [state_out for _ in range(substeps - 1)]
    #     # setup state_list if not provided
    #     if state_list is None or len(state_list) == 0:
    #         state_list = [model.state() for _ in range(substeps - 1)]

    #     # run forward simulate substeps with integrator
    #     for state, state_next in zip([state_in] + state_list[:-1], state_list[1:] + [state_out]):
    #         state.clear_forces()
    #         if model.ground:
    #             wp.sim.collide(model, state_in)
    #         state_next = integrator.simulate(
    #             model,
    #             state,
    #             state_next,
    #             dt,
    #         )
    #         # if state_next is not state_out:
    #         #     state_next.clear_forces()

    #     # # if body_f is included (to compute grads from body_f), copy (Euler) or integrate body_f (XPBD)
    #     # if body_f is not None:
    #     #     if isinstance(integrator, wp.sim.SemiImplicitIntegrator):
    #     #         body_f.assign(state_list[1].body_f)  # takes instantaneous force from last substep
    #     #     elif isinstance(integrator, wp.sim.XPBDIntegrator):
    #     #         # captures applied joint torques
    #     #         body_f.assign(state_out.body_f)
    #     #         integrate_body_f(
    #     #             model,
    #     #             state_in.body_qd,
    #     #             state_out.body_q,
    #     #             state_out.body_qd,
    #     #             body_f,
    #     #             dt * substeps,
    #     #         )
    #     # if joint_q is not None:
    #     #     wp.sim.eval_ik(model, state_out, joint_q, joint_qd)
    #     # return state_out.body_q, state_out.body_qd, joint_q, joint_qd

    #     # return state_list
    #     return state_out
    #     # return state, state_out
    #     # return state_next

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
    def control_body_delta(
        control_deltas: wp.array(dtype=wp.float32),
        joint_target: wp.array(dtype=wp.float32),
        action_dim: int,
    ):
        for i in range(action_dim):
            joint_target[i] = joint_target[i] + control_deltas[i]

        # tid = wp.tid()
        # joint_target[tid] = joint_target[tid] + control_deltas[tid]

    @wp.kernel
    def control_body_abs(
        control_target: wp.array(dtype=wp.float32),
        joint_target: wp.array(dtype=wp.float32),
        action_dim: int,
    ):
        for i in range(action_dim):
            joint_target[i] = control_target[i]

    # @wp.kernel
    # def get_control_deltas(
    #     goal: wp.array(dtype=wp.float32),
    #     err: wp.array(dtype=wp.float32),
    #     manipulability: wp.array(dtype=wp.float32, ndim=2),
    #     control_deltas: wp.array(dtype=wp.float32),
    #     joint_q: wp.array(dtype=wp.transformf),
    #     scale: float,
    # ):
    #     # cube_pos = wp.transform_get_translation(joint_q[0])
    #     # err = goal - cube_pos
    #     # goal = wp.transform_get_translation(joint_q[0])

    #     err[0] = goal[0] - wp.transform_get_translation(joint_q[0])[0]
    #     err[1] = goal[1] - wp.transform_get_translation(joint_q[0])[1]
    #     err[2] = goal[2] - wp.transform_get_translation(joint_q[0])[2]

    #     # manipulability[0, 0] = 0.

    #     # control_deltas = wp.transpose(err) @ manipulability[:, 3:]
    #     # control_deltas = wp.matmul(err, manipulability[:, 3:])
    #     control_deltas = wp.matmul(err, manipulability)
        # control_deltas = wp.normalize(control_deltas) * scale

    def get_control_deltas(self, goal, curr_state, manipulability,  scale=.1):
        err = goal - curr_state
        action = err.T @ manipulability[:, 3:] # ignoring the first 3 columns (cube position) for now TODO: simplify this
        print('unnormalized action: ', action)
        if np.linalg.norm(action) > 1e-6:
            action = action / np.linalg.norm(action) * scale
        else:
            action = np.zeros_like(action)
        return wp.array(action, dtype=wp.float32)

    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state(requires_grad = True)
        self.state_1 = self.model.state(requires_grad = True)
        # self.cube_goal = np.array([-1.0, 0.5, 0.0], dtype=np.float32) # move left
        # self.cube_goal = np.array([1.0, 0.5, -1.0], dtype=np.float32) # move right and back
        self.cube_goal = np.array([0.0, 3.0, 0.0], dtype=np.float32) # move up

        self.state_list = [self.model.state(requires_grad = True) for i in range(self.sim_substeps - 1)]
        self.state_list.append(self.state_1)

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_1)

        profiler = {}

        all_labels = ['cube_x', 'cube_y', 'cube_z','cube_i', 'cube_j', 'cube_k', 'cube_w']
        for ball in self.ball_list:
            all_labels += ['ball{}_x'.format(ball), 'ball{}_y'.format(ball), 'ball{}_z'.format(ball), 'ball{}_i'.format(ball), 'ball{}_j'.format(ball), 'ball{}_k'.format(ball), 'ball{}_w'.format(ball)]

        manip_rows = ['cube_x', 'cube_y', 'cube_z']
        
        # manip_cols = ['cube_x', 'cube_y', 'cube_z', 'ball1_x', 'ball1_y', 'ball1_z', 'ball2_x', 'ball2_y', 'ball2_z', 'ball3_x', 'ball3_y', 'ball3_z']
        manip_cols = ['cube_x', 'cube_y', 'cube_z']
        for ball in self.ball_list:
            manip_cols += ['ball{}_x'.format(ball), 'ball{}_y'.format(ball), 'ball{}_z'.format(ball)]
            
        # manip_rows = all_labels
        # manip_cols = all_labels

        manipulability_ad = np.zeros((len(manip_rows), len(manip_cols)), dtype=np.float32)
        manipulability_fd = np.zeros((len(manip_rows), len(manip_cols)), dtype=np.float32)
        control_deltas = wp.zeros(len(manip_cols) - len(manip_rows), dtype=np.float32)

        # simulate
        self.update()
        # self.state_0, self.state_1 = self.state_1, self.state_0
        # self.update(model=self.model,
        #             integrator=self.model.integrator,
        #             state_in=self.state_0,
        #             state_out=self.state_1,
        #             # state_list=self.state_list,
        #             state_list=None,
        #             substeps=self.sim_substeps,
        #             dt=self.sim_dt,
        #             body_f=None,
        #             joint_q=None,
        #             joint_qd=None,
        #             act_params=None,
        #             record_forward=False,
        #         )


        with wp.ScopedTimer("simulate", detailed=False, print=False, active=False, dict=profiler):

            for f in range(0, self.episode_frames):
                self.sim_time += self.frame_dt

                if (self.enable_rendering):
                    with wp.ScopedTimer("render", active=False):
                        self.render()

                tape = wp.Tape()
                with tape:
                    # control_deltas = wp.array([wp.sin(self.sim_time)*.05, wp.sin(self.sim_time)*.05, wp.sin(self.sim_time)*.05, *wp.quat_identity()], dtype=wp.float32)
                    # control_deltas = wp.array([0,0.5,0, 0,0.5,0, 0,0,0.5,0], dtype=wp.float32) 
                    control_deltas = self.get_control_deltas(self.cube_goal, self.state_1.body_q.numpy()[0][0:3], manipulability_ad)

                    # err = wp.zeros((1, 3), dtype=wp.float32)
                    # wp.launch(self.get_control_deltas, dim=1, inputs=[self.cube_goal, err, wp.array(manipulability_fd, dtype=wp.float32, ndim=2), self.model.joint_q, 1.0], outputs=[control_deltas], device="cuda")
                    
                    wp.launch(self.control_body_delta, dim=1, inputs=[control_deltas, self.model.joint_target, (len(manip_cols) - len(manip_rows))], outputs=[], device="cuda")
                    # wp.launch(self.control_body_abs, dim=1, inputs=[self.cube_goal, self.state_1.joint_q], outputs=[], device="cuda")
                    with wp.ScopedTimer("simulate", active=False):
                        self.update()
                        # self.state_0, self.state_1 = self.state_1, self.state_0
                        # self.update(model=self.model,
                        #             integrator=self.model.integrator,
                        #             state_in=self.state_0,
                        #             state_out=self.state_1,
                        #             # state_list=self.state_list,
                        #             state_list=None,
                        #             substeps=self.sim_substeps,
                        #             dt=self.sim_dt,
                        #             body_f=None,
                        #             joint_q=None,
                        #             joint_qd=None,
                        #             act_params=None,
                        #             record_forward=False,
                        #         )

                    # print('self.state_1.body_q: ', self.state_1.body_q)
                    # print('self.state_1.body_f: ', self.state_1.body_f)
                        
                # check_backward_pass(tape=tape)

                manipulability_ad = get_manipulability_ad(tape, dim=self.state_0.body_q.shape[0] * 7, input_state=self.state_0, output_state=self.state_1)
                
                # check_tape_safety(self.get_manipulability_fd_tape, inputs=[tape, 14])
                
                # manipulability_fd = self.get_manipulability_fd(self.state_1, dim=7, eps=1e-5, input_index=1, output_index=0)
                # manipulability_fd = get_manipulability_fd_composed(self.update_for_jacobian, self.state_1, eps=1e-2, input_indices=[0, 1, 2, 3], output_indices=[0, 1, 2, 3])
                manipulability_fd = get_manipulability_fd_composed(self.update_for_jacobian, self.state_1, eps=1e-5, input_indices=[0, *self.ball_list], output_indices=[0, *self.ball_list])
                
                manipulability_ad = self.select_rows_cols(manipulability_ad,
                                    rows=[all_labels.index(label) for label in manip_rows],
                                    cols=[all_labels.index(label) for label in manip_cols]
                                    )
                
                # print('manipulability_fd before select: ')
                # labeled_matrix_print(manipulability_fd, manip_rows, manip_cols)

                print('manipulability_fd before select:\n', manipulability_fd)

                manipulability_fd = self.select_rows_cols(manipulability_fd,
                                    rows=[all_labels.index(label) for label in manip_rows],
                                    cols=[all_labels.index(label) for label in manip_cols]
                                    )
                
                print('shape after select: ', manipulability_fd.shape)

                # visualizing the autodiff manipulability
                print('manipulability_ad: ')
                labeled_matrix_print(manipulability_ad, manip_rows, manip_cols, precision=6)
                # print(manipulability_ad)
                print('manipulability_fd: ')
                labeled_matrix_print(manipulability_fd, manip_rows, manip_cols, precision=6)
                # print(manipulability_fd)
                # matrix_to_heatmap(manipulability_fd, manip_rows, manip_cols, title="heatmap (fd)", vis=True)
                # matrix_to_heatmap(manipulability_ad, manip_rows, manip_cols, title="heatmap (ad)", vis=True)
                # plot_eigenvalues(manipulability_ad, manip_rows, manip_cols, title="eigenvalues (ad)", vis=True)
                # visualize_unit_ball(manipulability_ad, manip_rows, manip_cols, title="unit ball (ad)", vis=True)
                # cv2.waitKey(1)  # Small delay to display images

                # print('determinant: ', np.linalg.det(manipulability_ad))
                # print('trace: ', np.trace(manipulability_ad))
                # print('eigenvalues: ', np.linalg.eigvals(manipulability_ad))
                # print('eigenvectors: ', np.linalg.eig(manipulability_ad))
                U, S, V = np.linalg.svd(manipulability_fd[:, 3:])
                print('singular values: ', S)
                # print('singular vectors: ', V)
                frobenius_norm = np.linalg.norm(manipulability_fd[:, 3:], ord='fro')
                print('frobenius norm: ', frobenius_norm)
                nuclear_norm = np.linalg.norm(manipulability_fd[:, 3:], ord='nuc')
                print('nuclear norm: ', nuclear_norm)
                inf_norm = np.linalg.norm(manipulability_fd[:, 3:], ord=np.inf)
                print('inf norm: ', inf_norm)
                one_norm = np.linalg.norm(manipulability_fd[:, 3:], ord=1)
                print('one norm: ', one_norm)
                two_norm = np.linalg.norm(manipulability_fd[:, 3:], ord=2)
                print('two norm: ', two_norm)
                print('error: ', self.cube_goal - self.state_1.body_q.numpy()[0][0:3])
                print('control_deltas: ', control_deltas)
                # print('self.state_1.body_q: ', self.state_1.body_q)
                # print('self.model.joint_target: ', self.model.joint_target)
                
                # # printing attributes of the state
                # for attr in dir(self.state_0):
                #     print("obj.%s = %r" % (attr, getattr(self.state_0, attr)))

                # print('fd == ad:\n', (manipulability_fd - manipulability_ad) < 1e-2)
                print('\n\n')

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
robot = Example(stage, render=True)
robot.run()