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
        parser.add_argument("--move_obj", action='store_true', default=False)
        parser.add_argument("--use_ad", action='store_true', default=False)
        parser.add_argument("--num_balls", type=int, default=1)
        parser.add_argument("--goal_x", type=float, default=0.)
        parser.add_argument("--goal_y", type=float, default=0.5)
        parser.add_argument("--goal_z", type=float, default=0.)
        parser.add_argument("--update_func", type=str, default="krishnan")
        self.args = parser.parse_args()

        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        self.scale = 0.8

        if self.args.update_func == "krishnan":
            self.ke = 3.e6 # 1.e+2 # 1.e+5
            self.kd = 1e4  # 250.0
            self.kf = 1e3 # 0.5
            self.joint_target_ke = 1e8
            self.joint_target_kd = 1e6
        else:
            self.ke = 2.e3 # 1.e+2 # 1.e+5
            self.kd = 10000.0
            self.kf = 1000 # 0.5
            self.joint_target_ke = 1e3
            self.joint_target_kd = 0
        
        # self.ke = 1.e+2
        # self.kd = 250.0
        # self.kf = 500.0
        self.num_balls = self.args.num_balls
        

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
        )

        self.ball_list = []
        self.ball_pos = {}

        for i in range(self.num_balls):
            # index = builder.add_body(origin=wp.transform((wp.cos(wp.pi * 2 * i / self.num_balls), 0.5, wp.sin(wp.pi * 2 * i / self.num_balls)), wp.quat_identity()))
            ball_pos = (wp.cos(wp.pi * 2 * i / self.num_balls)*0.9, 0.5, wp.sin(wp.pi * 2 * i / self.num_balls)*0.9)
            index = builder.add_body(origin=wp.transform(ball_pos, wp.quat_identity()))
            self.ball_pos[index] = ball_pos

            builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0),
                radius=0.75 * self.scale,
                body=index,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                # density=1.e3  # 1000.0
            )

            # builder.add_shape_box(
            #     pos=(0.0, 0.0, 0.0),
            #     hx=0.5 * self.scale,
            #     hy=0.5 * self.scale,
            #     hz=0.5 * self.scale,
            #     body=index,
            #     ke=self.ke,
            #     kd=self.kd,
            #     kf=self.kf,
            #     density=1.e3  # 1000.0 
            # )


            self.ball_list.append(index)

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

        if self.args.no_control:
            # initial ball velocity
            # builder.body_qd[1] = (0.0, 0.0, 0.0, -5.0, 0.0, 0.0)
            builder.body_qd[1] = (-5.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
                        target_ke=1e5,  # Stiffness
                        target_kd=1e6  # Damping
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
    def update_list(self, substeps):
        self.state_list = [self.state_0]

        for i in range(substeps):
            self.state_list[-1].clear_forces()
            wp.sim.collide(self.model, self.state_list[-1])
            next_state = self.model.state(requires_grad = True)
            self.integrator.simulate(self.model, self.state_list[-1], next_state, self.sim_dt)

            if i < self.sim_substeps - 1:
                self.state_list.append(next_state)

        self.state_0 = self.state_list[1]
        # self.state_0 = self.state_list[-2]
        self.state_1 = self.state_list[-1]

        for state in self.state_list[:-1]:
            state.clear_forces()

    # dynamically building state_list (old)
    def update_list_old(self, substeps):
        self.state_list = [self.state_0]

        for i in range(substeps):
            self.state_list[-1].clear_forces()
            wp.sim.collide(self.model, self.state_list[-1])
            next_state = self.model.state(requires_grad = True)
            self.integrator.simulate(self.model, self.state_list[-1], next_state, self.sim_dt)

            if i < self.sim_substeps - 1:
                self.state_list.append(next_state)

        # self.state_0 = self.state_list[1]
        self.state_0 = self.state_list[-2]
        self.state_1 = self.state_list[-1]

    # Krishnan's version
    def update_krishnan(
        self,
        model,
        integrator,
        state_in,
        state_out,
        state_list=None,
        substeps=10,
        dt=1.0 / 60.0,
        body_f=None,
        joint_q=None,
        joint_qd=None,
        act_params: dict = None,
        record_forward=False,
        ):
        # if in graph capture mode, only use state_in and state_out
        if record_forward:
            state_list = [state_out for _ in range(substeps - 1)]
        # setup state_list if not provided
        if state_list is None or len(state_list) == 0:
            state_list = [model.state() for _ in range(substeps - 1)]

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
            if state_next is not state_out:
                state_next.clear_forces()

        # # if body_f is included (to compute grads from body_f), copy (Euler) or integrate body_f (XPBD)
        # if body_f is not None:
        #     if isinstance(integrator, wp.sim.SemiImplicitIntegrator):
        #         body_f.assign(state_list[1].body_f)  # takes instantaneous force from last substep
        #     elif isinstance(integrator, wp.sim.XPBDIntegrator):
        #         # captures applied joint torques
        #         body_f.assign(state_out.body_f)
        #         integrate_body_f(
        #             model,
        #             state_in.body_qd,
        #             state_out.body_q,
        #             state_out.body_qd,
        #             body_f,
        #             dt * substeps,
        #         )
        # if joint_q is not None:
        #     wp.sim.eval_ik(model, state_out, joint_q, joint_qd)
        # return state_out.body_q, state_out.body_qd, joint_q, joint_qd

        # return state_list
        return state_out
        # return state, state_out
        # return state_next

    # from example_sim_grad_control.py
    def update_control(self, state: wp.sim.State, substeps, requires_grad=True) -> wp.sim.State:
        """
        Simulate the system for the given states.
        """
        for _ in range(substeps):
            self.model.allocate_rigid_contacts(requires_grad=requires_grad)
            if requires_grad:
                next_state = self.model.state(requires_grad=True)
            else:
                next_state = state
            next_state.clear_forces()

            # if self.model.ground:
            # self.model.allocate_rigid_contacts(requires_grad=requires_grad)
            wp.sim.collide(self.model, state)
            state = self.integrator.simulate(self.model, state, next_state, self.sim_dt, requires_grad=requires_grad)

        return state

    # from example_sim_rigid_trajopt.py
    def update_trajopt(self, state: wp.sim.State, substeps, requires_grad=False) -> wp.sim.State:
        """
        Simulate the system for the given states.
        """

        for _ in range(self.sim_substeps):
            if requires_grad:
                next_state = self.model.state(requires_grad=True)
            else:
                next_state = state
                next_state.clear_forces()

            if self.model.ground:
                wp.sim.collide(self.model, state)
            state = self.integrator.simulate(self.model, state, next_state, self.sim_dt, requires_grad=requires_grad)
        return state

    def update(
        self,
        model,
        integrator,
        state_in,
        state_out,
        state_list=None,
        substeps=10,
        dt=1.0 / 60.0,
        body_f=None,
        joint_q=None,
        joint_qd=None,
        act_params: dict = None,
        record_forward=False,
        ):

        if self.args.update_func == "krishnan":
            return self.update_krishnan(
                model=model,
                integrator=integrator,
                state_in=state_in,
                state_out=state_out,
                state_list=state_list,
                substeps=substeps,
                dt=dt,
                body_f=body_f,
                joint_q=joint_q,
                joint_qd=joint_qd,
                act_params=act_params,
                record_forward=record_forward,
            )
        elif self.args.update_func == "control":
            return self.update_control(state=state_in, substeps=substeps, requires_grad=True)
        elif self.args.update_func == "trajopt":
            return self.update_trajopt(state=state_in, substeps=substeps, requires_grad=True)
        elif self.args.update_func == "list":
            self.update_list(substeps)
        elif self.args.update_func == "list_old":
            self.update_list_old(substeps)
        else:
            raise NotImplementedError

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_1)
        # self.renderer.render(self.state_list[-1])
        # self.renderer.render(self.curr_state)
        self.renderer.end_frame()

    # # single substep version for finite difference
    # def update_for_jacobian(self, start_state):
    #     start_state.clear_forces()
    #     wp.sim.collide(self.model, start_state)

    #     end_state = self.model.state(requires_grad = True)
    #     self.integrator.simulate(model=self.model, state_in=start_state, state_out=end_state, dt=self.sim_dt, requires_grad=True)

    #     return end_state

    # # full substep version for finite difference
    # def update_for_jacobian(self, start_state, substeps=2):
    #     state_list = [start_state]

    #     for i in range(substeps):
    #         # self.state_list[-1].clear_forces()
    #         wp.sim.collide(self.model, state_list[-1])
    #         next_state = self.model.state(requires_grad = True)
    #         self.integrator.simulate(self.model, state_list[-1], next_state, self.sim_dt)

    #         if i < self.sim_substeps - 1:
    #             state_list.append(next_state)

    #     end_state = state_list[-1]

    #     for state in self.state_list:
    #         state.clear_forces()

    def update_for_jacobian(self, start_state, substeps=2):
        end_state = self.model.state(requires_grad=True)
        self.update(model=self.model,
                    integrator=self.model.integrator,
                    state_in=start_state,
                    state_out=end_state,
                    state_list=None,
                    substeps=substeps,
                    dt=self.sim_dt,
                    body_f=None,
                    joint_q=None,
                    joint_qd=None,
                    act_params=None,
                    record_forward=False,
                )
        
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

    def get_control_deltas(self, goal, curr_state, manipulability,  scale=.1, window_len=10):
        err = goal - curr_state
        print('goal error: ', err)
        action = err.T @ manipulability
        print('unnormalized action: ', action)
        if np.linalg.norm(action) > 1e-6:
            action = action / np.linalg.norm(action) * scale
        else:
            action = np.zeros_like(action)

        if window_len > 1:
            # averaging the last window_len actions
            self.action_list.append(action)
            if len(self.action_list) > window_len:
                self.action_list.pop(0)
            final_action = np.mean(self.action_list, axis=0)
        else:
            final_action = action

        return wp.array(final_action, dtype=wp.float32)

    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state(requires_grad = True)
        self.state_1 = self.model.state(requires_grad = True)
        self.current_state = self.model.state(requires_grad = True)
        # self.cube_goal = np.array([-1.0, 0.5, 0.0], dtype=np.float32) # move left
        # self.cube_goal = np.array([1.0, 0.5, -1.0], dtype=np.float32) # move right and back
        # self.cube_goal = np.array([0.0, 3.0, 0.0], dtype=np.float32) # move up
        self.cube_goal = np.array([self.args.goal_x, self.args.goal_y, self.args.goal_z], dtype=np.float32)

        self.state_list = [self.model.state(requires_grad = True) for i in range(self.sim_substeps - 1)]
        self.state_list.append(self.state_1)

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_1)
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.current_state)

        profiler = {}

        all_labels = ['cube_x', 'cube_y', 'cube_z', 'cube_i', 'cube_j', 'cube_k', 'cube_w']
        all_labels += ['ball{}_{}'.format(ball, axis) for ball in self.ball_list for axis in ['x', 'y', 'z', 'i', 'j', 'k', 'w']]
        
        if self.args.move_obj:
            manip_rows = ['cube_x', 'cube_y', 'cube_z'] + ['ball{}_{}'.format(ball, axis) for ball in self.ball_list for axis in ['x', 'y', 'z']] # outputs
            manip_cols = ['cube_x', 'cube_y', 'cube_z'] # inputs
        else:
            manip_rows = ['cube_x', 'cube_y', 'cube_z'] # outputs
            manip_cols = ['cube_x', 'cube_y', 'cube_z'] + ['ball{}_{}'.format(ball, axis) for ball in self.ball_list for axis in ['x', 'y', 'z']] # inputs

        # manip_rows = all_labels
        # manip_cols = all_labels

        manipulability_ad = np.zeros((len(manip_rows), len(manip_cols)), dtype=np.float32)
        manipulability_fd = np.zeros((len(manip_rows), len(manip_cols)), dtype=np.float32)
        action_dim = abs(len(manip_cols) - len(manip_rows))
        control_deltas = wp.zeros(action_dim, dtype=np.float32)
        self.action_list = []

        # simulate

        # tape = wp.Tape()
        # with tape:
        # self.update()
            # tape.forward()
        # check_backward_pass(tape=tape)

        # self.state_0, self.state_1 = self.state_1, self.state_0
        self.state_0 = self.state_1
        self.update(model=self.model,
                    integrator=self.model.integrator,
                    state_in=self.state_0,
                    state_out=self.state_1,
                    # state_list=self.state_list,
                    state_list=None,
                    substeps=self.sim_substeps,
                    dt=self.sim_dt,
                    body_f=None,
                    joint_q=None,
                    joint_qd=None,
                    act_params=None,
                    record_forward=False,
                )

        # self.state_1 = self.update(self.state_0, requires_grad=True)
        # self.state_1 = self.update(self.state_1, requires_grad=True)

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
                    if self.args.use_ad:
                        control_manipulability = manipulability_ad
                    else:
                        control_manipulability = manipulability_fd

                    # wp.launch(self.get_control_deltas, dim=1, inputs=[self.cube_goal, err, wp.array(manipulability_fd, dtype=wp.float32, ndim=2), self.model.joint_q, 1.0], outputs=[control_deltas], device="cuda")
                    if not self.args.no_control:
                        if self.args.move_obj:
                            control_deltas = self.get_control_deltas(self.cube_goal, self.state_1.body_q.numpy()[0][0:3], control_manipulability.T[:, 3:]) # ignoring the first 3 columns (cube position) for now TODO: simplify this
                        else:
                            control_deltas = self.get_control_deltas(self.cube_goal, self.state_1.body_q.numpy()[0][0:3], control_manipulability[:, 3:]) # ignoring the first 3 columns (cube position) for now TODO: simplify this

                        wp.launch(self.control_body_delta, dim=1, inputs=[control_deltas, self.model.joint_target, action_dim], outputs=[], device="cuda")
                        print('control_deltas: ', control_deltas)
                        # self.model.joint_target_ke = wp.array([self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke], dtype=wp.float32)
                        # self.model.joint_target_kd = wp.array([self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd], dtype=wp.float32)
                    # wp.launch(self.control_body_abs, dim=1, inputs=[self.cube_goal, self.state_1.joint_q], outputs=[], device="cuda")

                    with wp.ScopedTimer("simulate", active=False):
                        # self.update()
                        # self.state_0, self.state_1 = self.state_1, self.state_0
                        self.state_0 = self.state_1
                        self.update(model=self.model,
                                    integrator=self.model.integrator,
                                    state_in=self.state_0,
                                    state_out=self.state_1,
                                    # state_list=self.state_list,
                                    state_list=None,
                                    substeps=self.sim_substeps,
                                    dt=self.sim_dt,
                                    body_f=None,
                                    joint_q=None,
                                    joint_qd=None,
                                    act_params=None,
                                    record_forward=False,
                                )
                        
                        # self.state_1 = self.update(self.state_0, requires_grad=True)

                    # print('self.state_1.body_q: ', self.state_1.body_q)
                    # print('self.state_1.body_f: ', self.state_1.body_f)
                        
                # check_backward_pass(tape=tape)

                manipulability_ad = get_manipulability_ad(tape, dim=self.state_0.body_q.shape[0] * 7, input_state=self.state_0, output_state=self.state_1)
                # manipulability_ad = get_manipulability_ad(tape, dim=self.state_1.body_q.shape[0] * 7, input_state=self.state_1, output_state=self.state_1)

                # check_tape_safety(self.get_manipulability_fd_tape, inputs=[tape, 14])
                
                # manipulability_fd = self.get_manipulability_fd(self.state_1, dim=7, eps=1e-5, input_index=1, output_index=0)
                # manipulability_fd = get_manipulability_fd_composed(self.update_for_jacobian, self.state_1, eps=1e-2, input_indices=[0, 1, 2, 3], output_indices=[0, 1, 2, 3])
                # target_ke=1e8,  # Stiffness
                # target_kd=1e6  # Damping

                # setting stiffness and damping to 0 for the goal cube
                self.model.joint_target_ke = wp.array([0., 0., 0., 0., 0., 0.], dtype=wp.float32)
                self.model.joint_target_kd = wp.array([0., 0., 0., 0., 0., 0.], dtype=wp.float32)
                manipulability_fd = get_manipulability_fd_composed(self.update_for_jacobian, self.state_1, eps=1e-2, input_indices=[0, *self.ball_list], output_indices=[0, *self.ball_list])
                self.model.joint_target_ke = wp.array([self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke, self.joint_target_ke], dtype=wp.float32)
                self.model.joint_target_kd = wp.array([self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd, self.joint_target_kd], dtype=wp.float32)

                manipulability_ad = self.select_rows_cols(manipulability_ad,
                                    rows=[all_labels.index(label) for label in manip_rows],
                                    cols=[all_labels.index(label) for label in manip_cols]
                                    )
                
                # print('manipulability_fd before select:\n', manipulability_fd)

                manipulability_fd = self.select_rows_cols(manipulability_fd,
                                    rows=[all_labels.index(label) for label in manip_rows],
                                    cols=[all_labels.index(label) for label in manip_cols]
                                    )
                
                print('manip fd after select: ', manipulability_fd.shape)

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

                print('\n\n')

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
robot = Example(stage, render=True)
robot.run()