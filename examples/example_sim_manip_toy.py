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
from warp.envs import WarpEnv
# from pair_warp.envs.warp_env import WarpEnv
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
        b1 = builder.add_body(origin=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))

        s1 = builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=0.5 * self.scale,
            hy=0.5 * self.scale,
            hz=0.5 * self.scale,
            body=b1,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
        )

        # spheres
        b2 = builder.add_body(origin=wp.transform((-2.0, 1.0, 0.0), wp.quat_identity()))

        s2 = builder.add_shape_sphere(
            pos=(0.0, 0.0, 0.0), radius=0.75 * self.scale, body=b2, ke=self.ke, kd=self.kd, kf=self.kf
        )

        # initial ball velocity
        builder.body_qd[1] = (0.0, 0.0, 0.0, 5.0, 0.0, 0.0)
            
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

    # def update(self):
    #     # for _ in range(self.sim_substeps):
    #     state_in = self.state_0
    #     # for state_out in self.state_list:c
    #     for i in range(len(self.state_list)):
    #         state_in.clear_forces()
    #         wp.sim.collide(self.model, state_in)
    #         self.state_list[i] = self.model.state(requires_grad = True)
    #         self.integrator.simulate(self.model, state_in, self.state_list[i] , self.sim_dt)
    #         # state_in, state_out = state_out, state_in
    #         state_in = self.state_list[i]
        
        # self.state_0 = self.state_list[-1]

    # Jeremy's version
    # def update(self):
    #     # self.state_list[0] = self.state_0
    #     for i in range(self.sim_substeps):
    #         self.state_0.clear_forces()
    #         wp.sim.collide(self.model, self.state_0)
    #         # self.state_1 = self.model.state(requires_grad = True)
    #         self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
    #         self.state_list[i] = self.state_1 # adding the most recent state to the list
    #         self.state_0, self.state_1 = self.state_1, self.state_0

    # gets identity correct for any number of substeps, but still overwrites tape
    # def update(self):
    #     for i in range(self.sim_substeps):
    #         self.state_0.clear_forces()
    #         wp.sim.collide(self.model, self.state_0)
    #         if i < self.sim_substeps - 1:
    #             # Use a temporary state for intermediate steps
    #             temp_state = self.model.state(requires_grad=True)
    #             self.integrator.simulate(self.model, self.state_0, temp_state, self.sim_dt)
    #             # Update state_0 to the new state
    #             self.state_0 = temp_state
    #         else:
    #             # In the last step, update state_1
    #             self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

    # directly using state_pairs
    # def update(self):
    #     # self.state_list[0] = self.state_0
    #     self.state_pairs = list(zip([self.state_0] + self.state_list[:-1], self.state_list)) # [(state_0, state_1), (state_1, state_2), ...]

    #     for i in range(self.sim_substeps):
    #         self.state_pairs[i][0].clear_forces()
    #         wp.sim.collide(self.model, self.state_pairs[i][0])
    #         self.integrator.simulate(self.model, self.state_pairs[i][0], self.state_pairs[i][1], self.sim_dt)
    #         # self.state_list[i] = self.state_1 # adding the most recent state to the list
    #         # adding most recent state to list
    #         if i < self.sim_substeps - 1:
    #             self.state_pairs[i][1] = self.state_pairs[i+1][0]
    #         # self.state_0, self.state_1 = self.state_1, self.state_0

    #     self.state_0 = self.state_pairs[-1][0]
    #     self.state_1 = self.state_pairs[-1][1]

    # # directly using state_pairs as list of lists
    # def update(self):
    #     # self.state_list[0] = self.state_0
    #     # self.state_pairs = list(zip([self.state_0] + self.state_list[:-1], self.state_list)) # [(state_0, state_1), (state_1, state_2), ...]
    #     # self.state_pairs = list(zip([self.model.state(requires_grad = True) for i in range(self.sim_substeps)], [self.model.state(requires_grad = True) for i in range(self.sim_substeps)]))
    #     self.state_pairs = [[self.model.state(requires_grad = True), self.model.state(requires_grad = True)] for i in range(self.sim_substeps)]

    #     # for state_in, state_out in self.state_pairs:
    #     for i in range(self.sim_substeps):
    #         self.state_pairs[i][0].clear_forces()
    #         wp.sim.collide(self.model, self.state_pairs[i][0])
    #         self.integrator.simulate(self.model, self.state_pairs[i][0], self.state_pairs[i][1], self.sim_dt)

    #         if i < self.sim_substeps - 1:
    #             self.state_pairs[i+1][0] = self.state_pairs[i][1] # this might be problematic

    #     self.state_0 = self.state_pairs[-1][0]
    #     self.state_1 = self.state_pairs[-1][1]

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

    # Krishnan's version
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

    # # Krishnan's version
    # def update(self):
    # # self.state_list[0] = self.state_0
    # state_pairs = zip([self.state_0] + self.state_list[:-1], self.state_list)

    # # for i in range(self.sim_substeps): # TODO: pop off intermediate states from state list 
    # for state_in, state_out in state_pairs:
    #     state_in.clear_forces()
    #     wp.sim.collide(self.model, state_in)
    #     # self.state_1 = self.model.state(requires_grad = True)
    #     self.integrator.simulate(self.model, state_in, state_out, self.sim_dt)
    #     # self.state_list[i] = self.state_1 # adding the most recent state to the list
    #     # self.state_0, self.state_1 = self.state_1, self.state_0

    # dynamically building state_list
    def update_for_jacobian(self, start_state):
        start_state.clear_forces()
        wp.sim.collide(self.model, start_state)

        end_state = self.model.state(requires_grad = True)
        self.integrator.simulate(model=self.model, state_in=start_state, state_out=end_state, dt=self.sim_dt, requires_grad=True)

        return end_state

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        # self.renderer.render(self.state_list[-1])
        # self.renderer.render(self.curr_state)
        self.renderer.end_frame()

    # @wp.func
    def pose_from_transformf(v: wp.array(dtype=wp.transformf)):
        return wp.array([v[0:3], wp.quat_from_axis_angle(v[3:7])])

    @wp.kernel
    # @jacobian_check(input_names=["state_0", "state_1"], output_names=["manipulability"])
    def manipulability_kernel(state_0: wp.array(dtype=wp.transformf), state_1: wp.array(dtype=wp.transformf), manipulability: wp.array(dtype=wp.float32, ndim=2)):
    # def manipulability_kernel(state_in: wp.array(dtype=wp.transformf), state_out: wp.array(dtype=wp.transformf)):
        # unpack transform
        cube_0_trans = wp.transform_get_translation(state_0[0])
        cube_0_rot = wp.transform_get_rotation(state_0[0])

        cube_1_trans = wp.transform_get_translation(state_1[0])
        cube_1_rot = wp.transform_get_rotation(state_1[0])

        ball_0_trans = wp.transform_get_translation(state_0[1])
        ball_0_rot = wp.transform_get_rotation(state_0[1])

        ball_1_trans = wp.transform_get_translation(state_1[1])
        ball_1_rot = wp.transform_get_rotation(state_0[1])

        for i in range(7):
            for j in range(7):
                if (ball_1_trans[j] - ball_0_trans[j]) != 0:
                    manipulability[i][j] = (cube_1_trans[i] - cube_0_trans[i]) / (ball_1_trans[j] - ball_0_trans[j])
                else:
                    manipulability[i][j] = 0.0

        # manipulability = (cube_1 - cube_0) / (ball_1 - ball_0)
        # manipulability = wp.cw_div((cube_1_trans - cube_0_trans), (ball_1_trans - ball_0_trans))

    # @tape_check(tol=1e-5, check_nans=False)
    def get_manipulability_ad(self, inner_tape, dim):
        manipulability_ad = np.empty((dim, dim), dtype=np.float32) # we want to compute the Jacobian of the cube output with respect to the ball input.
        # output_buffer = manipulability_kernel(self.state_0.body_q, self.state_1.body_q)
        output_buffer = self.state_1.body_q
        # output_buffer = self.state_list[-1].body_q
        # output_buffer = self.curr_state.body_q
        # output_buffer =  wp.launch(self.manipulability_kernel, dim=1, inputs=[self.state_0.body_q], outputs=[self.state_1.body_q], device="cuda")
        
        for output_index in range(dim):
            # select which row of the Jacobian we want to compute
            select_index = np.zeros(dim)
            select_index[output_index] = 1.0
            e = wp.array(select_index, dtype=wp.transformf)
            # pass input gradients to the output buffer to apply selection
            inner_tape.backward(grads={output_buffer: e})
            # inner_tape.backward(grads={self.state_1.body_q: e})
            q_grad_i = inner_tape.gradients[self.state_0.body_q]
            # q_grad_i = inner_tape.gradients[self.state_list[-2].body_q]
            # print("inner_tape.gradients: ", inner_tape.gradients)
            # q_grad_i = inner_tape.gradients[self.prev_state.body_q]
            # q_grad_i = inner_tape.gradients[self.state_pairs[-1][0].body_q]
            manipulability_ad[output_index, :] = q_grad_i.numpy().flatten().astype(np.float32)
            inner_tape.zero()

        return manipulability_ad
        # return wp.array(manipulability_ad, dtype=wp.float32)
        # return wp.from_numpy(manipulability_ad, dtype=wp.float32, device="cuda")

    
    def get_manipulability_fd_tape(self, tape, dim, eps=1e-4, max_fd_dims_per_var=500):
        manipulability_fd = tape_jacobian_fd(tape=tape, inputs=[self.state_0.body_q], outputs=[self.state_1.body_q], eps=1e-4, max_fd_dims_per_var=500)
        # manipulability_fd = tape_jacobian_fd(tape=tape, inputs=[self.state_list[-2].body_q], outputs=[self.state_list[-1].body_q], eps=1e-4, max_fd_dims_per_var=500)
        # manipulability_fd = tape_jacobian_fd(tape=tape, inputs=[self.prev_state.body_q], outputs=[self.curr_state.body_q], eps=1e-4, max_fd_dims_per_var=500)
        return wp.array(manipulability_fd, dtype=wp.float32)

    # @wp.func
    # def simulate_step(model: wp.sim.model.Model, integrator: wp.sim.SemiImplicitIntegrator, state_in: wp.array(dtype=wp.transformf), state_out: wp.array(dtype=wp.transformf), sim_dt: wp.float32):
    #     state_in.clear_forces()
    #     wp.sim.collide(model, state_in)
    #     integrator.simulate(model, state_in, state_out, sim_dt)

    # @wp.kernel
    # def manipulability_fd_kernel(in_state: wp.array(dtype=wp.transformf), out_state: wp.array(dtype=wp.transformf), manipulability: wp.array(dtype=wp.float32, ndim=2), f: wp.func, model: wp.sim.model.Model, sim_dt: wp.float32):
    # # def manipulability_fd_kernel(self, in_state, dim, eps=1e-5, manipulability: wp.array(dtype=wp.float32, ndim=2)):
    #     eps=1e-5
    #     # num_in = len(x)
    #     # num_out = len(f(x))
    #     # self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
    #     for i in range(14):
    #         in_state[i].body_q += eps
    #         f1 = f(model, in_state, out_state, sim_dt)
    #         in_state[i].body_q -= 2 * eps
    #         f2 = f(model, in_state, out_state, sim_dt)
    #         in_state[i].body_q += eps
    #         manipulability[:, i] = (f1 - f2) / (2 * eps)
    #     return manipulability

    # from example_sim_grad_control.py
    # @wp.kernel
    # def inplace_assign(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
    #     tid = wp.tid()
    #     b[tid] = a[tid]

    def warp_step_generalized(self, q: wp.array, qd: wp.array, tau: wp.array, q_next: wp.array, qd_next: wp.array, requires_grad=False, check_diffs=False):
        """
        Advances the system dynamics given the generalized rigid-body state and joint torques [q, qd, tau].
        Simulates for the set number of substeps and returns the next state in generalized coordinates [q_next, qd_next].
        XXX The forward kinematics and inverse kinematics projection steps may lead to known numerical issues where the
        32-bit floating point precision is not sufficient to represent the state updates, leading to a damped/frozen system.
        """

        if check_diffs:
            model_before = self.builder.finalize(self.device)

        # start_state = self.model.state(requires_grad=requires_grad)
        # wp.sim.eval_fk(self.model, q, qd, None, start_state)
        start_state = self.state_0
        # start_state = self.curr_state


        start_state.clear_forces()
        wp.sim.collide(self.model, start_state)

        end_state = self.model.state(requires_grad = True)

        # assign input controls as joint torques
        # wp.launch(self.inplace_assign, dim=3, inputs=[tau], outputs=[self.model.joint_act], device=self.device)

        # end_state = self.simulate(start_state, requires_grad=requires_grad)
        # end_state = self.integrator.simulate(model=self.model, state_in=self.state_0, state_out=self.state_1, dt=self.sim_dt, requires_grad=requires_grad)
        self.integrator.simulate(model=self.model, state_in=start_state, state_out=end_state, dt=self.sim_dt, requires_grad=requires_grad)

        wp.sim.eval_ik(self.model, end_state, q_next, qd_next)

        if check_diffs:
            assert requires_grad, "check_diffs requires requires_grad=True because the state gets overwritten otherwise"
            # check which arrays in model were modified
            for key, value in vars(model_before).items():
                if isinstance(value, wp.array) and len(value) > 0:
                    if not np.allclose(value.numpy(), getattr(self.model, key).numpy()):
                        print(f"model.{key} was modified")
                        print("  before:", value.numpy().flatten())
                        print("  after: ", getattr(self.model, key).numpy().flatten())
            # check which arrays in state were modified
            for key, value in vars(start_state).items():
                if isinstance(value, wp.array) and len(value) > 0:
                    if not np.allclose(value.numpy(), getattr(end_state, key).numpy()):
                        print(f"state.{key} was modified")
                        print("  before:", value.numpy().flatten())
                        print("  after: ", getattr(end_state, key).numpy().flatten())

        # if (self.render):
        #     self._render(end_state)

    @staticmethod
    def fd_jacobian(f, x, eps=1e-5):
        num_in = len(x)
        num_out = len(f(x))
        jac = np.zeros((num_out, num_in), dtype=np.float32)
        for i in range(num_in):
            x[i] += eps
            f1 = f(x)
            x[i] -= 2 * eps
            f2 = f(x)
            x[i] += eps
            jac[:, i] = (f1 - f2) / (2 * eps)
        return jac

    def fd_jacobian_generalized(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, eps=1e-5):
        # build a vector function that accepts the concatenation of (q, qd, tau) and
        # returns the concatenation of (q_next, qd_next)
        def f(q_qd_tau):
            # dof = 3
            # q = wp.array(q_qd_tau[:self.dof_q], dtype=wp.float32, device=self.device)
            # qd = wp.array(q_qd_tau[self.dof_q:self.dof_q + self.dof_qd], dtype=wp.float32, device=self.device)
            # tau = wp.array(q_qd_tau[-self.dof_qd:], dtype=wp.float32, device=self.device)

            # q = wp.array(q_qd_tau[:dof], dtype=wp.float32, device=self.device)
            # qd = wp.array(q_qd_tau[dof:2*dof], dtype=wp.float32, device=self.device)
            # tau = wp.array(q_qd_tau[-dof:], dtype=wp.float32, device=self.device)

            q = wp.array(q_qd_tau[:7], dtype=wp.float32, device=self.device)
            qd = wp.array(q_qd_tau[7:13], dtype=wp.float32, device=self.device)
            tau = wp.array(q_qd_tau[13:], dtype=wp.float32, device=self.device)

            # print("q in f: ", q.shape)
            # print("qd in f: ", qd.shape)
            # print("tau in f: ", tau.shape)

            q_next = wp.zeros_like(q)
            qd_next = wp.zeros_like(qd)
            self.warp_step_generalized(q, qd, tau, q_next, qd_next)
            return np.concatenate([q_next.numpy(), qd_next.numpy()])

        q_qd_tau = np.concatenate([q, qd, tau], dtype=np.float32, axis=0)
        return self.fd_jacobian(f, q_qd_tau, eps)

    def get_manipulability_fd_generalized(self):
        print("body_q: ", self.model.body_q.numpy())
        print("body_qd: ", self.model.body_qd.numpy())

        # q = self.model.body_q.numpy() # (2, 7)
        # qd = self.model.body_qd.numpy() # (2, 6)
        q = self.model.body_q.numpy()[0] # (1, 7)
        qd = self.model.body_qd.numpy()[0] # (1, 6)
        # q = self.model.body_q.numpy()[:, :3].flatten() # (2, 3) linear translation
        # qd = self.model.body_qd.numpy()[:, :3].flatten() # (2, 3) linear velocity

        # tau = self.model.joint_act.numpy()

        # randomize inputs
        # np.random.seed(123)
        # q = np.random.randn(*q.shape) * 0.5
        # qd = np.random.randn(*qd.shape) * 6.5

        # tau = np.zeros(self.model.joint_dof_count)  # np.random.randn(sim.dof_qd) * 10.0
        # tau = np.zeros((2, 3))
        # tau = np.zeros(6)
        # tau = np.random.randn(3) * 10.0
        tau = np.random.randn(6) * 10.0

        print("q:  ", q)
        print("qd: ", qd)
        print("tau:", tau)

        return self.fd_jacobian_generalized(q, qd, tau, eps=1e-5)

    def select_rows_cols(self, arr, rows, cols):
        arr = arr[rows, :]
        arr = arr[:, cols]

        return arr

    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state(requires_grad = True)
        self.state_1 = self.model.state(requires_grad = True)

        self.state_list = [self.model.state(requires_grad = True) for i in range(self.sim_substeps - 1)]
        self.state_list.append(self.state_1)

        # self.state_0.body_q.requires_grad = True
        # self.state_0.body_qd.requires_grad = True
        # self.state_0.body_f.requires_grad = True
        # self.state_1.body_q.requires_grad = True
        # self.state_1.body_qd.requires_grad = True
        # self.state_1.body_f.requires_grad = True

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_1)

        profiler = {}

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
        
        # self.state_0, self.state_1 = self.curr_state, self.prev_state
        # self.state_0, self.state_1 = self.state_1, self.state_0

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
                    # wp.launch(self.manipulability_kernel, dim=1, inputs=[self.state_0.body_q, self.state_1.body_q], outputs=[manipulability_fd], device="cuda")
                    # wp.launch(self.manipulability_fd_kernel(self.state_0.body_q, self.state_1.body_q, manipulability_fd, self.simulate_step, self.model, self.sim_dt), dim=1, inputs=[self.state_0.body_q, self.state_1.body_q], outputs=[manipulability_fd], device="cuda")
                    with wp.ScopedTimer("simulate", active=False):
                        # wp.capture_launch(graph)
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
                        
                        # self.state_0, self.state_1 = self.curr_state, self.prev_state
                        
                # check_backward_pass(tape=tape)

                # check_tape_safety(self.get_manipulability_ad, inputs=[tape, 14])
                manipulability_ad = self.get_manipulability_ad(tape, dim=14)
                # manipulability_ad = manipulability_ad.numpy()

                # check_tape_safety(self.get_manipulability_fd_tape, inputs=[tape, 14])
                # manipulability_fd = self.get_manipulability_fd_tape(tape, dim=14, eps=1e-5)
                # manipulability_fd = self.get_manipulability_fd(self.state_list[-1], dim=14)
                # manipulability_fd = self.get_manipulability_fd_generalized()
                # manipulability_fd = function_jacobian_fd(self.update_for_jacobian, inputs=[self.state_0])

                all_labels = ['cube_x', 'cube_y', 'cube_z',
                            'cube_i', 'cube_j', 'cube_k', 'cube_w',
                            'ball_x', 'ball_y', 'ball_z',
                            'ball_i', 'ball_j', 'ball_k', 'ball_w']

                # manip_rows = ['cube_x', 'cube_y', 'cube_z']
                # manip_cols = ['ball_x', 'ball_y', 'ball_z']
                manip_rows, manip_cols = all_labels, all_labels
                
                # manipulability_ad = self.select_rows_cols(manipulability_ad,
                #                     rows=[all_labels.index(label) for label in manip_rows],
                #                     cols=[all_labels.index(label) for label in manip_cols]
                #                     )
                print('manipulability (ad): \n', manipulability_ad)

                # visualizing the autodiff manipulability
                # labeled_matrix_print(manipulability_ad, manip_rows, manip_cols)
                # matrix_to_heatmap(manipulability_ad, manip_rows, manip_cols, title="heatmap (ad)", vis=True)
                # plot_eigenvalues(manipulability_ad, manip_rows, manip_cols, title="eigenvalues (ad)", vis=True)
                # visualize_unit_ball(manipulability_ad, manip_rows, manip_cols, title="unit ball (ad)", vis=True)
                # cv2.waitKey(1)  # Small delay to display the image


                # print('manipulability (fd): \n', manipulability_fd)
                # print('determinant: ', np.linalg.det(manipulability_ad.numpy()))
                # print('trace: ', np.trace(manipulability_ad.numpy()))
                # print('eigenvalues: ', np.linalg.eigvals(manipulability_ad))
                # print('eigenvectors: ', np.linalg.eig(manipulability_ad))

                # # printing attributes of the state
                # for attr in dir(self.state_0):
                #     print("obj.%s = %r" % (attr, getattr(self.state_0, attr)))

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
robot = Example(stage, render=True)
robot.run()
