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
from pair_warp.utils.autograd import *
from warp.envs import WarpEnv
# from pair_warp.envs.warp_env import WarpEnv
from pair_warp.envs.environment import Environment, RenderMode


wp.init()


class Example:
    frame_dt = 1.0 / 60.0

    episode_duration = 6.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0

    def __init__(self, stage=None, render=True):
        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        self.num_bodies = 8
        self.scale = 0.8
        self.ke = 1.e+2
        self.kd = 250.0
        self.kf = 500.0

        # boxes
        for i in range(1):
            b = builder.add_body(origin=wp.transform((i, 1.0, 0.0), wp.quat_identity()))

            s = builder.add_shape_box(
                pos=(0.0, 0.0, 0.0),
                hx=0.5 * self.scale,
                hy=0.5 * self.scale,
                hz=0.5 * self.scale,
                body=i,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
            )

        # spheres
        for i in range(1):
            b = builder.add_body(origin=wp.transform((i, 1.0, 4.0), wp.quat_identity()))

            s = builder.add_shape_sphere(
                pos=(0.0, 0.0, 0.0), radius=0.25 * self.scale, body=b, ke=self.ke, kd=self.kd, kf=self.kf
            )

        # # capsules
        # for i in range(self.num_bodies):
        #     b = builder.add_body(origin=wp.transform((i, 1.0, 6.0), wp.quat_identity()))

        #     s = builder.add_shape_capsule(
        #         pos=(0.0, 0.0, 0.0),
        #         radius=0.25 * self.scale,
        #         half_height=self.scale * 0.5,
        #         up_axis=0,
        #         body=b,
        #         ke=self.ke,
        #         kd=self.kd,
        #         kf=self.kf,
        #     )

        # # initial spin
        # for i in range(len(builder.body_qd)):
        builder.body_qd[1] = (0.0, 0.0, 0.0, 0.0, 0.0, -2.0)

        # # meshes
        # bunny = self.load_mesh(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"), "/bunny/bunny")
        # for i in range(self.num_bodies):
            
        #     b = builder.add_body(origin=wp.transform(
        #         (i*0.5*self.scale, 1.0 + i*1.7*self.scale, 4.0 + i*0.5*self.scale),
        #         wp.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi*0.1*i)))

        #     s = builder.add_shape_mesh(
        #             body=b,
        #             mesh=bunny,
        #             pos=(0.0, 0.0, 0.0),
        #             scale=(self.scale*1.73, self.scale*0.73, self.scale*0.73),
        #             ke=self.ke,
        #             kd=self.kd,
        #             kf=self.kf,
        #             # density=1e0,
        #             thickness=0.0
        #         )
            
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

        # s = builder.add_shape_mesh(
        #     body=b,
        #     mesh=bunny,
        #     pos=(0.0, 0.0, 0.0),
        #     scale=(self.scale, self.scale, self.scale),
        #     ke=self.ke,
        #     kd=self.kd,
        #     kf=self.kf,
        #     density=1e3,
        #     )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.model.integrator = self.integrator # had to add this line for some reason

        # -----------------------
        # set up Usd renderer
        if (self.enable_rendering):
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=0.5)

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        return wp.sim.Mesh(points, indices)

    def update(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    @wp.kernel
    def manipulability_kernel(state_in: wp.array(dtype=wp.transformf), state_out: wp.array(dtype=wp.transformf)):
        # We want to compute the Jacobian of this kernel to get the manipulability matrix
        # derivative of state_out with respect to state_in
        pass

    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.state_0.body_q.requires_grad = True
        self.state_0.body_qd.requires_grad = True
        self.state_0.body_f.requires_grad = True
        self.state_1.body_q.requires_grad = True
        self.state_1.body_qd.requires_grad = True
        self.state_1.body_f.requires_grad = True

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        profiler = {}

        # # create update graph
        # wp.capture_begin()

        # simulate
        self.update()

        # graph = wp.capture_end()

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=False, dict=profiler):

            for f in range(0, self.episode_frames):
                
                # with wp.ScopedTimer("simulate", active=False):
                #     # wp.capture_launch(graph)
                #     self.update()
                self.sim_time += self.frame_dt

                if (self.enable_rendering):
                    with wp.ScopedTimer("render", active=False):
                        self.render()

                cube_pose_np = self.state_1.body_q.numpy()[0]
                cube_vel_np = self.state_1.body_qd.numpy()[0]
                ball_pose_np = self.state_1.body_q.numpy()[1]
                ball_vel_np = self.state_1.body_qd.numpy()[1]

                manipulability = np.empty((14, 14), dtype=np.float32) # we want to compute the Jacobian of the cube output with respect to the ball input.
                tape = wp.Tape()
                with tape:
                    with wp.ScopedTimer("simulate", active=False):
                        # wp.capture_launch(graph)
                        self.update()
                    # output_buffer =  wp.launch(self.manipulability_kernel, dim=14, inputs=[self.state_0.body_q], outputs=[self.state_1.body_q], device="cuda")
                    wp.launch(self.manipulability_kernel, dim=14, inputs=[self.state_0.body_q], outputs=[self.state_1.body_q], device="cuda")
                    output_buffer = self.state_1.body_q
                for output_index in range(14):
                    # select which row of the Jacobian we want to compute
                    select_index = np.zeros(14)
                    select_index[output_index] = 1.0
                    e = wp.array(select_index, dtype=wp.transformf)
                    # pass input gradients to the output buffer to apply selection
                    tape.backward(grads={output_buffer: e})
                    q_grad_i = tape.gradients[self.state_0.body_q]
                    manipulability[output_index, :] = q_grad_i.numpy().flatten().astype(np.float32)
                    tape.zero()

                print('manipulability: ', manipulability)

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
robot = Example(stage, render=True)
robot.run()
