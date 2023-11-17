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


class Example():
# class Example(WarpEnv):
    frame_dt = 1.0 / 60.0

    episode_duration = 6.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0

    def __init__(self, stage=None, render=True):
    #     super().__init__(
    #         num_envs=1,
    #         num_obs=1,
    #         num_act=1,
    #         episode_length=self.episode_frames,
    #         seed=0,
    #         no_grad=False,
    #         render=True,
    #         stochastic_init=False,
    #         device="cuda",
    #         env_name="toy_env",
    #         render_mode=RenderMode.OPENGL,
    #         stage_path=None,
    # )
        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        # self.num_bodies = 1
        self.scale = 0.8
        self.ke = 1.e+2
        self.kd = 250.0
        self.kf = 500.0

        # boxes
        b = builder.add_body(origin=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))

        s = builder.add_shape_box(
            pos=(0.0, 0.0, 0.0),
            hx=0.5 * self.scale,
            hy=0.5 * self.scale,
            hz=0.5 * self.scale,
            body=0,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
        )

        # spheres
        b = builder.add_body(origin=wp.transform((0.0, 1.0, 2.0), wp.quat_identity()))

        s = builder.add_shape_sphere(
            pos=(-1.0, 0.0, 0.0), radius=0.5 * self.scale, body=b, ke=self.ke, kd=self.kd, kf=self.kf
        )

        # printing out all attributes of builder
        for attr in dir(builder):
            print("builder.{} = {}".format(attr, getattr(builder, attr)))
   
        # # initial spin/velocity
        # for i in range(len(builder.body_qd)):
        #     # builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)
        #     builder.body_qd[i] = (0.0, 0.0, 0.0, 0.1, 0.0, 0.0)

        builder.body_qd[1] = (0.0, 0.0, 0.0, 1.0, 0.0, -2.0) # ball initial velocity
        
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
        self.model.integrator = self.integrator # had to add this line for some reason

        # -----------------------
        # set up Usd renderer
        if (self.enable_rendering):
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage, scaling=0.5)

        # self.init_sim()
        # self.setup_autograd_vars()
        

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

            # print('state_0: ', self.state_0.body_q)
            # print('state_1: ', self.state_1.body_q)
            # for attr in dir(self.state_0):
            #     print("state_0.{} = {}".format(attr, getattr(self.state_0, attr)))

    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    @wp.kernel
    def manipulability_kernel(cube_position: wp.array(dtype=wp.vec3f), ball_position: wp.array(dtype=wp.vec3f)):
    # def manipulability_kernel(state: wp.array(dtype=wp.transformf), new_state: wp.array(dtype=wp.transformf)):
        # We want to compute the Jacobian of this kernel to get the manipulability matrix
        # cube_pose = state[0]
        # ball_pose = state[1]
        # new_state = state
        pass

    def run(self, render=True):
        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        self.update()

        graph = wp.capture_end()

        # wp.launch(self.manipulability_kernel, dim=1, inputs=[self.state_1.body_q], outputs=[self.state_1.body_q], device="cuda")

        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=False, dict=profiler):


            for f in range(0, self.episode_frames):
                
                with wp.ScopedTimer("simulate", active=False):
                    wp.capture_launch(graph)
                self.sim_time += self.frame_dt

                if (self.enable_rendering):

                    with wp.ScopedTimer("render", active=False):
                        self.render()

                # # printing out all attributes of the model
                # for attr in dir(self.model):
                #     print("model.{} = {}".format(attr, getattr(self.model, attr)))

                # print("model.body_q = {}".format(self.model.body_q))
                # print('state_1 pos: ', self.state_1.body_q)
                # print('state_1 vel: ', self.state_1.body_qd)


                cube_pose_np = self.state_1.body_q.numpy()[0]
                cube_vel_np = self.state_1.body_qd.numpy()[0]
                ball_pose_np = self.state_1.body_q.numpy()[1]
                ball_vel_np = self.state_1.body_qd.numpy()[1]

                # print('cube_pose_np: ', cube_pose_np)

                # state_np = self.state_1.body_q.numpy()
                # state_wp = wp.from_numpy(self.state_1.body_q, dtype=wp.transformf)
                # print('self.state_1.body_q: ', self.state_1.body_q)
                # print('state_np: ', state_np)
                # print('state_wp: ', state_wp)

                # cube_pose = wp.from_numpy(cube_pose_np, dtype=wp.transformf)
                # cube_vel = wp.from_numpy(cube_vel_np, dtype=wp.transformf)
                # ball_pose = wp.from_numpy(ball_pose_np, dtype=wp.transformf)
                # ball_vel = wp.from_numpy(ball_vel_np, dtype=wp.transformf)

                cube_position = wp.from_numpy(cube_pose_np[0:3], dtype=wp.vec3f)
                cube_rotation = wp.from_numpy(cube_pose_np[3:7], dtype=wp.quatf)
                cube_linear_velocity = wp.from_numpy(cube_vel_np[0:3], dtype=wp.vec3f)
                cube_angular_velocity = wp.from_numpy(cube_vel_np[3:6], dtype=wp.vec3f)

                ball_position = wp.from_numpy(ball_pose_np[0:3], dtype=wp.vec3f)
                ball_rotation = wp.from_numpy(ball_pose_np[3:7], dtype=wp.quatf)
                ball_linear_velocity = wp.from_numpy(ball_vel_np[0:3], dtype=wp.vec3f)
                ball_angular_velocity = wp.from_numpy(ball_vel_np[3:6], dtype=wp.vec3f)

                # manipulability = kernel_jacobian(kernel=self.manipulability_kernel, dim=7, inputs=[self.state_1.body_q], outputs=[self.state_1.body_q])
                wp.launch(self.manipulability_kernel, dim=1, inputs=[ball_position], outputs=[cube_position], device="cuda")


                manipulability = kernel_jacobian(kernel=self.manipulability_kernel, dim=3, inputs=[ball_position], outputs=[cube_position])
                # tape = wp.Tape()
                # with tape:

                if f % 10 == 0:
                    # print('cube_pose: ', cube_pose)
                    # print('cube_vel: ', cube_vel)
                    # print('ball_pose: ', ball_pose)
                    # print('ball_vel: ', ball_vel)
                    print('manipulability: ', manipulability)
                

            wp.synchronize()

        self.renderer.save()

stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_manip_toy.usd")
robot = Example(stage, render=True)
robot.run()