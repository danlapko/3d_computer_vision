#! /usr/bin/env python3
import glm

__all__ = [
    'CameraTrackRenderer'
]

from typing import List, Tuple

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders
from OpenGL import GLUT
from OpenGL.arrays import vbo
import math

import data3d


def createP(fov_y, aspect_ratio, near, far):
    P = np.array([
        [1 / math.tan(fov_y / 2) / aspect_ratio, 0, 0, 0],
        [0, 1 / math.tan(fov_y / 2), 0, 0],
        [0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    return P


def rotate_R(A):
    return A @ np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])


def createM(R, t):
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    M = M
    return M


def createV(R, t):
    V = np.eye(4, dtype=np.float32)
    V[:3, :3] = R
    V[:3, 3] = t
    return np.linalg.inv(V)


def _build_example_program():
    example_vertex_shader = shaders.compileShader(
        """
        #version 140
        uniform mat4 mvp;

        in vec3 position;
        in vec3 in_color;
        out vec3 out_color;
        
        void main() {
            vec4 camera_space_position = mvp * vec4(position, 1.0);
            gl_Position = camera_space_position;
            out_color = in_color;
        }""",
        GL.GL_VERTEX_SHADER
    )
    example_fragment_shader = shaders.compileShader(
        """
        #version 140
        in vec3 out_color;
        out vec3 color;

        void main() {
            color = out_color;
        }""",
        GL.GL_FRAGMENT_SHADER
    )

    return shaders.compileProgram(
        example_vertex_shader, example_fragment_shader
    )


class CameraTrackRenderer:

    def __init__(self,
                 cam_model_files: Tuple[str, str],
                 tracked_cam_parameters: data3d.CameraParameters,
                 tracked_cam_track: List[data3d.Pose],
                 point_cloud: data3d.PointCloud):
        """
        Initialize CameraTrackRenderer. Load camera model, create buffer objects, load textures,
        compile shaders, e.t.c.

        :param cam_model_files: path to camera model obj file and texture. The model consists of
        triangles with per-point uv and normal attributes
        :param tracked_cam_parameters: tracked camera field of view and aspect ratio. To be used
        for building tracked camera frustrum
        :param point_cloud: colored point cloud
        """
        R_transform = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])

        self.tracked_cam_ts = np.array([pos.t_vec * [1, -1, -1] for pos in tracked_cam_track], dtype=np.float32)
        self.tracked_cam_Rs = np.array(
            [np.linalg.inv(R_transform) @ pos.r_mat @ R_transform for pos in tracked_cam_track],
            dtype=np.float32)
        self.tracked_cam_colors = np.array([(1, 1, 1) for _ in tracked_cam_track], dtype=np.float32)

        self.cloud_points = np.array([point * [1, -1, -1] for point in point_cloud.points], dtype=np.float32)
        self.cloud_colors = np.array(point_cloud.colors, dtype=np.float32)

        self.tracked_cam_near = 1
        self.tracked_cam_far = 20
        self.render_cam_near = 1
        self.render_cam_far = 50

        self.tracked_cam_M = None
        self.tracked_cam_P = createP(tracked_cam_parameters.fov_y, tracked_cam_parameters.aspect_ratio,
                                     self.render_cam_near, self.tracked_cam_far)

        self.tracked_cam_frustrum = np.array([[-1, -1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1],
                                              [-1, 1, -1],
                                              [-1, 1, 1], [-1, -1, 1], [-1, 1, 1],
                                              [1, 1, 1], [1, -1, 1], [1, 1, 1],
                                              [1, 1, -1], [1, -1, -1], [1, 1, -1],
                                              [-1, 1, - 1]],
                                             dtype=np.float32)

        self.tracked_cam_frustrum_colors = np.array([[1, 1, 0]] * len(self.tracked_cam_frustrum), dtype=np.float32)

        self.tracked_cam_point = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.tracked_cam_point_color = np.array([[1, 0, 1]] * len(self.tracked_cam_point), dtype=np.float32)

        # vbos
        self._cloud_points_vbo = vbo.VBO(self.cloud_points)
        self._cloud_colors_vbo = vbo.VBO(self.cloud_colors)

        self._tracked_cam_ts_vbo = vbo.VBO(self.tracked_cam_ts)
        self._tracked_cam_colors_vbo = vbo.VBO(self.tracked_cam_colors)

        self._tracked_cam_frustrum_vbo = vbo.VBO(self.tracked_cam_frustrum)
        self._tracked_cam_frustrum_colors_vbo = vbo.VBO(self.tracked_cam_frustrum_colors)

        self._tracked_cam_point = vbo.VBO(self.tracked_cam_point)
        self._tracked_cam_point_color = vbo.VBO(self.tracked_cam_point_color)

        # compile
        self._example_program = _build_example_program()

        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_DEPTH)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def display(self, camera_tr_vec, camera_rot_mat, camera_fov_y, tracked_cam_track_pos_float):
        """
        Draw everything with specified render camera position, projection parameters and 
        tracked camera position

        :param camera_tr_vec: vec3 position of render camera in global space
        :param camera_rot_mat: mat3 rotation matrix of render camera in global space
        :param camera_fov_y: render camera field of view. To be used for building a projection
        matrix. Use glutGet to calculate current aspect ratio
        :param tracked_cam_track_pos_float: a frame in which tracked camera
        model and frustrum should be drawn (see tracked_cam_track_pos for basic task)
        :return: returns nothing
        """

        # a frame in which a tracked camera model and frustrum should be drawn
        # without interpolation
        tracked_cam_track_pos = int(tracked_cam_track_pos_float)
        tracked_cam_R = self.tracked_cam_Rs[tracked_cam_track_pos]
        tracked_cam_t = self.tracked_cam_ts[tracked_cam_track_pos]

        h = GLUT.glutGet(GLUT.GLUT_SCREEN_HEIGHT)
        w = GLUT.glutGet(GLUT.GLUT_SCREEN_WIDTH)
        aspect_ratio = w / h

        # camera_tr_vec = (camera_tr_vec * [1, -1, -1]).astype(np.float32)
        M = np.eye(4, dtype=np.float32)
        V = createV(camera_rot_mat, camera_tr_vec)
        P = createP(camera_fov_y, aspect_ratio, self.render_cam_near, self.render_cam_far)
        mvp = P @ V @ M

        self.tracked_cam_M = createM(tracked_cam_R, tracked_cam_t)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self._render_example_point(mvp)

        GLUT.glutSwapBuffers()

    def _render_example_point(self, mvp):
        shaders.glUseProgram(self._example_program)

        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._example_program, 'mvp'),
            1, True, mvp)

        position_loc = GL.glGetAttribLocation(self._example_program, 'position')
        in_color_loc = GL.glGetAttribLocation(self._example_program, 'in_color')

        GL.glEnableVertexAttribArray(position_loc)
        GL.glEnableVertexAttribArray(in_color_loc)

        # render cloud ===============================================================
        self._cloud_points_vbo.bind()
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._cloud_points_vbo)
        self._cloud_points_vbo.unbind()

        self._cloud_colors_vbo.bind()
        GL.glVertexAttribPointer(in_color_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._cloud_colors_vbo)
        self._cloud_colors_vbo.unbind()

        GL.glDrawArrays(GL.GL_POINTS, 0, len(self.cloud_points))

        # render tracked_cameras track ===============================================================
        self._tracked_cam_ts_vbo.bind()
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._tracked_cam_ts_vbo)
        self._tracked_cam_ts_vbo.unbind()

        self._tracked_cam_colors_vbo.bind()
        GL.glVertexAttribPointer(in_color_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._tracked_cam_colors_vbo)
        self._tracked_cam_colors_vbo.unbind()

        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(self.tracked_cam_ts))

        # render tracked cam frustrum ===============================================================
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._example_program, 'mvp'),
            1, True, mvp @ self.tracked_cam_M @ np.linalg.inv(self.tracked_cam_P))

        self._tracked_cam_frustrum_vbo.bind()
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._tracked_cam_frustrum_vbo)
        self._tracked_cam_frustrum_vbo.unbind()

        self._tracked_cam_frustrum_colors_vbo.bind()
        GL.glVertexAttribPointer(in_color_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._tracked_cam_frustrum_colors_vbo)
        self._tracked_cam_frustrum_colors_vbo.unbind()

        GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(self.tracked_cam_frustrum))

        # render tracked cam point ===============================================================
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(self._example_program, 'mvp'),
            1, True, mvp @ self.tracked_cam_M)

        self._tracked_cam_point.bind()
        GL.glVertexAttribPointer(position_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._tracked_cam_point)
        self._tracked_cam_point.unbind()

        self._tracked_cam_point_color.bind()
        GL.glVertexAttribPointer(in_color_loc, 3, GL.GL_FLOAT,
                                 False, 0,
                                 self._tracked_cam_point_color)
        self._tracked_cam_point_color.unbind()

        GL.glDrawArrays(GL.GL_POINTS, 0, len(self.tracked_cam_point))

        # finish
        GL.glDisableVertexAttribArray(position_loc)
        GL.glDisableVertexAttribArray(in_color_loc)

        shaders.glUseProgram(0)
