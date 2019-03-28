#! /usr/bin/env python3

from typing import List, Tuple

import cv2
import numpy as np

import frameseq
from _camtrack import *
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
from params import *


def create_P_and_V(R, t):
    P = Pose(R.T, R.T @ -t)
    V = pose_to_view_mat3x4(P)
    return P, V


def real_P_from_4Rts(R1, R2, t, corresps, intrin_mat, triangl_params):
    V0 = eye3x4()

    real_pts_ids = []
    real_pts3d = []
    real_P = None

    for R, t_ in ((R1, t), (R1, -t), (R2, t), (R2, -t)):
        P, V1 = create_P_and_V(R, t_)
        pts3d, pts_ids = triangulate_correspondences(corresps, V0, V1, intrin_mat, triangl_params)

        if len(pts_ids) > len(real_pts_ids):
            real_pts_ids = pts_ids
            real_pts3d = pts3d
            real_P = P
    return real_P, real_pts_ids, real_pts3d


def find_Pose_by_2_frames(corners1, corners2, intrin_mat, triangl_params):
    corresps = build_correspondences(corners1, corners2)

    if len(corresps.ids) < min_corresp_count:
        print(f' inliers 0/{len(corners2.ids)} (too few corresponding corners)')
        return None, [], []

    H, hom_inliers_mask = cv2.findHomography(corresps.points_1,
                                             corresps.points_2,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=homo_ransacReprojThreshold,
                                             confidence=homo_ransacConfidence)
    homo_n_inliers = np.count_nonzero(hom_inliers_mask)

    E, essen_inliers_mask = cv2.findEssentialMat(corresps.points_1,
                                                 corresps.points_2,
                                                 cameraMatrix=intrin_mat, method=cv2.RANSAC,
                                                 prob=essen_ransacConfidence,
                                                 threshold=essen_threshold)
    if E.shape != (3, 3):
        print(f' inliers 0/{len(corners2.ids)} (refused by check E shape)')
        return None, [], []

    essen_n_inliers = np.count_nonzero(essen_inliers_mask)

    # If centers of cams are same then we can't calculate Essential matrix
    # At the same time in such case Homography between correspondeces should exist
    # So we check if number homography inliers greater then num of essential inliers
    if homo_n_inliers / essen_n_inliers > max_homo_to_essen_inliers_ratio:
        print(f' inliers 0/{len(corners2.ids)} (refused by homography check)')
        return None, [], []

    corresps = remove_correspondences_with_ids(corresps, np.where(essen_inliers_mask == 0)[0])

    R1, R2, t = cv2.decomposeEssentialMat(E)
    P, pts_ids, pts3d = real_P_from_4Rts(R1, R2, t, corresps, intrin_mat, triangl_params)

    print(f' inliers {len(pts_ids)}/{len(corners2.ids)}' + (' (refused by triangulation check)' if len(
        pts_ids) == 0 else ''))

    return P, pts_ids, pts3d


def initialize_camera_tracking(corner_storage, intrinsic_mat, triang_params):
    best_other_frame_idx = -1
    best_other_P = None
    best_other_pts_ids = []
    best_other_pts3d = []

    base_cornets = corner_storage[0]
    for i, other_corners in enumerate(corner_storage[1:], 1):
        print(f'initialize_camera_tracking: other_frame {i},', end="")

        P, pts_ids, pts3d = find_Pose_by_2_frames(base_cornets, other_corners, intrinsic_mat, triang_params)

        if len(pts_ids) > len(best_other_pts_ids):
            best_other_frame_idx = i
            best_other_P = P
            best_other_pts_ids = pts_ids
            best_other_pts3d = pts3d

    return best_other_frame_idx, (best_other_P, best_other_pts_ids, best_other_pts3d)


def update_cloud_till_frame(corner_storage, Vs, till_frame_ix, intrin_mat, cloud_builder, triangl_params):
    base_corners = corner_storage[till_frame_ix]
    base_V = Vs[-1]

    added_pts_cnt = 0
    for other_ix, other_corners in enumerate(corner_storage[:till_frame_ix]):
        corresps = build_correspondences(other_corners, base_corners, ids_to_remove=cloud_builder.ids)

        if len(corresps.ids) == 0:
            continue

        pts, ids = triangulate_correspondences(corresps, Vs[other_ix], base_V, intrin_mat, triangl_params)
        cloud_builder.add_points(ids, pts)

        added_pts_cnt += len(ids)
    return added_pts_cnt


def pnp(corner_storage, init_P, init_ix, cloud_builder, intrin_mat, triangl_params):
    base_V = eye3x4()
    Vs = [base_V]

    for cur_ix, cur_corners in enumerate(corner_storage[1:], 1):
        if cur_ix == init_ix:
            V = pose_to_view_mat3x4(init_P)
            Vs.append(V)
            update_cloud_till_frame(corner_storage, Vs, cur_ix, intrin_mat, cloud_builder, triangl_params)
            continue

        cloud_points, ixs_in_cur_corners_ids = cloud_builder.get_points_by_ids(cur_corners.ids)

        if len(cloud_points) < min_n_cloud_points:
            break

        retval, rvec, tvec, inliers_idxs = cv2.solvePnPRansac(cloud_points,
                                                              cur_corners.points[ixs_in_cur_corners_ids],
                                                              cameraMatrix=intrin_mat,
                                                              distCoeffs=np.array([]),
                                                              flags=cv2.SOLVEPNP_EPNP)
        if not retval:
            break

        V = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

        Vs.append(V)

        n_new_points = update_cloud_till_frame(corner_storage, Vs, cur_ix, intrin_mat, cloud_builder,
                                               triangl_params)
        print(f'pnp: frame {cur_ix}/{len(corner_storage)}, build new pos by {len(inliers_idxs)}/{len(cloud_points)} '
              f'inliers, cloud_size {len(cloud_builder.ids)}, added {n_new_points} new points ')

    return Vs, cloud_builder


def _track_camera(corner_storage, intrinsic_mat):
    init_triangl_params = TriangulationParameters(max_reprojection_error=init_max_reprojection_error,
                                                  min_triangulation_angle_deg=init_min_triangulation_angle_deg,
                                                  min_depth=init_min_depth)

    update_triangl_params = TriangulationParameters(max_reprojection_error=update_max_reprojection_error,
                                                    min_triangulation_angle_deg=update_min_triangulation_angle_deg,
                                                    min_depth=update_min_depth)

    init_frame_idx, (init_P, init_pts_ids, init_pts3d) = initialize_camera_tracking(corner_storage,
                                                                                    intrinsic_mat,
                                                                                    init_triangl_params)

    if init_frame_idx == -1:
        return None, None

    cloud_builder = PointCloudBuilder()
    cloud_builder.add_points(init_pts_ids, init_pts3d)

    print(f'\n--> INITIALIZED by {init_frame_idx} other_frame. Init cloud size {len(cloud_builder.ids)}\n')

    return pnp(corner_storage, init_P, init_frame_idx, cloud_builder, intrinsic_mat, update_triangl_params)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    view_mats, point_cloud_builder = _track_camera(
        corner_storage,
        intrinsic_mat
    )
    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    create_cli(track_and_calc_colors)()
