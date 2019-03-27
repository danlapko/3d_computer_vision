#! /usr/bin/env python3
from scipy.spatial.distance import squareform, pdist

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli
from itertools import count
from scipy import spatial
from params import *


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def find_corners(image_0, min_dist, q=0.03):
    # params for ShiTomasi corner detection
    image_0 = (image_0 * 255).astype(np.uint8)

    feature_params = dict(maxCorners=init_max_corners,
                          qualityLevel=q,
                          minDistance=min_dist)  # block_size - window for derivative calculation
    corns = cv2.goodFeaturesToTrack(image_0, mask=None, **feature_params)
    # print("\nINITIAL CORNS", corns.shape)

    corners = FrameCorners(
        np.arange(corns.shape[0]),
        np.squeeze(corns),
        np.array([5] * corns.shape[0])
    )
    return corners


def flow_corners(image_0, image_1, corners_0, max_corn_id, min_dist, max_new_corns, q):
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=lk_win_size,
                     maxLevel=lk_maxLevel,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               lk_iterative_search_stop_n_iter, lk_iterative_search_stop_epsilon),
                     minEigThreshold=lk_min_eig_threshold)

    # calculate optical flow
    points_0 = corners_0.points.reshape((-1, 1, 2))
    image_0 = (image_0 * 255).astype(np.uint8)
    image_1 = (image_1 * 255).astype(np.uint8)

    corns, statuses, _ = cv2.calcOpticalFlowPyrLK(image_0, image_1, points_0, None, **lk_params)
    statuses = np.squeeze(statuses) == 1

    ids = corners_0.ids[statuses]
    points = np.squeeze(corns[statuses])
    sizes = corners_0.sizes[statuses]

    new_corners = find_corners(image_1, min_dist, q=q)
    new_points = new_corners.points
    pointsKDTree = spatial.KDTree(points)

    distances, _ = pointsKDTree.query(new_points)  # nearest points
    args_dists = np.argsort(distances)
    new_indxs = distances > min_dist
    new_indxs[args_dists[:-max_new_corns]] = False

    new_points = new_points[new_indxs]
    new_ids = np.arange(max_corn_id + 1, max_corn_id + 1 + new_points.shape[0]).reshape(-1, 1)
    new_sizes = new_corners.sizes[new_indxs] * 2

    # print(" SHAPES:", ids.shape, points.shape, sizes.shape, new_ids.shape, new_points.shape, new_sizes.shape)

    ids = np.concatenate((ids, new_ids), axis=0)
    points = np.concatenate((points, new_points), axis=0)
    sizes = np.concatenate((sizes, new_sizes), axis=0)

    max_corn_id += new_points.shape[0]

    corners = FrameCorners(
        ids,
        points,
        sizes
    )

    distances.sort()
    # print(f" NEXT: prev {corners_0.ids.shape}, next {corners.ids.shape} ")

    return corners, max_corn_id


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]

    corners = find_corners(image_0, min_dist=init_min_dist_btw_corners, q=init_min_quality_of_corners)
    max_corn_id = corners.ids.max()

    builder.set_corners_at_frame(0, corners)
    for i_frame, image_1 in enumerate(frame_sequence[1:], 1):
        corners, max_corn_id = flow_corners(image_0, image_1, corners, max_corn_id,
                                            min_dist=update_min_dist_btw_corners,
                                            max_new_corns=update_max_new_corners,
                                            q=update_min_quality_of_corners)
        builder.set_corners_at_frame(i_frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
