# CORNERS params

init_max_corners = 2000
init_min_dist_btw_corners = 8
init_min_quality_of_corners = 0.02  # relative to best quality of found courners

lk_win_size = (7, 7)  # winSize - size of the search window at each pyramid level.
lk_maxLevel = 2  # 0-based maximal pyramid level number; if set to 0, pyramids are not used
lk_min_eig_threshold = 1e-3
lk_iterative_search_stop_n_iter = 30
lk_iterative_search_stop_epsilon = 0.001

update_min_dist_btw_corners = 8
update_max_new_corners = 50
update_min_quality_of_corners = 0.02

# CAMTRACKER params

homo_ransacReprojThreshold = 2  # maximum allowed reprojection error in pixels (usually from 1 to 10)
homo_ransacConfidence = 0.999  # between 0 and 1
essen_threshold = 2  # max allowed distance from point to epipolar line in pixels (usually from 1 to 3)
essen_ransacConfidence = 0.999  # between 0 and 1

max_homo_to_essen_inliers_ratio = 1  # threshold  of n_hom_inliers/n_essentila_inliers

min_corresp_count = 5
essen_min_inlier_count = 5
essen_min_inlier_ratio = 0.1

init_max_reprojection_error = 1.0
init_min_triangulation_angle_deg = 4
init_min_depth = 0.1

update_max_reprojection_error = 1.0
update_min_triangulation_angle_deg = 1
update_min_depth = 0.1

min_n_cloud_points = 4
