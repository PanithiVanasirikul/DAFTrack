import numpy as np

from .basetrack import BaseTrack, TrackState
# from .UnscentedGroundRF_filter import UnscentedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from .UnscentedGroundRF_matching import calc_distance, calc_distance_pix, linear_assignment, mahalanobis_distance, negative_log_likelihood
from .UnscentedGroundRF_utils import batched_rotated_box_intersections

import random

def f_world(x, dt):
    # F = np.array([[1, dt, 0, 0],
    #               [0,  1, 0, 0],
    #               [0,  0, 1, dt],
    #               [0,  0, 0, 1]], dtype=float)
    F = np.array([[1,  0, dt,  0],
                  [0,  1,  0, dt],
                  [0,  0, 1,   0],
                  [0,  0, 0,   1]], dtype=float)
    return F@x

def h_world_to_pixel(x, **hx_args):
    scale = 100
    x = x/scale
    # multiplier = (2/np.pi)*(np.arctan(np.sqrt(x[0]**2 + x[2]**2))/(np.sqrt(x[0]**2 + x[2]**2)))
    # return [multiplier*x[0]*hx_args["radius"] + hx_args["center"][0], multiplier*x[2]*hx_args["radius"] + hx_args["center"][1]]

    multiplier = (2/np.pi)*(np.arctan(np.sqrt(x[0]**2 + x[1]**2))/(np.sqrt(x[0]**2 + x[1]**2)))
    return [multiplier*x[0]*hx_args["radius"] + hx_args["center"][0], multiplier*x[1]*hx_args["radius"] + hx_args["center"][1]]
    # return [x[0], x[2]]

def pix_to_world(pixel_coords):
    scale = 100 # for Numerical Stability
    norm = np.linalg.norm(pixel_coords, axis=-1, keepdims=True)
    # norm = np.clip(norm, 1e-6, 0.99)
    return (pixel_coords / norm) * np.tan(np.pi * norm / 2) * scale

def world_to_pix(world_coords):
    scale = 100
    world_coords = world_coords/scale
    norm = np.linalg.norm(world_coords, axis=-1, keepdims=True)
    return (world_coords / norm) * np.arctan(norm)* 2/np.pi


def xywhr_to_meancov_batch(xywhr_batch):
    """
    Transform batch of xywhr to Gaussian distribution.
    
    Parameters:
    - xywhr_batch: np.ndarray of shape (N, 5), where N is the number of boxes.
    
    Returns:
    - means: np.ndarray of shape (N, 2)
    - covs: np.ndarray of shape (N, 2, 2)
    """
    xywhr_batch = np.asarray(xywhr_batch)
    x = xywhr_batch[:, 0]
    y = xywhr_batch[:, 1]
    w = xywhr_batch[:, 2]
    h = xywhr_batch[:, 3]
    r = np.deg2rad(xywhr_batch[:, 4])
    cos_r = np.cos(r)
    sin_r = np.sin(r)

    R = np.stack([
        np.stack([cos_r, -sin_r], axis=-1),
        np.stack([sin_r,  cos_r], axis=-1)
    ], axis=-2)  # shape (N, 2, 2)

    S = 0.5 * np.stack([
        np.stack([w, np.zeros_like(w)], axis=-1),
        np.stack([np.zeros_like(h), h], axis=-1)
    ], axis=-2)  # shape (N, 2, 2)

    sigma = R @ (S ** 2) @ np.transpose(R, (0, 2, 1))  # shape (N, 2, 2)
    means = np.stack([x, y], axis=-1)  # shape (N, 2)

    return means, sigma/25

class UTrack(UKF, BaseTrack):
    def __init__(self, dim_x, dim_z, dt, hx, fx, center, radius, points, bbox, score, mu_pixel, cov_pixel, mu_world, cov_world,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):
        super().__init__(dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=sqrt_fn, x_mean_fn=x_mean_fn, z_mean_fn=z_mean_fn,
                 residual_x=residual_x,
                 residual_z=residual_z) 
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.center = center
        self.radius = radius
        
        self.bbox = bbox
        self.score = score

        self.mu_pixel = mu_pixel
        self.cov_pixel = cov_pixel

        self.mu_world = mu_world
        self.cov_world = cov_world
        # keep in mind that this gets written only once, at the init.

        # self.x[0] = mu_world[0]
        # self.x[2] = mu_world[1]
        self.x[0] = mu_world[0]
        self.x[1] = mu_world[1]

        # std_P = [
        #      2 * self._std_weight_position * cov_world[0,0],
        #     10 * self._std_weight_velocity * cov_world[0,0],
        #      2 * self._std_weight_position * cov_world[1,1],
        #     10 * self._std_weight_velocity * cov_world[1,1],
        #     ]
        
        # std_Q = [
        #     1*self._std_weight_position * np.sqrt(cov_world[0,0]),
        #     1*self._std_weight_velocity * np.sqrt(cov_world[0,0]),
        #     1*self._std_weight_position * np.sqrt(cov_world[1,1]),
        #     1*self._std_weight_velocity * np.sqrt(cov_world[1,1]),
        #     ]
        std_Q = [
            1*self._std_weight_position * np.sqrt(cov_world[0,0]),
            1*self._std_weight_position * np.sqrt(cov_world[1,1]),
            1*self._std_weight_velocity * np.sqrt(cov_world[0,0]),
            1*self._std_weight_velocity * np.sqrt(cov_world[1,1]),
            ]

        # self.P = np.diag(np.square(std_P))
        self.P[:2, :2] = cov_world
        self.P[2:, 2:] = 5 * cov_world
        # self.P = self.P[np.ix_([0, 2, 1, 3], [0, 2, 1, 3])]


        self.Q = np.diag(np.square(std_Q))

        # self.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
        # self.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)

        # self.P[0,0] = cov_world[0,0]
        # self.P[0,2] = cov_world[0,1]
        # self.P[2,0] = cov_world[1,0]
        # self.P[2,2] = cov_world[1,1]

        # self.P[1,1] = cov_world[0,0]*11.5
        # self.P[3,3] = cov_world[1,1]*11.5
        
        self.is_activated = False
        self.tracklet_len = 0
        # self.newTrack = self.bbox

        # self.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
        # self.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)
        # We have to vary R as the measurements would have different Rs
    
    def update_track(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        # self.newTrack = new_track.bbox
        # self.mean, self.covariance = self.update(self.mean, self.covariance, new_track.meancov)
        self.R = new_track.cov_pixel
        # self.R = new_track.covMatrix[1]
        # std_R = [0.1, 0.1]
        # self.R = np.diag(np.square(std_R))
        # breakpoint()
        # self.update(new_track.mu_pixel)
        self.update(new_track.mu_pixel*self.radius + self.center, **{'center': self.center, 'radius': self.radius})
        # self.update(new_track.mu_world)

        self.bbox = new_track.bbox
        self.score = new_track.score

        self.mu_pixel = new_track.mu_pixel
        self.cov_pixel = new_track.cov_pixel

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # breakpoint()
        self.R = new_track.cov_pixel
        # self.R = new_track.cov_world
        # self.update(new_track.covMatrix[0])
        # self.update(new_track.mu_pixel)
        self.update(new_track.mu_pixel*self.radius + self.center, **{'center': self.center, 'radius': self.radius})
        # self.update(new_track.mu_world)


        self.bbox = new_track.bbox
        self.score = new_track.score

        self.mu_pixel = new_track.mu_pixel
        self.cov_pixel = new_track.cov_pixel

        
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        # if new_id:
        #     self.track_id = self.next_id()
        # new_id is always False
        self.score = new_track.score
    
    @property
    def covMatrix(self):
        """
        The Gaussian distibution of the current track.
        """
        # mean = np.array(self.x[[0,2]])
        # cov = self.P[[0,2],:][:, [0,2]]

        mean = np.array(self.x[[0,1]])
        cov = self.P[[0,1],:][:, [0,1]]

        ret = [mean,cov]
        return ret
    
    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

'''Main Tracker code '''
class UnscentedGroundRFTrack(object):
    def __init__(self, args, center, radius):
        self.reset(args)
        self.args = args
        self.track_thresh = args.track_thresh
        self.track_buffer = args.track_buffer
        self.match_thresh = args.match_thresh

        self.det_thresh = self.track_thresh + 0.1
        self.buffer_size = int((args.fps / 30.0 ) * self.track_buffer)
        self.max_time_lost = self.buffer_size     

        self.height_sum = 1
        self.height_conf_sum = 1
        # self.match1_numerator = 100/args.fps
        # self.match1_numerator = 75
        # self.match1_numerator = 100
        # self.match1 = 150/1.7
        self.match1 = args.match1
        self.use_camera_height = args.use_camera_height
        self.estimate_camera_height = args.camera_height == -1
        self.camera_height = args.camera_height
        # self.match1 = 150/1.7
        # self.match1 = 30
        # self.match2 = 150
        self.match2 = 0.7
        # self.match2 = 80

        self.center = center
        self.radius = radius

        self.match1_thresh_scale = 1.3
    
    def reset(self,args):
        BaseTrack._count = 0
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
    
    def update(self, output_results):
        # indices = np.random.permutation(output_results.shape[0])
        # output_results = output_results[indices]
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        '''cx,cy,w,h,angle'''
        scores = output_results[:, 5]
        bboxes = output_results[:, :5]
        bboxes_norm = bboxes.copy()
        bboxes_norm[:, 0] -= self.center[0]
        bboxes_norm[:, 1] -= self.center[1]
        bboxes_norm[:, :4] /= self.radius
        if self.estimate_camera_height:
            inter_points, valid = batched_rotated_box_intersections(bboxes_norm)
            valid[:] = True
            d_pixels = np.linalg.norm(inter_points[valid], axis=2)
            z = 1.70/(1 - np.tan(np.pi*np.min(d_pixels, axis=1)/2)/np.tan(np.pi*np.max(d_pixels, axis=1)/2))
            self.height_sum += (((z-0.85)*scores[valid])).sum()
            self.height_conf_sum += scores[valid].sum()
            self.camera_height = self.height_sum/self.height_conf_sum

        all_mus_pixel, all_covs_pixel = xywhr_to_meancov_batch(bboxes_norm)

        points_pix_to_world = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=0.)
        points_UKF = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=0.)

        sigmas = [points_pix_to_world.sigma_points(all_mus_pixel[i], all_covs_pixel[i]) for i in range(len(all_mus_pixel))]
        sigmas_transformed = [pix_to_world(e) for e in sigmas]
        all_mu_cov_world = [unscented_transform(e, points_pix_to_world.Wm, points_pix_to_world.Wc, residual_fn=np.subtract) for e in sigmas_transformed]     

        remain_inds = scores > self.track_thresh 
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        '''High Score detections'''
        dets = bboxes[remain_inds]
        mus_pixel = all_mus_pixel[remain_inds]
        covs_pixel = all_covs_pixel[remain_inds]
        mus_covs_world = [all_mu_cov_world[i] for i in range(len(remain_inds)) if remain_inds[i]]
        # mus_covs_world = [mu for mu, remain in zip(all_mu_cov_world, remain_inds) if remain]
        scores_keep = scores[remain_inds]
        '''Low Score detections'''
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        mus_pixel_second = all_mus_pixel[inds_second]
        covs_pixel_second = all_covs_pixel[inds_second]
        mus_covs_world_second = [all_mu_cov_world[i] for i in range(len(inds_second)) if inds_second[i]]
        scores_second = scores[inds_second]
        '''Detections to track'''
        if len(dets) > 0:
            detections = [UTrack(4, 2, 1, h_world_to_pixel, f_world, self.center, self.radius, points_UKF, bbox, score, mu_pixel, cov_pixel,
                              mu_cov_world[0], mu_cov_world[1],) for bbox, score, mu_pixel, cov_pixel, mu_cov_world in zip(dets, scores_keep, mus_pixel, covs_pixel, mus_covs_world)]
        else:
            detections = []
        ''' Split currently tracked into last seen or not'''
        unconfirmed = []
        tracked_stracks = []  # type: list[UTrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        for e in strack_pool:
            e.predict()
        for e in unconfirmed:
            if (e.sigmas_f == 0).sum() == e.sigmas_f.shape[0]*e.sigmas_f.shape[1]:
                e.predict()
        dists = calc_distance(strack_pool, detections, match=self.args.match, frame_size=self.args.frame_size, frame_id=self.frame_id)
        if self.use_camera_height:
            dists *= self.camera_height
        dists[dists > self.match1*self.match1_thresh_scale] = self.match1*self.match1_thresh_scale
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match1)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update_track(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [UTrack(4, 2, 1, f_world, h_world_to_pixel, self.center, self.radius, points_UKF, bbox, score, mu_pixel, cov_pixel,
                              mu_cov_world[0], mu_cov_world[1],) for bbox, score, mu_pixel, cov_pixel, mu_cov_world in zip(dets_second, scores_second, mus_pixel_second, covs_pixel_second, mus_covs_world_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = calc_distance_pix(r_tracked_stracks, detections_second, match=self.args.match, frame_size=self.args.frame_size, center=self.center, radius=self.radius)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=self.match2)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update_track(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # this should never happen because the r_tracked_stracks are selected from only the trackers that TrackState.Tracked
                # breakpoint()
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = calc_distance(unconfirmed, detections, match=self.args.match, frame_size=self.args.frame_size)
        if self.use_camera_height:
            dists *= self.camera_height

        dists[dists > self.match1*self.match1_thresh_scale] = self.match1*self.match1_thresh_scale
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=self.match1)


        for itracked, idet in matches:
            try:
                unconfirmed[itracked].update_track(detections[idet], self.frame_id)
            except:
                breakpoint()
            activated_stracks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]

            if track.score < self.det_thresh:
                continue

            track.activate(self.frame_id)
            activated_stracks.append(track)
        
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]

        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks,match=self.args.match)
        # can I actually comment this for faster performance?
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        for track in self.tracked_stracks:
            if np.isnan(track.x).any() or np.isnan(track.P).any():
                breakpoint()

        return output_stracks
        # return self.tracked_stracks
    
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb,match ):
    pdist = calc_distance(stracksa, stracksb,match=match)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb