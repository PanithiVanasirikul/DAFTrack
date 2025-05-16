import numpy as np
import scipy
import lap
import torch
#from pycocotools import mask as maskUtils


def linear_assignment(cost_matrix, thresh, use_lap=False):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
 
    if use_lap:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0 and cost_matrix[ix, mx] <= thresh]

        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        x, y = scipy.optimize.linear_sum_assignment(cost_matrix)
        matches = np.asarray([[x[i], y[i]] for i in range(len(x)) if cost_matrix[x[i], y[i]] <= thresh])

        if len(matches) == 0:
            unmatched_a = list(np.arange(cost_matrix.shape[0]))
            unmatched_b = list(np.arange(cost_matrix.shape[1]))
        else:
            unmatched_a = list(set(np.arange(cost_matrix.shape[0])) - set(matches[:, 0]))
            unmatched_b = list(set(np.arange(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b

def kld(pred, target,  tau=1.0, alpha=1.0, sqrt=True,beta = 1.5):
    """
    Kullback-Leibler Divergence.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        sqrt (bool): Whether to sqrt the distance. Defaults to True.
        beta (float): Defaults to 1.5.

    Returns:
        similarity (torch.Tensor)
    """
    xy_t, Sigma_t = target
    _shape = xy_t.shape

    xy_p = pred[0].repeat(_shape[0],1)
    Sigma_p = pred[1].repeat(_shape[0],1,1)

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    
    if sqrt:
        distance = distance.clamp(1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])

    res = (1 - 1 / (tau + torch.log1p(distance)))* beta
    return res

def gwd(pred, target, tau=1.0, alpha=2., normalize=True, beta = 1.6):
    """
        Gaussian Wasserstein distance.
    """
    xy_t, Sigma_t = target
    xy_p = pred[0].repeat(xy_t.shape[0],1)
    Sigma_p = pred[1].repeat(xy_t.shape[0],1,1)

    xy_distance = (xy_p - xy_t).square().sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    res =  (1 - 1 / (tau + torch.log1p(distance))) *beta
    return res


def bh_dist_torch(pred, target,beta = 2.2): 
    """
        Bhattacharyya distance.
    """ 
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target
    
    cov_avg = 0.5 * (Sigma_p + Sigma_t)  
  
    # Calculate the determinant of the average covariance  
    det_avg = torch.det(cov_avg)  

    zeros = (det_avg == 0).nonzero()

    if len(zeros) > 0 :
        for i,val in enumerate(zeros):
            cov_avg[val.item()][0][0]+=0.5
            cov_avg[val.item()][1][1]+=0.5
  
    # Calculate the inverse of the average covariance  
    cov_avg_inv = torch.inverse(cov_avg)  
  
    # Calculate the mean difference  
    mean_diff = xy_t - xy_p 

    # Calculate the Mahalanobis distance (term1) 
    term1 = torch.matmul(mean_diff.unsqueeze(1), torch.matmul(cov_avg_inv, mean_diff.unsqueeze(-1)))  
    
    # Calculate the Bhattacharyya distance  
    term2 = torch.log(det_avg / torch.sqrt(torch.det(Sigma_p) * torch.det(Sigma_t)))  
    
    # Combine term1 and term2
    b_distance = 0.125 * term1.squeeze() + 0.5 * term2 

    res = (1 - 1 / (1 + torch.log1p(torch.sqrt(b_distance)))) *beta
    
    return res


def mod_ratio(sigma,mplier= 1.3,dynamic=True,xy=None,frame_size=None):
    
    sig_copy = sigma.clone()

    if dynamic:
        assert frame_size is not None
        
        mid = np.array(frame_size)/2
        dist = np.abs(xy - mid)/mid
        mplier = ((1- torch.mean(dist)) * (mplier-1)) + 1

    return sig_copy * mplier

import numpy as np
import torch

def mahalanobis_distance(atracks, btracks) -> torch.Tensor:
    """
    Compute Mahalanobis distance between points in amean and Gaussians defined by bmean and bcov.
    
    Parameters:
        amean: (N, 2) array of query points
        bmean: (M, 2) array of Gaussian means
        bcov: (M, 2, 2) array of Gaussian covariances
    
    Returns:
        (N, M) torch tensor of Mahalanobis distances
    """
    if len(btracks) == 0:
        return np.zeros([len(atracks),0])
    if len(atracks) == 0:
        return np.zeros([0,len(btracks)])
    a_means = torch.tensor(np.array([track.covMatrix[0] for track in atracks]))
    a_covs = torch.tensor(np.array([track.covMatrix[1] for track in atracks]))
    b_means = torch.tensor(np.array([track.covMatrix[0] for track in btracks]))
    b_covs = torch.tensor(np.array([track.covMatrix[1] for track in btracks]))
    N, M = a_means.shape[0], b_means.shape[0]

    # Convert to torch
    a = a_means.float().unsqueeze(1)  # (N, 1, 2)
    b = b_means.float().unsqueeze(0)  # (1, M, 2)
    cov = b_covs.float()              # (M, 2, 2)
    delta = a - b  # (N, M, 2)
    
    # Inverse covariance: (M, 2, 2)
    cov_inv = torch.linalg.inv(cov)  # (M, 2, 2)
    
    # Compute Mahalanobis distance using einsum
    dists = torch.einsum('nmk,mkl,nml->nm', delta, cov_inv, delta)  # (N, M)
    return (torch.sqrt(dists + 1e-7)).numpy()  # small epsilon to avoid sqrt(0)


def negative_log_likelihood(atracks, btracks) -> torch.Tensor:
    """
    Compute negative log-likelihood for points in amean under Gaussians defined by bmean and bcov.
    
    Parameters:
        amean: (N, 2) array of query points
        bmean: (M, 2) array of Gaussian means
        bcov: (M, 2, 2) array of Gaussian covariances
    
    Returns:
        (N, M) torch tensor of NLL values
    """
    if len(btracks) == 0:
        return np.zeros([len(atracks),0])
    if len(atracks) == 0:
        return np.zeros([0,len(btracks)])
    d = 2
    a_means = torch.tensor(np.array([track.covMatrix[0] for track in atracks]))
    a_covs = torch.tensor(np.array([track.covMatrix[1] for track in atracks]))
    b_means = torch.tensor(np.array([track.covMatrix[0] for track in btracks]))
    b_covs = torch.tensor(np.array([track.covMatrix[1] for track in btracks]))
    N, M = a_means.shape[0], b_means.shape[0]

    # Convert to torch
    a = a_means.float().unsqueeze(1)  # (N, 1, 2)
    b = b_means.float().unsqueeze(0)  # (1, M, 2)
    cov = b_covs.float()  

    delta = a - b  # (N, M, 2)

    cov_inv = torch.linalg.inv(cov)  # (M, 2, 2)
    cov_det = torch.linalg.det(cov)  # (M,)

    # Quadratic form
    quad_form = torch.einsum('nmk,mkl,nml->nm', delta, cov_inv, delta)  # (N, M)

    # Log determinant term (broadcasted)
    log_det = torch.log(cov_det + 1e-7).unsqueeze(0)  # (1, M)

    nll = 0.5 * (log_det + quad_form + d * np.log(2 * np.pi))  # (N, M)
    return nll.numpy()



def calc_distance(atracks, 
                btracks, 
                match='gwd',
                # ratio=1.3,
                # dynamic=True,
                frame_size=None, frame_id=None):

    """
    Calculates the cost matrix given the association method

    Args:
        atracks (list[STrack]): KF estimated bboxes.
        btracks (list[STrack]): New detected bboxes.
        match ([gwd,kld,bd]): Association method. Defaults to gwd.
        ratio (float): Buffer max expansion. Defaults to 1.3.
        dynamic (bool): Whether the buffer is dynamic. Defaults to True.
        frame_size (list[h,w]): Required for dynamic to calculate the distance to the centre. Defaults to 1.5.

    Returns:
        cost_matrix (np.ndarray)
    """
    # if frame_id == 17:
    #     breakpoint()

    if len(btracks) == 0:
        return np.zeros([len(atracks),0])
    a_means = torch.tensor(np.array([track.covMatrix[0] for track in atracks]))
    a_covs = torch.tensor(np.array([track.covMatrix[1] for track in atracks]))
    b_means = torch.tensor(np.array([track.covMatrix[0] for track in btracks]))
    b_covs = torch.tensor(np.array([track.covMatrix[1] for track in btracks]))

    # a_means *= 100
    # b_means *= 100
    # a_covs  *= 10000
    # b_covs  *= 10000

    cost_matrix = torch.zeros([a_means.shape[0], b_means.shape[0]])
    
    # a_covs= a_covs.clamp(1e-12)
    # b_covs= b_covs.clamp(1e-12)
    
    tracker = {
        'gwd':gwd,
        'bd': bh_dist_torch,
        'kld':kld,
    }

    # for i,pred1 in enumerate(a_means):
    #     # cov1 = mod_ratio(a_covs[i],ratio,dynamic=dynamic,xy=pred1,frame_size=frame_size)
    #     cost_matrix[i,:] = tracker[match]([pred1,a_covs[i]],[b_means, b_covs])
        # cost_matrix[i,:] = torch.clamp(tracker[match]([pred1,cov1],[b_means, b_covs]),max=1)
    if len(a_means) == 0 or len(b_means) == 0:
        cost_matrix = torch.zeros((len(a_means), len(b_means)))
    else:
        cost_matrix = torch.cdist(a_means, b_means)

    return cost_matrix.numpy()

def calc_distance_pix(atracks, 
                btracks, 
                match='gwd',
                frame_size=None,
                center=100,
                radius=200):

    """
    Calculates the cost matrix given the association method

    Args:
        atracks (list[STrack]): KF estimated bboxes.
        btracks (list[STrack]): New detected bboxes.
        match ([gwd,kld,bd]): Association method. Defaults to gwd.
        ratio (float): Buffer max expansion. Defaults to 1.3.
        dynamic (bool): Whether the buffer is dynamic. Defaults to True.
        frame_size (list[h,w]): Required for dynamic to calculate the distance to the centre. Defaults to 1.5.

    Returns:
        cost_matrix (np.ndarray)
    """

    if len(btracks) == 0:
        return np.zeros([len(atracks),0])
    a_means = torch.tensor(np.array([track.mu_pixel for track in atracks]))
    a_covs = torch.tensor(np.array([track.cov_pixel for track in atracks]))
    b_means = torch.tensor(np.array([track.mu_pixel for track in btracks]))
    b_covs = torch.tensor(np.array([track.cov_pixel for track in btracks]))

    if len(a_means) > 0:
        a_means = a_means*radius
        a_means[:, 0] = a_means[:, 0] + center[0]
        a_means[:, 1] = a_means[:, 1] + center[1]
        a_covs = a_covs*radius**2

    if len(b_means) > 0:
        b_means = b_means*radius
        b_means[:, 0] = b_means[:, 0] + center[0]
        b_means[:, 1] = b_means[:, 1] + center[1]
        b_covs = b_covs*radius**2

    cost_matrix = torch.zeros([a_means.shape[0], b_means.shape[0]])
    
    tracker = {
        'gwd':gwd,
        'bd': bh_dist_torch,
        'kld':kld,
    }

    for i,pred1 in enumerate(a_means):
        cost_matrix[i,:] = torch.clamp(tracker[match]([pred1,a_covs[i]],[b_means, b_covs]), max=1)
        # cost_matrix[i, :] = tracker[match]([pred1,a_covs[i]],[b_means, b_covs])

    return cost_matrix.numpy()

