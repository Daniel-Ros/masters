import numpy as np


def calc_bat(p1 : np.float32 ,p2 : np.float32,phi,mu1,mu2,cov1,cov2):
    SQRTP1P2 = np.sqrt(p1*p2)
    

    phi_d_mu  = phi.T  @ ( mu1 - mu2 )
    phi_d_mu_t = phi_d_mu.T

    phi_sig12_phi = ( phi.T @ (cov1+cov2) @ phi ) / 2
    phi_sig1_phi = phi.T @ (cov1) @ phi
    phi_sig2_phi = phi.T @ (cov2) @ phi
    

    K12 = 1/8 * ( phi_d_mu_t @ phi_sig12_phi @ phi_d_mu) + 1/2*np.log(np.linalg.det(phi_sig12_phi)/(np.linalg.det(phi_sig1_phi)*np.linalg.det(phi_sig2_phi)))

    return SQRTP1P2*np.exp(-K12)