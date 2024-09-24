
# change euclidean points (xy) to ellipsoidal parameters (ep)
# [x**2, xy, y**2, x, y, 1 ]
def coords_euclidean_to_ep(X):
    new_coords = torch.stack([X[:,0]**2,X[:,0]*X[:,1],X[:,1]**2,X[:,0],X[:,1],torch.ones_like(X[:,0])],axis=1)
    return new_coords

# change parameters from ep to a,b,theta,x,y
# https://www.geometrictools.com/Documentation/InformationAboutEllipses.pdf
#
def params_ep_to_ab(P):
    A,B,C,D,E,F = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4],P[:,5]
    
    B_half = B/2.
    
    k1 = (C*D - B_half*E) / (2*(B_half*B_half - A*C))
    k2 = (A*E - B_half*D) / (2*(B_half*B_half - A*C))
    mu = 1./(A*k1*k1 + 2*B_half*k1*k2 + C*k2*k2 - F)
    
    m11 = mu*A
    m12 = mu*B_half
    m22 = mu*C
    
    lambda1 = (0.5)*(m11 + m22+torch.sqrt((m11-m22)**2 + 4*(m12**2)))
    lambda2 = (0.5)*(m11 + m22-torch.sqrt((m11-m22)**2 + 4*(m12**2)))

    
    a = 1./torch.sqrt(lambda1)
    b = 1./torch.sqrt(lambda2)
    if a < b:
        a,b = b,a
    
    theta = 0.5 * torch.atan2(-2*B_half,C-A)
    
    return torch.stack((a,b,theta,k1,k2),axis=-1)


# change parameters from a,b,theta,x,y to ep
# https://en.wikipedia.org/wiki/Ellipse#General_ellipse
#
def params_ab_to_ep(P):
    a,b,theta,x,y = P[:,0],P[:,1],P[:,2],P[:,3],P[:,4]
    
    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)
    
    A = a*a*sintheta*sintheta + b*b*costheta*costheta
    B = 2*(b*b - a*a)*sintheta*costheta
    C = a*a*costheta*costheta + b*b*sintheta*sintheta
    
    D = -2*A*x - B*y
    E = -B*x - 2*C*y
    F = A*x*x + B*x*y + C*y*y - a*a*b*b
    
    return torch.stack((A,B,C,D,E,F),axis=-1)

# SVD Fitzgibbon 1996
def fitzgibbon(M_mat,C_mat):
    U, s, V = torch.svd(torch.matmul(torch.linalg.inv(M_mat).to(torch.float32), C_mat.to(torch.float32)))
    a = U[:, 0:1]
    return torch.transpose(a,-2,-1)

# halir flusser 1998
def halir_flusser(M_mat,C1_mat):
    S1 = M_mat[0:3,0:3]
    S2 = M_mat[0:3,3:]
    S2_t = M_mat[3:,0:3]
    S3 = M_mat[3:,3:]
    
    T = -torch.linalg.inv(S3) @ torch.transpose(S2,-2,-1)
    M1 = torch.linalg.inv(C1_mat.to(torch.float32)) @ (S1 + S2@T)
    
    U,S,V = torch.svd(M1)
    
    cond = 4* U[0,:] * U[2,:] - U[1,:]**2
    a1 = U[:,2:3]
    
    a2 = T @ a1
    
    return torch.transpose(torch.concat((a1,a2),axis=-2),-2,-1)


def coords_to_scatter_mat(X):
    X_ep = coords_euclidean_to_ep(X)

    X_ep_T = torch.transpose(X_ep,-2,-1)

    M_matrix = torch.matmul(X_ep_T,X_ep)
    return M_matrix

def constraint_mats():
    constraint_mat_1 = torch.tensor([[1,0],[0,0]])
    constraint_mat_2 = torch.tensor([[0,0,2],[0,-1,0],[2,0,0]])
    C = torch.kron(constraint_mat_1,constraint_mat_2)
    return constraint_mat_1, constraint_mat_2, C


