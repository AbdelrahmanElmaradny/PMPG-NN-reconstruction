import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt


# Creates spherical grid 2d
def generate_grid(num_points_theta,num_points_r,internal_radius, external_radius, theta_max):
    theta_values = torch.linspace(0.0001, theta_max, num_points_theta)
    r_values = torch.logspace(torch.log10(torch.tensor(internal_radius)), torch.log10(torch.tensor(external_radius)), num_points_r)
    # phi_values = torch.linspace(0, 2*torch.pi, num_points_theta)

# Create mesh grid
    # R, Theta,Phi = torch.meshgrid(r_values, theta_values,phi_values)
# Create mesh grid
    R, Theta = torch.meshgrid(r_values, theta_values)

# Convert to Cartesian coordinates with assumption phi equal zero
    X = R * torch.sin(Theta) #Notice different than cylindrical
    Z = R * torch.cos(Theta)
    
    R=torch.tensor(R, dtype=torch.float32, requires_grad=True)
    Theta=torch.tensor(Theta, dtype=torch.float32, requires_grad=True)
    # Phi=torch.tensor(Phi, dtype=torch.float32, requires_grad=True)
    R=R.flatten()#.unsqueeze(1)
    Theta=Theta.flatten()#.unsqueeze(1)
    # Phi=Phi.flatten()#.unsqueeze(1)
    X=X.flatten()#.unsqueeze(1)
    Z=Z.flatten()#.unsqueeze(1)

    return R, Theta, r_values,theta_values, X, Z

# Flow over a sphere in z direction
def sphere_velocity_field(R,Theta):
    psi = torch.zeros_like(R)
    psi = -0.5 * torch.sin(Theta)**2 * R**2 * (1/R**3 -1)
    
    # Compute velocity components
    dpsi_dtheta = torch.autograd.grad(psi.sum(), Theta, create_graph=True, retain_graph=True)[0]
    
    dpsi_dr = torch.autograd.grad(psi.sum(), R, create_graph=True, retain_graph=True)[0]
    
    # Compute velocity components in spherical coordinates
    V_theta = -dpsi_dr /(R * (torch.sin(Theta)))
    V_r =  dpsi_dtheta /(R**2 * (torch.sin(Theta)))
    
    u_x = V_r * torch.sin(Theta) + V_theta * torch.cos(Theta)
    u_z = V_r * torch.cos(Theta) - V_theta * torch.sin(Theta)
    
    V= torch.sqrt(u_x**2 + u_z**2)

    return u_x,u_z,V_theta,V_r,V,psi
    

def grid_spherical_to_cartesian(rho,theta,phi):
    X= rho * torch.sin(theta)*torch.cos(phi)
    Y= rho * torch.sin(theta)*torch.sin(phi)
    Z= rho * torch.cos(theta)

    return X,Y,Z


def grid_cartesian_to_spherical(X,Y,Z):
    rho= torch.sqrt(X**2+Y**2+Z**2)
    theta= torch.atan(torch.sqrt(X**2+Y**2)/Z)
    phi= torch.atan(Y/X)

    return rho,theta,phi

#Calculate acceleration
def spherical_acceleration_field(R,Theta,V_r,V_theta):
    dV_r_dr = torch.autograd.grad(V_r.sum(), R, create_graph=True, retain_graph=True)[0]
    dV_r_dtheta = torch.autograd.grad(V_r.sum(), Theta, create_graph=True, retain_graph=True)[0]
    
    dV_theta_dr = torch.autograd.grad(V_theta.sum(), R, create_graph=True, retain_graph=True)[0]
    dV_theta_dtheta = torch.autograd.grad(V_theta.sum(), Theta, create_graph=True, retain_graph=True)[0]
    
    
    convective_r = V_r *dV_r_dr + V_theta *dV_r_dtheta / R - V_theta**2 / R
    convective_theta = V_r *dV_theta_dr + V_theta *dV_theta_dtheta / R + V_theta* V_r / R
    
    a_x = convective_r * torch.sin(Theta) + convective_theta * torch.cos(Theta)
    a_z = convective_r * torch.cos(Theta) - convective_theta * torch.sin(Theta)

    A= torch.sqrt(a_x**2 + a_z**2)

    return a_x,a_z,convective_r,convective_theta,A

#Calculate vorticity
def cylinder_vorticity_field(R,Theta,V_r,V_theta):
    drV_theta_dr = torch.autograd.grad((R*V_theta).sum(), R, create_graph=True, retain_graph=True)[0]
    dV_r_dtheta = torch.autograd.grad(V_r.sum(), Theta, create_graph=True, retain_graph=True)[0]
    
    w=(drV_theta_dr - dV_r_dtheta)/R
    return w



#Integrate over spherical coordinates
def integrate_r_theta_spherical_axis_symmetric(Integrand,r_values,theta_values,num_points_theta,num_points_r):
    I2=torch.reshape(Integrand,(num_points_theta,num_points_r))
    S1 = torch.trapz(I2*torch.sin(theta_values), theta_values, dim=1)
    S = torch.trapz(S1*r_values**2, r_values)
    result = S.item()
    return result


#Conformal mappings


# Define the conformal map zeta (cylinder) -> z (airfoil)
def zeta_to_z(zeta,tau,D):
    ecce = tau / ((3 * np.sqrt(3)) / 4)
    C = 1 / (1 + ecce)
    mu = ecce * C
    Delta = (1 - D) / (1 + D)
    return (zeta - mu) + (1 - D) / (1 + D) * C**2 / (zeta - mu)

# Define the derivative of the conformal map dz/dzeta
def dz_dzeta(zeta,tau,D):
    ecce = tau / ((3 * np.sqrt(3)) / 4)
    C = 1 / (1 + ecce)
    mu = ecce * C
    Delta = (1 - D) / (1 + D)
    return 1 - ((1 - D) / (1 + D) * C**2) / (zeta - mu)**2

# Define the scaling factor G
def scaling_factor_G(zeta,tau,D):
    ecce = tau / ((3 * np.sqrt(3)) / 4)
    C = 1 / (1 + ecce)
    mu = ecce * C
    Delta = (1 - D) / (1 + D)
    return (zeta - mu)**2 / ((zeta - mu)**2 - Delta * C**2)

# Define the complex velocity in the airfoil domain
def ubar(zeta,Ubar,tau,D):
    return Ubar * scaling_factor_G(zeta,tau,D)

# Define the complex acceleration in the airfoil domain
def complex_acceleration_a(zeta, A, Ubar_zeta,tau,D):
    ecce = tau / ((3 * np.sqrt(3)) / 4)
    C = 1 / (1 + ecce)
    mu = ecce * C
    Delta = (1 - D) / (1 + D)
    G = scaling_factor_G(zeta,tau,D)
    dG_dzeta = (2 * (zeta - mu) * ((zeta - mu)**2 - Delta * C**2) - (zeta - mu)**2 * 2 * (zeta - mu)) / ((zeta - mu)**2 - Delta * C**2)**2
    return torch.abs(G) * (torch.conj(G) * A + torch.abs(Ubar_zeta)**2 * torch.conj(dG_dzeta))

#Integrate in airfoil domain
def integrate_airfoil_domain(f,Zeta,r_values,theta_values,tau,D,num_points_theta,num_points_r):
    G = scaling_factor_G(Zeta,tau,D)
    integrand1 = torch.abs(f)**2 / torch.abs(G)**2
    Integral = integrate_r_theta(integrand1,r_values,theta_values,num_points_theta,num_points_r)
    return Integral
    
#Create circle for boundary conditions
def generate_cylinder_circumference_points(num_points, cylinder_radius):
    theta = torch.linspace(0, 2 * np.pi, num_points)
    x = cylinder_radius * np.cos(theta)
    y = cylinder_radius * np.sin(theta)

    positions = np.column_stack([x, y])
    return positions
    
#Decomposition of components to normal and tangential
def normal_tangential_decomposition(x,y):
    x = torch.tensor(x)
    y = torch.tensor(y)

    # Calculate the tangential vector (dx/dy)
    dx = torch.gradient(x)[0]
    dy = torch.gradient(y)[0]
    tangential_vectors = torch.stack((dx, dy), dim=1)

    dx_diff = torch.diff(x)
    dy_diff = torch.diff(y)
    ds = torch.sqrt(dx_diff**2 + dy_diff**2)
    ds = torch.cat((ds, ds[-1].unsqueeze(0)))  # Make sure ds has the same length as x and y


    # Normalize the tangential vector
    tangential_magnitudes = torch.norm(tangential_vectors, dim=1)
    tangential_vectors = tangential_vectors / tangential_magnitudes.unsqueeze(1)
    
    # Calculate the normal vector (rotate tangential vector by 90 degrees)
    normal_vectors = torch.stack((-tangential_vectors[:, 1], tangential_vectors[:, 0]), dim=1)
    return tangential_vectors, normal_vectors,ds


