# Shape From Shading

### Shape can be expressed as
- depth
    - distance from camera to surface points
    - relative surface height above the x-y plane   
- surface normal
- surface gradient
- surface slant and tilt



## Reflectance models
### Lambertian surface
- surface having only diffuse reflectance
- brightness of a Lambertian surface is proportional to the energy of the incident light
- the amount of light energy falling on a surface element is proportional to the area of the surface element as seen from the light source position (the foreshortened area)
- The foreshortened area is a cosine function of the angle between the surface orientation and the light source direction.
- the Lambertian surface can be modeled as the product of the strength of the light source A, the albedo of the surface rho, and the foreshortened area cos(theta_i) as follows:
> I_L = R = A rho cos(theta_i)
- where R is the reflectance map, theta_i is the angle between the surface normal N = (n_x, n_y, n_z) and the source direction S=(s_x, s_y, s_z)
- Let the surface normal and light source direction both be unit vectors, the above formula can be written as
> I_L = A rho N dot S
- Pros: simplicity
- Cons: poor approximation to the diffuse component of rough surfaces

### Specular surface
- incident angle of the light source is equal to the reflected angle
> I_S = B delta (theta_s - 2theta_r)
- where I_S is the specular brightness, B is the strength of the specular component, theta_s is the angle between the light source direction and the viewing direction, and thet_r is the angle between the surface normal and the viewing direction
- the model assumes that the highlight caused by specular reflection is only a single point

### Hybrid Surface
> I = (1-w) I_L + w I_S
- where I is the total brightness for the hybrid surface, I_S is specular brightness, I_L is Lambertian brightness, w is the weight of the specular component
> I = K_dl cos(theta_i) + K_sl e^(-beta^2/(2sigma^2)) + K_ss delta(theta_i - theta_r) theta(phi_r)
- where K_dl, K_sl and K_ss are the strengths of the three components, 
  beta is the angle between surface normal of a micro-facet on a patch and 
  the mean normal of this surface patch, 
  sigma is its standard derivation. 
  If we consider the surface normal being in the Z direction, then 
  (theta_i, phi_i) is the direction of incidence light in terms of the slant and tilt in 3D, 
  (theta_r, phi_r) is the direction of reflected light 


## Shape from shading algorithms

##### assumptions: Light source direction is known

### Minimization approaches
- minimizses an energy function over the entire image
- involve the brightness constraint and other constraints, such as smoothness constraint, the integrability constraint, the gradient constraint, and the unit normal constraint 

#### Brightness constraint
- &Int;(I-R)&sup2;dxdy
- it indicates the total brightness error of the reconstructed image compared with the input image
- I is the measured intensity, and R is the estimated reflectance map

##### Smoothness Constraint
> &Int;(p_x^2 + p_y^2 + q_x^2 + q_y^2)dxdy
- where p and q are surface gradients along the x and y directions
> &Int;(||N_x||^2 + ||N_y||^2)dxdy
- the smoothness constraint is described in terms of the surface normal

##### Integrability constraint
- it ensures valid surfaces, that is Z_x,y = Z_y,x
> &Int;(p_y - q_x)^2 dxdy
or
> &Int;((Z_x - p)^2 + (Z_y - q)^2)dxdy

##### The intensity gradient constraint
- Requires that the **intensity gradient** of the reconstructed image be close to 
  the intensity gradient of the input image in both the x and y directions
> &Int;((R_x - I_x)^2 + (R_y - I_y)^2)dxdy

##### The unit normal constraint
- forces the recovered surface normals to be unit vectors
> &Int;(||N||^2-1)dxdy

#### Zheng and Chellappa
- Apply the intensity gradient constraint
- The energy function is
> &Int;((I-R)^2 + ((R_x - I_x)^2 + (R_y - I_y)^2) + &mu;((Z_x-p)^2+(Z_y - q)^2))dxdy
- The minimization of the function was done through variational calculus

### Others