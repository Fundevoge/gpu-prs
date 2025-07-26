# How to use scene.json
Transforms as o_R_x, o_T_ox \
Scale as o_S_x

[Default] Grid to world :  [o_S_x * o_R_x, o_T_ox; 0, 0, 0, 1] @ [x_P, 1]\
world to grid : [o_S_x^-1 * o_R_x.T, -(o_S_x^-1 * o_R_x.T) @ o_T_ox] @ [o_P, 1]

Warning: D is not built for repeated inversion, errors may accumulate \
o_S_x = L2-norm of first three entries of D => Divide by L2-norm squared

## Angle tradeoff factor
The factor used in $d_x + w * d_\phi$
For 10° to be equivalent to 5mm use: $w=\frac{0.005}{\pi \cdot 10°/180°}$\

# The rust library
Transforms (Transform and TransformQuat) represent rotation, scaling, and then translating.
Interpolating between TransformQuat assumes they have the same scale