use cust::DeviceCopy;

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, DeviceCopy, Hash, PartialEq, Eq)]
pub struct U32_3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl From<[u32; 3]> for U32_3 {
    fn from(val: [u32; 3]) -> Self {
        U32_3 {
            x: val[0],
            y: val[1],
            z: val[2],
        }
    }
}

impl U32_3 {
    pub fn as_array(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy)]
pub struct Matrix34(pub [[f32; 4]; 3]);

impl Matrix34 {
    pub fn multiply_vec_from_right(&self, v: &Vec3) -> Vec3 {
        let mut new = Vec3::default();
        for i in 0..3 {
            new.0[i] = scalar_product(self.0[i].iter(), v.0.iter()) + self.0[i][3];
        }
        new
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Matrix3(pub [[f32; 3]; 3]);

impl Matrix3 {
    pub fn multiply_vec_from_right(&self, v: &Vec3) -> Vec3 {
        let mut new = Vec3::default();
        for i in 0..3 {
            new.0[i] = scalar_product(self.0[i].iter(), v.0.iter());
        }
        new
    }

    pub fn multiply_mat_from_right(&self, m: &Matrix3) -> Matrix3 {
        let mut new = Matrix3::default();
        for i in 0..3 {
            for j in 0..3 {
                new.0[i][j] = scalar_product(self.0[i].iter(), m.0.iter().map(|row| &row[j]));
            }
        }
        new
    }

    pub fn from_34(matrix34: &Matrix34) -> Self {
        let mut res = Self::default();
        for i in 0..3 {
            for j in 0..3 {
                res.0[i][j] = matrix34.0[i][j];
            }
        }
        res
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy, Debug)]
pub struct Vec3(pub [f32; 3]);

impl Vec3 {
    pub fn inverse(self) -> Self {
        self.scale(-1.)
    }

    pub fn subtract(mut self, other: Vec3) -> Vec3 {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self
    }

    pub fn add(mut self, other: Vec3) -> Vec3 {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self
    }

    pub fn norm(self) -> f32 {
        scalar_product(self.0.iter(), self.0.iter()).sqrt()
    }

    pub fn normalize(self) -> Self {
        let norm = self.norm();
        self.scale(1. / norm)
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.0[0] *= scale;
        self.0[1] *= scale;
        self.0[2] *= scale;
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy, Debug)]
pub struct Vec4(pub [f32; 4]);

impl Vec4 {
    pub fn inverse(self) -> Self {
        self.scale(-1.)
    }

    pub fn subtract(mut self, other: Vec4) -> Vec4 {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
        self
    }

    pub fn add(mut self, other: Vec4) -> Vec4 {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
        self
    }

    pub fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn dot(self, other: Self) -> f32 {
        scalar_product(self.0.iter(), other.0.iter())
    }

    pub fn normalize(self) -> Self {
        let norm = self.norm();
        self.scale(1. / norm)
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.0[0] *= scale;
        self.0[1] *= scale;
        self.0[2] *= scale;
        self.0[3] *= scale;
        self
    }

    pub fn quat_from_unit_axis_angle(axis: Vec3, angle: f64) -> Self {
        let quat_angle = angle / 2.;
        let w = quat_angle.cos() as f32;
        let [x, y, z] = axis.scale(quat_angle.sin() as f32).0;
        Vec4([w, x, y, z])
    }

    pub fn quat_multiply(self, other: Self) -> Self {
        let [w1, x1, y1, z1] = self.0;
        let [w2, x2, y2, z2] = other.0;
        let w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
        let x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
        let y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
        let z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

        Vec4([w, x, y, z])
    }
}

pub fn scalar_product<'a>(
    a_s: impl Iterator<Item = &'a f32>,
    b_s: impl Iterator<Item = &'a f32>,
) -> f32 {
    a_s.zip(b_s).map(|(a, b)| *a * *b).sum()
}

pub fn slerp(q0: Vec4, q1: Vec4, t: f32) -> Vec4 {
    let mut dot = q0.dot(q1);

    let mut q1_mod = q1;

    // Ensure shortest path
    if dot < 0.0 {
        q1_mod = q1_mod.scale(-1.0);
        dot = -dot;
    }

    const DOT_THRESHOLD: f32 = 0.9995;
    if dot > DOT_THRESHOLD {
        // Use LERP and normalize
        let lerped = q0.scale(1.0 - t).add(q1_mod.scale(t));
        return lerped.normalize();
    }

    let theta_0 = dot.acos(); // initial angle
    let sin_theta_0 = theta_0.sin();

    let theta = theta_0 * t;
    let sin_theta = theta.sin();

    let s0 = (theta_0 - theta).sin() / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;

    q0.scale(s0).add(q1_mod.scale(s1))
}
