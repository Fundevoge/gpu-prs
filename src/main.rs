use core::slice;
use std::{fs::File, io::Read, ops::Index, path::Path, time::Instant};

use cust::DeviceCopy;
use npyz::NpyFile;
use serde::Deserialize;

mod glue;

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct Config {
    voxel_grid_obj_path: String,
    voxel_grid_fixed_path: String,
    o_S_x: f32,
    o_D_obj: [[f32; 4]; 3],
    o_Ds_x: Vec<[[f32; 4]; 3]>,
}

fn load_json_config<P: AsRef<Path>>(path: P) -> Result<Config, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: Config = serde_json::from_str(&contents)?;
    Ok(config)
}

fn read_voxel_data<P: AsRef<Path>>(
    path: P,
) -> Result<(Vec<u8>, U32_3), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let npy = NpyFile::new(&bytes[..])?;
    let dims: [u32; 3] = npy
        .shape()
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>()
        .try_into()
        .unwrap();
    let order = npy.order();
    assert_eq!(order, npyz::Order::C, "Voxel data have to be C style");

    let data: Result<Vec<u8>, _> = npy.data::<bool>()?.map(|b| b.map(|b| b as u8)).collect();
    Ok((data?, dims.into()))
}

fn read_index_list<P: AsRef<Path>>(path: P) -> Result<Vec<U32_3>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let npy = NpyFile::new(&bytes[..])?;
    let dims: [u32; 2] = npy
        .shape()
        .iter()
        .map(|v| *v as u32)
        .collect::<Vec<u32>>()
        .try_into()
        .unwrap();
    let order = npy.order();
    assert_eq!(order, npyz::Order::C, "Index List data has to be C style");
    let mut data: Vec<U32_3> = vec![U32_3::default(); dims[0] as usize];
    let mut data_iter = npy.data::<i64>()?.map(|b| b.map(|b| b as u32)).peekable();

    for datum in &mut data {
        datum.x = data_iter.next().unwrap()? as u32;
        datum.y = data_iter.next().unwrap()? as u32;
        datum.z = data_iter.next().unwrap()? as u32;
    }

    Ok(data)
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug, DeviceCopy)]
struct U32_3 {
    x: u32,
    y: u32,
    z: u32,
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
    fn as_array(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VoxelGrid {
    data: *const u8,
    dims: U32_3,
}

impl Index<(u32, u32, u32)> for VoxelGrid {
    type Output = u8;

    fn index(&self, index: (u32, u32, u32)) -> &Self::Output {
        &self.as_slice()[((index.0 * self.dims.y + index.1) * self.dims.z + index.2) as usize]
    }
}

impl VoxelGrid {
    fn as_slice(&self) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                self.data,
                (self.dims.x * self.dims.y * self.dims.z) as usize,
            )
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
struct Transform {
    matrix: Matrix34,
}

impl Transform {
    // fn and_then(&self, other: &Transform) -> Transform {
    //     Transform {
    //         matrix: self.matrix.multiply_from_left(&other.matrix),
    //     }
    // }

    fn apply(self, v: &Vec3) -> Vec3 {
        self.matrix.multiply_vec_from_right(v)
    }

    fn from_homogeneous_and_scale(mut matrix: Matrix34, scale: f32) -> Self {
        for i in 0..3 {
            for j in 0..3 {
                matrix.0[i][j] *= scale;
            }
        }
        Self { matrix }
    }

    fn homogeneous_translation(&self) -> Vec3 {
        Vec3([
            self.matrix.0[0][3],
            self.matrix.0[1][3],
            self.matrix.0[2][3],
        ])
    }

    fn homogeneous_from_r_t(rotation: Matrix3, translation: Vec3) -> Self {
        let mut matrix = Matrix34::default();
        for i in 0..3 {
            for j in 0..3 {
                matrix.0[i][j] = rotation.0[i][j];
            }
            matrix.0[i][3] = translation.0[i];
        }
        Self { matrix }
    }

    #[allow(non_snake_case)]
    fn inverse(&self) -> Self {
        let mut new_rotation_matrix = Matrix3::default();
        let o_S_x_2: f32 =
            scalar_product(self.matrix.0[0][0..3].iter(), self.matrix.0[0][0..3].iter());

        for i in 0..3 {
            for j in 0..3 {
                new_rotation_matrix.0[j][i] = self.matrix.0[i][j] / o_S_x_2;
            }
        }

        let new_translation =
            new_rotation_matrix.multiply_vec_from_right(&self.homogeneous_translation().inverse());

        Transform::homogeneous_from_r_t(new_rotation_matrix, new_translation)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy)]
struct TransformQuat {
    scale: f32,
    t: Vec3,
    q: Vec4,
}

impl TransformQuat {
    fn from_transform(transform: &Transform) -> Self {
        let scale: f32 = scalar_product(
            transform.matrix.0[0][0..3].iter(),
            transform.matrix.0[0][0..3].iter(),
        )
        .sqrt();
        let cos_theta =
            ((transform.matrix.0[0][0] + transform.matrix.0[1][1] + transform.matrix.0[2][2])
                / scale
                - 1.)
                / 2.;
        let theta = cos_theta.clamp(-1., 1.).acos();
        let axis = Vec3([
            transform.matrix.0[2][1] - transform.matrix.0[1][2],
            transform.matrix.0[0][2] - transform.matrix.0[2][0],
            transform.matrix.0[1][0] - transform.matrix.0[0][1],
        ]);
        let axis_norm = axis.norm();
        let axis = if axis_norm > 5e-5 {
            axis.scale((theta / 2.).sin() / axis_norm)
        } else {
            Vec3::default()
        };

        let q = Vec4([(theta / 2.).cos(), axis.0[0], axis.0[1], axis.0[2]]);

        Self {
            t: transform.homogeneous_translation(),
            q,
            scale,
        }
    }

    // Assumes equal scale; will use self.scale
    fn interpolate<const N: usize>(&self, other: &TransformQuat) -> [TransformQuat; N] {
        let mut result = [TransformQuat::default(); N];
        result[N - 1] = *other;
        if N == 1 {
            return result;
        }
        let step_size = 1. / (N as f32);
        let dt = other.t.subtract(self.t).scale(step_size);

        let q0 = self.q;
        let qn = other.q;

        let qn_1 = slerp(q0, qn, (N as f32 - 1.) * step_size);

        let mut t_curr = other.t.subtract(dt);

        result[N - 2] = TransformQuat {
            scale: self.scale,
            t: t_curr,
            q: qn_1,
        };

        if N == 2 {
            return result;
        }

        let mut q_prev = qn;
        let mut q_curr = qn_1;

        let c = 2.0 * qn_1.dot(qn);
        for i in (0..(N - 2)).rev() {
            let q_next = q_curr.scale(c).subtract(q_prev).normalize();
            t_curr = t_curr.subtract(dt);

            q_prev = q_curr;
            q_curr = q_next;

            result[i] = TransformQuat {
                scale: self.scale,
                t: t_curr,
                q: q_curr,
            };
        }
        result
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy)]
struct Matrix34([[f32; 4]; 3]);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Matrix3([[f32; 3]; 3]);

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy)]
struct Vec3([f32; 3]);

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy)]
struct Vec4([f32; 4]);

impl Vec4 {
    fn inverse(self) -> Self {
        self.scale(-1.)
    }

    fn subtract(mut self, other: Vec4) -> Vec4 {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self.0[3] -= other.0[3];
        self
    }

    fn add(mut self, other: Vec4) -> Vec4 {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self.0[3] += other.0[3];
        self
    }

    fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn dot(self, other: Self) -> f32 {
        scalar_product(self.0.iter(), other.0.iter())
    }

    fn normalize(self) -> Self {
        let norm = self.norm();
        self.scale(1. / norm)
    }

    fn scale(mut self, scale: f32) -> Self {
        self.0[0] *= scale;
        self.0[1] *= scale;
        self.0[2] *= scale;
        self.0[3] *= scale;
        self
    }
}

fn scalar_product<'a>(
    a_s: impl Iterator<Item = &'a f32>,
    b_s: impl Iterator<Item = &'a f32>,
) -> f32 {
    a_s.zip(b_s).map(|(a, b)| *a * *b).sum()
}

fn slerp(q0: Vec4, q1: Vec4, t: f32) -> Vec4 {
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

impl Matrix3 {
    fn multiply_vec_from_right(&self, v: &Vec3) -> Vec3 {
        let mut new = Vec3::default();
        for i in 0..3 {
            new.0[i] = scalar_product(self.0[i].iter(), v.0.iter());
        }
        new
    }
}

impl Matrix34 {
    fn multiply_vec_from_right(&self, v: &Vec3) -> Vec3 {
        let mut new = Vec3::default();
        for i in 0..3 {
            new.0[i] = scalar_product(self.0[i].iter(), v.0.iter()) + self.0[i][3];
        }
        new
    }
}

impl Vec3 {
    fn inverse(self) -> Self {
        self.scale(-1.)
    }

    fn subtract(mut self, other: Vec3) -> Vec3 {
        self.0[0] -= other.0[0];
        self.0[1] -= other.0[1];
        self.0[2] -= other.0[2];
        self
    }

    fn add(mut self, other: Vec3) -> Vec3 {
        self.0[0] += other.0[0];
        self.0[1] += other.0[1];
        self.0[2] += other.0[2];
        self
    }

    fn norm(self) -> f32 {
        scalar_product(self.0.iter(), self.0.iter()).sqrt()
    }

    fn normalize(self) -> Self {
        let norm = self.norm();
        self.scale(1. / norm)
    }

    fn scale(mut self, scale: f32) -> Self {
        self.0[0] *= scale;
        self.0[1] *= scale;
        self.0[2] *= scale;
        self
    }
}

#[repr(C)]
struct Task {
    transform_from_obj: Transform,
    transform_into_1: Transform,
    transform_into_2: Transform,
    index_list_obj: *const U32_3,
    num_indices: u32,
    grid_static: VoxelGrid,
    out_collision: *mut u8,
}

#[allow(non_snake_case)]
fn run_task(task: &mut Task) {
    let o_D_obj = task.transform_from_obj;
    for idx in 0..task.num_indices as usize {
        let U32_3 { x, y, z } = unsafe { *task.index_list_obj.wrapping_add(idx) };
        let idx_coords_o = o_D_obj.apply(&Vec3([x as f32, y as f32, z as f32]));

        // Intersection with ground plane
        if idx_coords_o.0[2] < 0. {
            unsafe { *task.out_collision = 254 };
            return;
        }

        let location_in_x = task.transform_into_1.apply(&idx_coords_o);
        let indices_in_x = [
            location_in_x.0[0].round() as i32,
            location_in_x.0[1].round() as i32,
            location_in_x.0[2].round() as i32,
        ];
        if !indices_in_x
            .iter()
            .zip(task.grid_static.dims.as_array().iter())
            .all(|(i_x, d_x)| *i_x >= 0 && *i_x < (*d_x as i32))
        {
            continue;
        }
        if task.grid_static[(
            indices_in_x[0] as u32,
            indices_in_x[1] as u32,
            indices_in_x[2] as u32,
        )] == 1
        {
            unsafe { *task.out_collision = 1 };
            return;
        }

        let location_in_x = task.transform_into_2.apply(&idx_coords_o);
        let indices_in_x = [
            location_in_x.0[0].round() as i32,
            location_in_x.0[1].round() as i32,
            location_in_x.0[2].round() as i32,
        ];
        if !indices_in_x
            .iter()
            .zip(task.grid_static.dims.as_array().iter())
            .all(|(i_x, d_x)| *i_x >= 0 && *i_x < (*d_x as i32))
        {
            continue;
        }
        if task.grid_static[(
            indices_in_x[0] as u32,
            indices_in_x[1] as u32,
            indices_in_x[2] as u32,
        )] == 1
        {
            unsafe { *task.out_collision = 2 };
            return;
        }
    }

    unsafe { *task.out_collision = 0 };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = load_json_config("./scenes/scene_1.json")?;

    let scale = config.o_S_x;
    println!("o_S_x: {scale}");
    println!("o_D_obj: {:?}", config.o_D_obj);
    println!("Static objects: {}", config.o_Ds_x.len());
    let static_transforms: Vec<_> = config
        .o_Ds_x
        .into_iter()
        .map(|matrix| Transform::from_homogeneous_and_scale(Matrix34(matrix), scale).inverse())
        .collect();
    let transform_from_obj = Transform::from_homogeneous_and_scale(Matrix34(config.o_D_obj), scale);

    let (voxel_data, dims) = read_voxel_data(&config.voxel_grid_fixed_path)?;
    println!("Voxel data dims: {dims:?}");
    println!("Voxel data length: {}", voxel_data.len());
    let data = voxel_data.leak().as_ptr();
    let grid = VoxelGrid { data, dims };

    let index_list = read_index_list(&config.voxel_grid_obj_path)?;
    println!("Index list data length: {}", index_list.len());
    println!("First index list tuple: {:?}", index_list[0]);

    let mut collision: u8 = 255;
    let mut task = Task {
        out_collision: &mut collision,
        grid_static: grid,
        index_list_obj: index_list.as_ptr(),
        num_indices: index_list.len() as u32,
        transform_into_1: static_transforms[0],
        transform_into_2: static_transforms[1],
        transform_from_obj,
    };
    let start = Instant::now();
    run_task(&mut task);
    let duration = start.elapsed();
    let collision = unsafe { *task.out_collision };
    println!("Collision {collision} Time taken: {duration:?}");

    let mut gpu_out = [255];
    glue::gpu_driver(
        glue::TaskStaticInfo {
            transform_into_1: task.transform_into_1,
            transform_into_2: task.transform_into_2,
            index_list_obj: index_list,
            num_indices: task.num_indices,
            voxel_grid: task.grid_static,
        },
        &[TransformQuat::from_transform(&transform_from_obj)],
        &mut gpu_out,
    );
    println!("GPU returned collision {}", gpu_out[0]);

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    fn quat_to_rot_matrix(q: &Vec4) -> Matrix3 {
        Matrix3([
            [
                1. - 2. * (q.0[2] * q.0[2] + q.0[3] * q.0[3]),
                2. * (q.0[1] * q.0[2] - q.0[3] * q.0[0]),
                2. * (q.0[1] * q.0[3] + q.0[2] * q.0[0]),
            ],
            [
                2. * (q.0[1] * q.0[2] + q.0[3] * q.0[0]),
                1. - 2. * (q.0[1] * q.0[1] + q.0[3] * q.0[3]),
                2. * (q.0[2] * q.0[3] - q.0[1] * q.0[0]),
            ],
            [
                2. * (q.0[1] * q.0[3] - q.0[2] * q.0[0]),
                2. * (q.0[2] * q.0[3] + q.0[1] * q.0[0]),
                1. - 2. * (q.0[1] * q.0[1] + q.0[2] * q.0[2]),
            ],
        ])
    }

    #[test]
    fn test_quaternion_conersion() {
        let m = Matrix34([
            [0.99884413, -0.00170316, 0.04803638, 0.],
            [0.00386124, -0.9932992, -0.11550674, 0.],
            [0.04791122, 0.11555871, -0.9921445, 0.],
        ]);
        let d = Transform::from_homogeneous_and_scale(m, 1.);
        let m_recovered = quat_to_rot_matrix(&TransformQuat::from_transform(&d).q);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_recovered.0[i][j] - m.0[i][j]).abs() < 5e-5,
                    "Matrix Quaternion conversion failed at index {i} {j}\n{:?}\n{:?}",
                    m.0,
                    m_recovered.0
                );
            }
        }

        let m = Matrix34([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]);
        let d = Transform::from_homogeneous_and_scale(m, 1.);
        let m_recovered = quat_to_rot_matrix(&TransformQuat::from_transform(&d).q);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m_recovered.0[i][j] - m.0[i][j]).abs() < 5e-5,
                    "Matrix Quaternion conversion failed at index {i} {j}\n{:?}\n{:?}",
                    m.0,
                    m_recovered.0
                );
            }
        }
    }

    #[test]
    fn test_quaternion_interpolation() {
        let scale = 0.1;
        let r3_2 = 3.0_f32.sqrt() / 2.;
        let m0 = Matrix34([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]);
        let m1 = Matrix34([[r3_2, -0.5, 0., 1.], [0.5, r3_2, 0., 2.], [0., 0., 1., 3.]]);
        let m2 = Matrix34([[0.5, -r3_2, 0., 2.], [r3_2, 0.5, 0., 4.], [0., 0., 1., 6.]]);
        let m3 = Matrix34([[0., -1., 0., 3.], [1., 0., 0., 6.], [0., 0., 1., 9.]]);
        let tq0 = TransformQuat::from_transform(&Transform::from_homogeneous_and_scale(m0, scale));
        let tq1 = TransformQuat::from_transform(&Transform::from_homogeneous_and_scale(m1, scale));
        let tq2 = TransformQuat::from_transform(&Transform::from_homogeneous_and_scale(m2, scale));
        let tq3 = TransformQuat::from_transform(&Transform::from_homogeneous_and_scale(m3, scale));

        let interps: [TransformQuat; 3] = tq0.interpolate(&tq3);

        for (i, (interp, sol)) in interps.iter().zip([tq1, tq2, tq3]).enumerate() {
            for (j, (q_interp, q_sol)) in interp.q.0.iter().zip(sol.q.0.iter()).enumerate() {
                assert!(
                    (q_interp - q_sol).abs() < 5e-5,
                    "Interpolation failed for step {}, index {j}, quat interp {:?} vs sol {:?}",
                    i + 1,
                    interp.q.0,
                    sol.q.0
                );
            }
            for (j, (t_interp, t_sol)) in interp.t.0.iter().zip(sol.t.0.iter()).enumerate() {
                assert!(
                    (t_interp - t_sol).abs() < 5e-5,
                    "Interpolation failed for step {}, index {j}, vec interp {:?} vs sol {:?}",
                    i + 1,
                    interp.t.0,
                    sol.t.0
                );
            }
            assert_eq!(
                interp.scale, sol.scale,
                "Scale wrong for Transform {i}: interp {:?} vs sol {:?}",
                interp.scale, sol.scale
            );
        }
    }
}
