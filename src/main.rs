use core::{f64, slice};
use std::{
    array,
    fs::File,
    io::{Read, Write},
    ops::Index,
    path::Path,
    time::Instant,
};

use cust::DeviceCopy;
use npyz::NpyFile;
use probability::prelude::{Sample, Uniform};
use serde::Deserialize;

use crate::math::{Matrix3, Matrix34, U32_3, Vec3, Vec4};

mod glue;
mod math;

#[allow(non_snake_case)]
#[derive(Debug, Deserialize)]
struct Config {
    voxel_grid_obj_path: String,
    voxel_grid_fixed_path: String,
    o_S_x: f32,
    o_D_obj: Matrix34,
    o_Ds_x: Vec<[[f32; 4]; 3]>,
    rrt_info: RRTInfo,
}
#[derive(Debug, Deserialize)]
struct StepSize {
    angle: f32,
    distance: f32,
}
#[derive(Debug, Deserialize)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}
#[derive(Debug, Deserialize)]
struct RRTInfo {
    angle_distance_tradeoff: f32,
    pos_bounds: Bounds,
    step_size: StepSize,
    n_interp_steps: usize,
    target: Matrix34,
    target_angle_disturbance: f64,
    target_distance_disturbance: f64,
    combinded_distance_success_threshold: f32,
    n_target_attempts: usize,
}

fn load_json_config<P: AsRef<Path>>(path: P) -> Result<Config, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let config: Config = serde_json::from_str(&contents)?;
    Ok(config)
}

fn write_json_results<P: AsRef<Path>>(
    path: P,
    contents: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::create(path)?;
    file.write_all(contents.as_bytes())?;
    Ok(())
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
    fn apply(self, v: &Vec3) -> Vec3 {
        self.matrix.multiply_vec_from_right(v)
    }

    fn and_then(&self, other: &Transform) -> Transform {
        let r_1 = Matrix3::from_34(&self.matrix);
        let r_2 = Matrix3::from_34(&other.matrix);
        let t_1 = self.homogeneous_translation();
        let t_2 = other.homogeneous_translation();
        Self::from_r_t(
            r_2.multiply_mat_from_right(&r_1),
            r_2.multiply_vec_from_right(&t_1).add(t_2),
        )
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

    fn from_r_t(rotation: Matrix3, translation: Vec3) -> Self {
        let mut matrix = Matrix34::default();
        for i in 0..3 {
            for j in 0..3 {
                matrix.0[i][j] = rotation.0[i][j];
            }
            matrix.0[i][3] = translation.0[i];
        }
        Self { matrix }
    }

    fn from_r_t_scale(rotation: Matrix3, translation: Vec3, scale: f32) -> Self {
        let mut matrix = Matrix34::default();
        for i in 0..3 {
            for j in 0..3 {
                matrix.0[i][j] = rotation.0[i][j] * scale;
            }
            matrix.0[i][3] = translation.0[i];
        }
        Self { matrix }
    }

    #[allow(non_snake_case)]
    fn inverse(&self) -> Self {
        let mut new_rotation_matrix = Matrix3::default();
        let o_S_x_2: f32 =
            math::scalar_product(self.matrix.0[0][0..3].iter(), self.matrix.0[0][0..3].iter());

        for i in 0..3 {
            for j in 0..3 {
                new_rotation_matrix.0[j][i] = self.matrix.0[i][j] / o_S_x_2;
            }
        }

        let new_translation =
            new_rotation_matrix.multiply_vec_from_right(&self.homogeneous_translation().inverse());

        Transform::from_r_t(new_rotation_matrix, new_translation)
    }

    fn from_transform_quat_unit_scale(t: TransformQuat) -> Self {
        Self::from_r_t(Matrix3::from_quat(t.q), t.t)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, DeviceCopy, Debug)]
struct TransformQuat {
    scale: f32,
    t: Vec3,
    q: Vec4,
}

impl TransformQuat {
    fn from_transform(transform: &Transform) -> Self {
        let (rotation_matrix, scale) = Matrix3::from_34_normalized(&transform.matrix);

        Self {
            t: transform.homogeneous_translation(),
            q: Vec4::quat_from_rotation_matrix(rotation_matrix),
            scale,
        }
    }

    fn distances(&self, point: Vec3, quat: Vec4) -> (f32, f32) {
        (
            self.t.euclidean_distance(point),
            self.q.quat_relative_angle(quat),
        )
    }

    fn combined_distance(&self, other: &TransformQuat, angle_distance_tradeoff: f32) -> f32 {
        self.t.euclidean_distance(other.t)
            + angle_distance_tradeoff * self.q.quat_relative_angle(other.q)
    }

    // Assumes equal scale; will use self.scale
    fn interpolate(&self, other: &TransformQuat, out_buffer: &mut [TransformQuat]) {
        let n = out_buffer.len();
        out_buffer[n - 1] = *other;
        if n == 1 {
            return;
        }
        let step_size = 1. / (n as f32);
        let dt = other.t.subtract(self.t).scale(step_size);

        let q0 = self.q;
        let qn = other.q;

        let qn_1 = math::slerp(q0, qn, ((n - 1) as f32) * step_size);

        let mut t_curr = other.t.subtract(dt);

        out_buffer[n - 2] = TransformQuat {
            scale: self.scale,
            t: t_curr,
            q: qn_1,
        };

        if n == 2 {
            return;
        }

        let mut q_prev = qn;
        let mut q_curr = qn_1;

        let c = 2.0 * qn_1.dot(qn);
        for i in (0..(n - 2)).rev() {
            let q_next = q_curr.scale(c).subtract(q_prev).normalize();
            t_curr = t_curr.subtract(dt);

            q_prev = q_curr;
            q_curr = q_next;

            out_buffer[i] = TransformQuat {
                scale: self.scale,
                t: t_curr,
                q: q_curr,
            };
        }
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
        if indices_in_x
            .iter()
            .zip(task.grid_static.dims.as_array().iter())
            .all(|(i_x, d_x)| *i_x >= 0 && *i_x < (*d_x as i32))
            && task.grid_static[(
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
        if indices_in_x
            .iter()
            .zip(task.grid_static.dims.as_array().iter())
            .all(|(i_x, d_x)| *i_x >= 0 && *i_x < (*d_x as i32))
            && task.grid_static[(
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

fn random_sampling(
    base_transform_quat: &TransformQuat,
    out_buffer: &mut [TransformQuat],
    max_angle: f64,
    max_distance: f64,
    source: &mut impl probability::source::Source,
) {
    let base_quat = base_transform_quat.q;
    let base_vec = base_transform_quat.t;

    let std_gaussian = probability::distribution::Gaussian::default();
    let angle_distribution_uniform = probability::distribution::Uniform::new(-max_angle, max_angle);
    let uniform_distance = probability::distribution::Uniform::new(0., max_distance);
    for out in out_buffer {
        let random_quat = Vec4::random_quat(source, &angle_distribution_uniform);
        // Apply to base_transform (quat)
        let new_quat = random_quat.quat_multiply(base_quat);

        // Sample random vector
        let random_translation = Vec3(array::from_fn(|_| std_gaussian.sample(source) as f32))
            .normalize()
            .scale(uniform_distance.sample(source) as f32)
            .add(base_vec);

        *out = TransformQuat {
            scale: base_transform_quat.scale,
            t: random_translation,
            q: new_quat,
        };
    }
}

fn rrt(
    nodes: &[TransformQuat],
    info: &RRTInfo,
    out_buffer: &mut [TransformQuat],
    out_index_buffer: &mut [usize],
    sample_buffer: &mut [TransformQuat],
    source: &mut impl probability::source::Source,
) {
    assert!(
        out_buffer.len() % info.n_interp_steps == 0,
        "Output buffer size {} should be divisible by number of interpolation steps {}",
        out_buffer.len(),
        info.n_interp_steps,
    );
    assert!(
        out_buffer.len() / info.n_interp_steps == out_index_buffer.len(),
        "Output buffer size {} and Output index buffer size {} should be related by number of interpolation steps {}",
        out_buffer.len(),
        out_index_buffer.len(),
        info.n_interp_steps,
    );
    assert!(
        out_buffer.len() / info.n_interp_steps == sample_buffer.len(),
        "Output buffer size {} and Sample buffer size {} should be related by number of interpolation steps {}",
        out_buffer.len(),
        out_index_buffer.len(),
        info.n_interp_steps,
    );
    assert!(!nodes.is_empty(), "At least one starting node is necesary.");
    assert!(
        info.n_target_attempts != 0,
        "One attempt should be reserved for reaching the target directly"
    );

    // Add n_target_attempts samples near target within angle/distance
    let target_transform = Transform::from_homogeneous_and_scale(info.target, nodes[0].scale);
    let target_transform_quat = TransformQuat::from_transform(&target_transform);
    sample_buffer[0] = target_transform_quat;
    random_sampling(
        &target_transform_quat,
        &mut sample_buffer[1..info.n_target_attempts],
        info.target_angle_disturbance,
        info.target_distance_disturbance,
        source,
    );

    // Fill with random samples
    let angle_distribution_uniform_full =
        Uniform::new(-f64::consts::PI + 1e-4, f64::consts::PI - 1e-4);
    let point_dists = [
        Uniform::new(info.pos_bounds.min[0] as f64, info.pos_bounds.max[0] as f64),
        Uniform::new(info.pos_bounds.min[1] as f64, info.pos_bounds.max[1] as f64),
        Uniform::new(info.pos_bounds.min[2] as f64, info.pos_bounds.max[2] as f64),
    ];
    for s in &mut sample_buffer[info.n_target_attempts..] {
        let random_quat = Vec4::random_quat(source, &angle_distribution_uniform_full);
        let random_point = Vec3(array::from_fn(|i| point_dists[i].sample(source) as f32));
        *s = TransformQuat {
            scale: nodes[0].scale,
            t: random_point,
            q: random_quat,
        }
    }

    for (sample, (out_index, out_slice)) in sample_buffer.iter().zip(
        out_index_buffer
            .iter_mut()
            .zip(out_buffer.chunks_mut(info.n_interp_steps)),
    ) {
        // Find nearest
        let (idx_min_node, min_node, (d_r, d_phi)) = nodes
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t, t.distances(sample.t, sample.q)))
            .min_by(
                |&(_i1, _t1, (d_r_1, d_phi_1)), &(_i2, _t2, (d_r_2, d_phi_2))| {
                    f32::total_cmp(
                        &(d_r_1 + d_phi_1 * info.angle_distance_tradeoff),
                        &(d_r_2 + d_phi_2 * info.angle_distance_tradeoff),
                    )
                },
            )
            .unwrap();
        // Ensure maximum step size is respected
        let target_quat = if d_phi <= info.step_size.angle {
            sample.q
        } else {
            math::slerp(min_node.q, sample.q, info.step_size.angle / d_phi)
        };
        let target_pos = if d_r <= info.step_size.distance {
            sample.t
        } else {
            math::lerp(min_node.t, sample.t, info.step_size.distance / d_r)
        };

        min_node.interpolate(
            &TransformQuat {
                t: target_pos,
                q: target_quat,
                ..*sample
            },
            out_slice,
        );
        *out_index = idx_min_node;
    }
}

fn process_rrt_results(
    processed_transforms: &[TransformQuat],
    collisions: &[u8],
    nodes: &mut Vec<TransformQuat>,
    edges: &mut Vec<(usize, usize)>,
    processing_transforms_base_node_indices: &[usize],
    info: &RRTInfo,
) -> bool {
    // Note: scale is not set correctly as it is not necessary to compute the distance between transforms
    let target_pose =
        TransformQuat::from_transform(&Transform::from_homogeneous_and_scale(info.target, 1.0));
    // Check n_target_samples: if any has no collision for last step and final step is close to target: mark as done
    let done = collisions
        .iter()
        .zip(processed_transforms)
        .skip(info.n_interp_steps - 1)
        .step_by(info.n_interp_steps)
        .take(info.n_target_attempts)
        .any(|(c, t)| {
            *c == 0
                && t.combined_distance(&target_pose, info.angle_distance_tradeoff)
                    <= info.combinded_distance_success_threshold
        });

    let mut n_coll_free = 0;
    let mut n_coll_free_steps = 0;
    let mut max_coll_free_steps = 0;
    // Go through collisions in chunks of n_interp_steps
    // Filter for those with at least one valid step
    for (i, (colls, trs)) in collisions
        .chunks(info.n_interp_steps)
        .zip(processed_transforms.chunks(info.n_interp_steps))
        .enumerate()
        .filter(|(_i, (c, _t))| c[0] == 0)
    {
        // Count how many until collision
        let index_last_collision_free = colls.iter().skip(1).take_while(|c| **c == 0).count();

        // add final transform to nodes and edge to edges
        let new_node_index = nodes.len();
        let base_node_index = processing_transforms_base_node_indices[i];
        let new_node = trs[index_last_collision_free];

        nodes.push(new_node);
        edges.push((base_node_index, new_node_index));

        n_coll_free += 1;
        n_coll_free_steps += index_last_collision_free + 1;
        max_coll_free_steps = max_coll_free_steps.max(index_last_collision_free + 1);
    }
    println!(
        "Collision free: {:.2}%; Avg number of steps of collision free: {:.2}; Max: {max_coll_free_steps}",
        n_coll_free as f32 / processing_transforms_base_node_indices.len() as f32 * 100.,
        n_coll_free_steps as f32 / n_coll_free as f32
    );

    // Find nearest
    let (idx_min_node, _min_node, (d_r, d_phi)) = nodes
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t, t.distances(target_pose.t, target_pose.q)))
        .min_by(
            |&(_i1, _t1, (d_r_1, d_phi_1)), &(_i2, _t2, (d_r_2, d_phi_2))| {
                f32::total_cmp(
                    &(d_r_1 + d_phi_1 * info.angle_distance_tradeoff),
                    &(d_r_2 + d_phi_2 * info.angle_distance_tradeoff),
                )
            },
        )
        .unwrap();
    println!(
        "Nearest node is now {idx_min_node} with d={d_r:.6}, phi={d_phi:.6}, d_total={:.6}",
        d_r + d_phi * info.angle_distance_tradeoff
    );
    let (idx_min_node, _min_node, d_r) = nodes
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t, t.distances(target_pose.t, target_pose.q).0))
        .min_by(|&(_i1, _t1, d_1), &(_i2, _t2, d_2)| f32::total_cmp(&(d_1), &(d_2)))
        .unwrap();
    println!("Nearest node by distance is now {idx_min_node} with d={d_r:.6}");
    let (idx_min_node, _min_node, d_phi) = nodes
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t, t.distances(target_pose.t, target_pose.q).1))
        .min_by(|&(_i1, _t1, d_1), &(_i2, _t2, d_2)| f32::total_cmp(&(d_1), &(d_2)))
        .unwrap();
    println!("Nearest node by angle is now {idx_min_node} with d={d_phi:.6}");

    let (idx_min_node, min_node) = nodes
        .iter()
        .enumerate()
        .min_by(|&(_i1, t1), &(_i2, t2)| f32::total_cmp(&t2.t.0[2], &t1.t.0[2]))
        .unwrap();
    let mut path = vec![idx_min_node];
    while !path.contains(&0) {
        path.push(
            edges
                .iter()
                .find(|(prev, curr)| curr == path.last().unwrap())
                .map(|(prev, curr)| *prev)
                .unwrap(),
        );
    }

    println!(
        "Highest z node is now {idx_min_node} with z={:.6}; x={:.6},y={:.6}, phi={:6}, path={:?}, R={:?}, t={:?}",
        min_node.t.0[2],
        min_node.t.0[0],
        min_node.t.0[1],
        min_node.q.quat_relative_angle(target_pose.q),
        path,
        Matrix3::from_quat(min_node.q).0,
        min_node.t.0,
    );

    done
}

fn solving_trajectory(
    info: &RRTInfo,
    nodes: &[TransformQuat],
    edges: &[(usize, usize)],
) -> Vec<TransformQuat> {
    let target_pose =
        TransformQuat::from_transform(&Transform::from_homogeneous_and_scale(info.target, 1.0));
    // find nearest
    let idx_min_node = nodes
        .iter()
        .enumerate()
        .min_by(|&(_i1, t1), &(_i2, t2)| {
            f32::total_cmp(
                &t1.combined_distance(&target_pose, info.angle_distance_tradeoff),
                &t2.combined_distance(&target_pose, info.angle_distance_tradeoff),
            )
        })
        .unwrap()
        .0;
    // go backwards
    let mut path = vec![idx_min_node];
    while !path.contains(&0) {
        path.push(
            edges
                .iter()
                .find(|(_prev, curr)| curr == path.last().unwrap())
                .map(|(prev, _curr)| *prev)
                .unwrap(),
        );
    }
    path.iter().rev().map(|idx| nodes[*idx]).collect()
}

fn show_collision_statistics(collisions: &[u8]) {
    let collision_free = collisions.iter().filter(|x| **x == 0).count();
    let collision_obj_1 = collisions.iter().filter(|x| **x == 1).count();
    let collision_obj_2 = collisions.iter().filter(|x| **x == 2).count();
    let collision_plane = collisions.iter().filter(|x| **x == 254).count();
    let n_f64 = collisions.len() as f64;
    println!(
        "GPU Collisions: {:.2}% free; {:.2}% with object 1; {:.2}% with object 2; {:.2}% with ground plane",
        collision_free as f64 / n_f64 * 100.,
        collision_obj_1 as f64 / n_f64 * 100.,
        collision_obj_2 as f64 / n_f64 * 100.,
        collision_plane as f64 / n_f64 * 100.
    );
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
    let transform_from_obj = Transform::from_homogeneous_and_scale(config.o_D_obj, scale);

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

    let static_info = glue::TaskStaticInfo {
        transform_into_1: task.transform_into_1,
        transform_into_2: task.transform_into_2,
        index_list_obj: index_list,
        num_indices: task.num_indices,
        voxel_grid: task.grid_static,
    };
    let rrt_info = config.rrt_info;

    let mut gpu_buffers = glue::setup_gpu(&static_info, N);

    let mut rng_source =
        probability::source::Default::new([3589646354398292094, 6717470606376064352]);

    const N: usize = 5120 * 4;
    let mut gpu_input_buffer_cpu = vec![TransformQuat::default(); N];
    let mut gpu_output_buffer_cpu = vec![255_u8; N];
    let mut currently_processing_base_node_indices = vec![0_usize; N / rrt_info.n_interp_steps];

    let mut nodes = vec![TransformQuat::from_transform(&transform_from_obj)];
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut rrt_sample_buffer = vec![TransformQuat::default(); N / rrt_info.n_interp_steps];

    loop {
        rrt(
            &nodes,
            &rrt_info,
            &mut gpu_input_buffer_cpu,
            &mut currently_processing_base_node_indices,
            &mut rrt_sample_buffer,
            &mut rng_source,
        );

        glue::gpu_driver(
            &mut gpu_buffers,
            &gpu_input_buffer_cpu,
            &mut gpu_output_buffer_cpu,
        );

        let done = process_rrt_results(
            &gpu_input_buffer_cpu,
            &gpu_output_buffer_cpu,
            &mut nodes,
            &mut edges,
            &currently_processing_base_node_indices,
            &rrt_info,
        );
        println!("Done? {done}");

        show_collision_statistics(&gpu_output_buffer_cpu);
        if done {
            break;
        }
    }

    let traj: Vec<Matrix34> = solving_trajectory(&rrt_info, &nodes, &edges)
        .into_iter()
        .map(Transform::from_transform_quat_unit_scale)
        .map(|t| t.matrix)
        .collect();
    write_json_results(
        "./scenes/scene_1_trajectory.json",
        serde_json::to_string(&traj).unwrap(),
    )
    .unwrap();

    Ok(())
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

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

        let mut interps = [TransformQuat::default(); 3];
        tq0.interpolate(&tq3, &mut interps);

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

    #[test]
    fn test_transform_inverse() {
        let m = Matrix34([
            [0.99884413, -0.00170316, 0.04803638, 1.],
            [0.00386124, -0.9932992, -0.11550674, 2.],
            [0.04791122, 0.11555871, -0.9921445, 3.],
        ]);
        let d = Transform::from_homogeneous_and_scale(m, 5.);
        let should_be_identity = d.and_then(&d.inverse());
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(
                        (should_be_identity.matrix.0[i][j] - 1.).abs() < 5e-5,
                        "Matrix (R) at index {i} {j} should be 1 but is {:.3}",
                        should_be_identity.matrix.0[i][j]
                    );
                } else {
                    assert!(
                        should_be_identity.matrix.0[i][j].abs() < 5e-5,
                        "Matrix (R) at index {i} {j} should be 0 but is {:.3}",
                        should_be_identity.matrix.0[i][j]
                    );
                }
            }
            assert!(
                should_be_identity.matrix.0[i][3].abs() < 5e-5,
                "Matrix (T) at index {i} should be 0 but is {:.3}",
                should_be_identity.matrix.0[i][3]
            );
        }
    }

    #[test]
    fn test_index_list_voxel_grid_identity() {
        let config = load_json_config("./scenes/scene_1.json").unwrap();
        let (voxel_data, dims) = read_voxel_data(&config.voxel_grid_fixed_path).unwrap();

        let data = voxel_data.leak().as_ptr();
        let grid = VoxelGrid { data, dims };
        let index_list = read_index_list(&config.voxel_grid_obj_path).unwrap();

        for idx in 0..index_list.len() as usize {
            let U32_3 { x, y, z } = unsafe { *index_list.as_ptr().wrapping_add(idx) };
            if grid[(x, y, z)] != 1 {
                panic!("Mismatch between index list and Voxel grid at index {x} {y} {z}");
            }
        }

        let index_set: HashSet<U32_3> = HashSet::from_iter(index_list);
        for x in 0..dims.x {
            for y in 0..dims.y {
                for z in 0..dims.z {
                    if grid[(x, y, z)] == 1 && !index_set.contains(&U32_3 { x, y, z }) {
                        panic!(
                            "Mismatch between Voxel grid and index list at index {x} {y} {z} (index set does not have index)"
                        );
                    }
                    if grid[(x, y, z)] == 0 && index_set.contains(&U32_3 { x, y, z }) {
                        panic!(
                            "Mismatch between Voxel grid and index list at index {x} {y} {z} (index set has spurious index)"
                        );
                    }
                }
            }
        }
    }

    #[allow(non_snake_case)]
    fn test_projected_indices(task: &mut Task) {
        let o_D_obj = task.transform_from_obj;
        for idx in 0..task.num_indices as usize {
            let U32_3 { x, y, z } = unsafe { *task.index_list_obj.wrapping_add(idx) };

            let idx_coords_o = o_D_obj.apply(&Vec3([x as f32, y as f32, z as f32]));

            let location_in_1 = task.transform_into_1.apply(&idx_coords_o);

            let relative_indices_in_1 = [
                location_in_1.0[0].round() as i32 - x as i32,
                location_in_1.0[1].round() as i32 - y as i32,
                location_in_1.0[2].round() as i32 - z as i32,
            ];

            let location_in_2 = task.transform_into_2.apply(&idx_coords_o);

            let relative_indices_in_2 = [
                location_in_2.0[0].round() as i32 - x as i32,
                location_in_2.0[1].round() as i32 - y as i32,
                location_in_2.0[2].round() as i32 - z as i32,
            ];

            for u in 0..3 {
                if relative_indices_in_1[u] != -relative_indices_in_2[u] {
                    panic!(
                        "Relative indices do not match for {x} {y} {z}; {u}: {relative_indices_in_1:?} vs {relative_indices_in_2:?}"
                    )
                }
            }
        }

        unsafe { *task.out_collision = 0 };
    }

    #[test]
    fn test_equal_projection() {
        let config = load_json_config("./scenes/scene_1.json").unwrap();

        let scale = config.o_S_x;

        let static_transforms: Vec<_> = config
            .o_Ds_x
            .into_iter()
            .map(|matrix| Transform::from_homogeneous_and_scale(Matrix34(matrix), scale).inverse())
            .collect();
        let transform_from_obj = Transform::from_homogeneous_and_scale(config.o_D_obj, scale);

        let (voxel_data, dims) = read_voxel_data(&config.voxel_grid_fixed_path).unwrap();

        let data = voxel_data.leak().as_ptr();
        let grid = VoxelGrid { data, dims };

        let index_list = read_index_list(&config.voxel_grid_obj_path).unwrap();

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
        test_projected_indices(&mut task);
    }
}
