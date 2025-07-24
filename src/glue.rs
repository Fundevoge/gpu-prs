use std::{ffi::CString, time::Instant};

use cust::{
    memory::DeviceCopy,
    module::{JitTarget, ModuleJitOption, OptLevel},
    prelude::*,
};

use crate::{Transform, TransformQuat, U32_3, VoxelGrid};

pub(crate) struct TaskStaticInfo {
    pub(crate) transform_into_1: Transform,
    pub(crate) transform_into_2: Transform,
    pub(crate) index_list_obj: Vec<U32_3>,
    pub(crate) num_indices: u32,
    pub(crate) voxel_grid: VoxelGrid,
}

#[repr(C)]
#[derive(DeviceCopy, Clone, Copy)]
struct TaskStaticInfoGpu {
    transform_into_1: Transform,
    transform_into_2: Transform,
    index_list_obj: DevicePointer<U32_3>,
    num_indices: u32,
    voxel_grid: VoxelGridGpu,
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
struct VoxelGridGpu {
    data: DevicePointer<u8>,
    dims: U32_3,
}

pub(crate) fn gpu_driver(
    task_static_info: TaskStaticInfo,
    input: &[TransformQuat],
    output: &mut [u8],
) {
    let _ctx = cust::quick_init().unwrap();
    let ptx = CString::new(include_str!("../kernel/kernel.ptx")).unwrap();
    let module = Module::from_ptx_cstr(
        &ptx,
        &[
            ModuleJitOption::OptLevel(OptLevel::O4),
            ModuleJitOption::Target(JitTarget::Compute86),
        ],
    )
    .unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    // Prepare static data
    let index_list_obj_buffer = DeviceBuffer::from_slice(&task_static_info.index_list_obj).unwrap();
    let voxel_grid_data_buffer =
        DeviceBuffer::from_slice(task_static_info.voxel_grid.as_slice()).unwrap();
    let voxel_grid = VoxelGridGpu {
        data: voxel_grid_data_buffer.as_device_ptr(),
        dims: task_static_info.voxel_grid.dims,
    };

    let task_static_info_gpu = TaskStaticInfoGpu {
        transform_into_1: task_static_info.transform_into_1,
        transform_into_2: task_static_info.transform_into_2,
        index_list_obj: index_list_obj_buffer.as_device_ptr(),
        num_indices: task_static_info.num_indices,
        voxel_grid,
    };
    let task_static_info_gpu_box = task_static_info_gpu.as_dbox().unwrap();

    let collision_kernel = module.get_function("run_task_kernel").unwrap();

    let collision_buffer: DeviceBuffer<u8> =
        unsafe { DeviceBuffer::uninitialized(output.len()) }.unwrap();
    let transforms_buffer: DeviceBuffer<TransformQuat> = DeviceBuffer::from_slice(input).unwrap();

    let start = Instant::now();

    unsafe {
        launch!(collision_kernel<<<1, 1, 0, stream>>>(
            transforms_buffer.as_device_ptr(), collision_buffer.as_device_ptr(), task_static_info_gpu_box.as_device_ptr(), 1
        ))
    }
    .unwrap();

    stream.synchronize().unwrap();
    let duration = start.elapsed();
    println!("GPU time: {duration:?}");

    collision_buffer.copy_to(output).unwrap();
}
