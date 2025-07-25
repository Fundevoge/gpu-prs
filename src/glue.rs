use std::{
    ffi::CString,
    mem,
    os::raw::c_void,
    sync::LazyLock,
    time::{Duration, Instant},
};

use cust::{
    error::{CudaError, CudaResult},
    memory::{DeviceBox, DeviceCopy, DeviceMemory},
    module::{JitTarget, ModuleJitOption, OptLevel},
    prelude::*,
};
use cust_raw::cudaError_enum;

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

pub struct TaskGpuBuffers {
    task_static_info: DeviceBox<TaskStaticInfoGpu>,
    input: DeviceBuffer<TransformQuat>,
    output: DeviceBuffer<u8>,
    _index_list_obj: DeviceBuffer<U32_3>,
    _voxel_data: DeviceBuffer<u8>,
}

pub struct GpuContext {
    buffers: TaskGpuBuffers,
    _ctx: Context,
    module: Module,
    stream: Stream,
}

static PTX: LazyLock<CString> =
    std::sync::LazyLock::new(|| CString::new(include_str!("../kernel/kernel.ptx")).unwrap());

fn to_result(input: cudaError_enum) -> CudaResult<()> {
    match input {
        cudaError_enum::CUDA_SUCCESS => Ok(()),
        cudaError_enum::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
        cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
        cudaError_enum::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
        cudaError_enum::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
        cudaError_enum::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
        cudaError_enum::CUDA_ERROR_PROFILER_NOT_INITIALIZED => {
            Err(CudaError::ProfilerNotInitialized)
        }
        cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STARTED => {
            Err(CudaError::ProfilerAlreadyStarted)
        }
        cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STOPPED => {
            Err(CudaError::ProfilerAlreadyStopped)
        }
        cudaError_enum::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
        cudaError_enum::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
        cudaError_enum::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
        cudaError_enum::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
        cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Err(CudaError::ContextAlreadyCurrent),
        cudaError_enum::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
        cudaError_enum::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
        cudaError_enum::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
        cudaError_enum::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
        cudaError_enum::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
        cudaError_enum::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
        cudaError_enum::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
        cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
        cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
        cudaError_enum::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
        cudaError_enum::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
        cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(CudaError::ContextAlreadyInUse),
        cudaError_enum::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Err(CudaError::PeerAccessUnsupported),
        cudaError_enum::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
        cudaError_enum::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => {
            Err(CudaError::InvalidGraphicsContext)
        }
        cudaError_enum::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
        cudaError_enum::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSource),
        cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
        cudaError_enum::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
            Err(CudaError::SharedObjectSymbolNotFound)
        }
        cudaError_enum::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
            Err(CudaError::SharedObjectInitFailed)
        }
        cudaError_enum::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
        cudaError_enum::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
        cudaError_enum::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
        cudaError_enum::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
        cudaError_enum::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
        cudaError_enum::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(CudaError::LaunchOutOfResources),
        cudaError_enum::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
        cudaError_enum::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
            Err(CudaError::LaunchIncompatibleTexturing)
        }
        cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
            Err(CudaError::PeerAccessAlreadyEnabled)
        }
        cudaError_enum::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(CudaError::PeerAccessNotEnabled),
        cudaError_enum::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(CudaError::PrimaryContextActive),
        cudaError_enum::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
        cudaError_enum::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
        cudaError_enum::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
        cudaError_enum::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
            Err(CudaError::HostMemoryAlreadyRegistered)
        }
        cudaError_enum::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
            Err(CudaError::HostMemoryNotRegistered)
        }
        cudaError_enum::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
        cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
        cudaError_enum::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
        cudaError_enum::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
        cudaError_enum::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
        cudaError_enum::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
        cudaError_enum::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
        cudaError_enum::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
        _ => Err(CudaError::UnknownError),
    }
}

fn copy_from_cpu<T: DeviceCopy>(src: &[T], dest: &mut DeviceSlice<T>) -> CudaResult<()> {
    let size = mem::size_of_val(src).min(mem::size_of::<T>() * dest.len());
    if size != 0 {
        unsafe {
            to_result(cust_raw::cuMemcpyHtoD_v2(
                dest.as_raw_ptr(),
                src.as_ptr() as *const c_void,
                size,
            ))?
        }
    }
    Ok(())
}

fn copy_to_cpu<T: DeviceCopy>(src: &DeviceSlice<T>, dest: &mut [T]) -> CudaResult<()> {
    let size = (mem::size_of::<T>() * src.len()).min(std::mem::size_of_val(dest));
    if size != 0 {
        unsafe {
            to_result(cust_raw::cuMemcpyDtoH_v2(
                dest.as_mut_ptr() as *mut c_void,
                src.as_device_ptr().as_raw(),
                size,
            ))?
        }
    }
    Ok(())
}

pub fn setup_gpu(task_static_info: &TaskStaticInfo, buffer_size: usize) -> GpuContext {
    let _ctx = cust::quick_init().unwrap();

    let module = Module::from_ptx_cstr(
        &PTX,
        &[
            ModuleJitOption::OptLevel(OptLevel::O4),
            ModuleJitOption::Target(JitTarget::Compute86),
        ],
    )
    .unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

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

    let collision_buffer: DeviceBuffer<u8> =
        unsafe { DeviceBuffer::uninitialized(buffer_size) }.unwrap();
    let transforms_buffer: DeviceBuffer<TransformQuat> =
        unsafe { DeviceBuffer::uninitialized(buffer_size) }.unwrap();

    GpuContext {
        _ctx,
        module,
        stream,
        buffers: TaskGpuBuffers {
            task_static_info: task_static_info_gpu_box,
            input: transforms_buffer,
            output: collision_buffer,
            _index_list_obj: index_list_obj_buffer,
            _voxel_data: voxel_grid_data_buffer,
        },
    }
}

pub(crate) fn gpu_driver(gpu_context: &mut GpuContext, input: &[TransformQuat], output: &mut [u8]) {
    assert!(
        input.len() <= gpu_context.buffers.input.len(),
        "Input does not fit into GPU buffer"
    );

    let collision_kernel = gpu_context.module.get_function("run_task_kernel").unwrap();

    let start = Instant::now();
    copy_from_cpu(input, &mut gpu_context.buffers.input).unwrap();

    const THREADS_PER_BLOCK: u32 = 256;
    let num_blocks = (input.len() as u32).div_ceil(THREADS_PER_BLOCK);

    let stream = &mut gpu_context.stream;

    unsafe {
        launch!(collision_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
            gpu_context.buffers.input.as_device_ptr(),
            gpu_context.buffers.output.as_device_ptr(),
            gpu_context.buffers.task_static_info.as_device_ptr(),
            input.len(),
        ))
    }
    .unwrap();

    std::thread::sleep(Duration::from_millis(10));

    stream.synchronize().unwrap();

    copy_to_cpu(&gpu_context.buffers.output, output).unwrap();
    let duration = start.elapsed();
    println!("GPU time: {duration:?}");
}
