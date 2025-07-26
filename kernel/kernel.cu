#include "kernel.cuh"
#include <cuda_runtime.h>

extern "C" __global__ void run_task_kernel(TransformQuat *obj_transforms, uint8_t *results, TaskStaticInfoGpu *static_info, int num_tasks)
{
    uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_id >= num_tasks)
        return;

    const Transform o_D_obj = obj_transforms[task_id].to_transform_matrix();

    for (uint32_t idx = 0; idx < static_info->num_indices; ++idx)
    {

        uint3 index = static_info->index_list_obj[idx];
        Vec3 input = {{(float)index.x, (float)index.y, (float)index.z}};
        Vec3 idx_coords_o = o_D_obj.apply(&input);

        if (idx_coords_o.data[2] < 0.0f)
        {
            results[task_id] = 254;
            return;
        }

        {
            Vec3 location_in_x = static_info->transform_into_1.apply(&idx_coords_o);

            int32_t indices_in_x[3] = {
                static_cast<int32_t>(roundf(location_in_x.data[0])),
                static_cast<int32_t>(roundf(location_in_x.data[1])),
                static_cast<int32_t>(roundf(location_in_x.data[2])),
            };

            if (indices_in_x[0] >= 0 && indices_in_x[0] < static_info->voxel_grid.dims.x &&
                indices_in_x[1] >= 0 && indices_in_x[1] < static_info->voxel_grid.dims.y &&
                indices_in_x[2] >= 0 && indices_in_x[2] < static_info->voxel_grid.dims.z &&
                static_info->voxel_grid.at(
                    static_cast<uint32_t>(indices_in_x[0]),
                    static_cast<uint32_t>(indices_in_x[1]),
                    static_cast<uint32_t>(indices_in_x[2])) == 1)
            {
                results[task_id] = 1;
                return;
            }
        }
        {
            Vec3 location_in_x = static_info->transform_into_2.apply(&idx_coords_o);

            int32_t indices_in_x[3] = {
                static_cast<int32_t>(roundf(location_in_x.data[0])),
                static_cast<int32_t>(roundf(location_in_x.data[1])),
                static_cast<int32_t>(roundf(location_in_x.data[2])),
            };

            if (indices_in_x[0] >= 0 && indices_in_x[0] < static_info->voxel_grid.dims.x &&
                indices_in_x[1] >= 0 && indices_in_x[1] < static_info->voxel_grid.dims.y &&
                indices_in_x[2] >= 0 && indices_in_x[2] < static_info->voxel_grid.dims.z &&
                static_info->voxel_grid.at(
                    static_cast<uint32_t>(indices_in_x[0]),
                    static_cast<uint32_t>(indices_in_x[1]),
                    static_cast<uint32_t>(indices_in_x[2])) == 1)
            {
                results[task_id] = 2;
                return;
            }
        }
    }

    results[task_id] = 0;
}
