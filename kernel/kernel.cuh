#pragma once
#include <stdint.h>
#include <stdio.h>

struct Vec3
{
    float data[3];
};

struct Vec4
{
    float data[4];
};

struct Transform
{
    float matrix[3][4]; // row-major

    __device__ Vec3 apply(const Vec3 *vec) const
    {
        Vec3 result;
        for (uint8_t i = 0; i < 3; ++i)
        {
            result.data[i] = __fmaf_rn(matrix[i][0], vec->data[0],
                                       __fmaf_rn(matrix[i][1], vec->data[1],
                                                 __fmaf_rn(matrix[i][2], vec->data[2],
                                                           matrix[i][3])));
        }
        return result;
    }
};

struct TransformQuat
{
    float scale;
    Vec3 t;
    Vec4 q;

    __device__ Transform to_transform_matrix() const
    {
        float one = 1.;
        float two = 2.;
        Transform result = {.matrix{
            {scale * (one - two * (q.data[2] * q.data[2] + q.data[3] * q.data[3])),
             scale * (two * (q.data[1] * q.data[2] - q.data[3] * q.data[0])),
             scale * (two * (q.data[1] * q.data[3] + q.data[2] * q.data[0])),
             t.data[0]},
            {scale * (two * (q.data[1] * q.data[2] + q.data[3] * q.data[0])),
             scale * (one - two * (q.data[1] * q.data[1] + q.data[3] * q.data[3])),
             scale * (two * (q.data[2] * q.data[3] - q.data[1] * q.data[0])),
             t.data[1]},
            {scale * (two * (q.data[1] * q.data[3] - q.data[2] * q.data[0])),
             scale * (two * (q.data[2] * q.data[3] + q.data[1] * q.data[0])),
             scale * (one - two * (q.data[1] * q.data[1] + q.data[2] * q.data[2])),
             t.data[2]},
        }};

        return result;
    }
};

struct VoxelGrid
{
    uint8_t *data;
    uint3 dims;

    __device__ uint8_t at(uint32_t x, uint32_t y, uint32_t z) const
    {

        // printf("Voxel at (%u, %u, %u) = %u\n", x, y, z, data[(x * dims.y + y) * dims.z + z]);

        return data[(x * dims.y + y) * dims.z + z];
    }
};

struct TaskStaticInfoGpu
{
    Transform transform_into_1;
    Transform transform_into_2;
    uint3 *index_list_obj;
    uint32_t num_indices;
    VoxelGrid voxel_grid;
};
