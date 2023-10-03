#include <metal_stdlib>

using namespace metal;

kernel void array_add_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] + B[index];
}

kernel void array_sub_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] - B[index];
}

kernel void array_mul_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] * B[index];
}

kernel void array_div_f(
  device const float* A,
  device const float* B,
  device float* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] / B[index];
}
