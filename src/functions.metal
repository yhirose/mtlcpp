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

kernel void array_add_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] + B[index];
}

kernel void array_sub_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] - B[index];
}

kernel void array_mul_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] * B[index];
}

kernel void array_div_i(
  device const int* A,
  device const int* B,
  device int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] / B[index];
}

kernel void array_add_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] + B[index];
}

kernel void array_sub_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] - B[index];
}

kernel void array_mul_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] * B[index];
}

kernel void array_div_u(
  device const unsigned int* A,
  device const unsigned int* B,
  device unsigned int* OUT,
  uint index [[thread_position_in_grid]])
{
  OUT[index] = A[index] / B[index];
}
