#include <cuda_runtime.h>
#include <cstdio>
int main() {
 int n=0; if(cudaGetDeviceCount(&n)!=cudaSuccess || n!=1) return 1;
 cudaDeviceProp p{}; if(cudaGetDeviceProperties(&p,0)!=cudaSuccess) return 2;
 printf("{\"name\":\"%s\",\"cc\":\"%d.%d\",\"sm_count\":%d,\"memory_bytes\":%zu,\"l2_bytes\":%d,\"memory_bus_bits\":%d}\n",p.name,p.major,p.minor,p.multiProcessorCount,p.totalGlobalMem,p.l2CacheSize,p.memoryBusWidth);
 return 0;
}
