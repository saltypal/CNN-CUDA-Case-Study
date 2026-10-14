#include <cuda_runtime.h>

constexpr int BLOCK = 256;
constexpr int TILE = 16;

__global__ void direct_forward(const float* x, const float* w,
    const float* bias, float* y, int n, int c, int h, int width,
    int k, int r) {
    const int p=h-r+1, q=width-r+1;
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=n*k*p*q) return;
    int temp=index;
    const int ox=temp%q; temp/=q;
    const int oy=temp%p; temp/=p;
    const int filter=temp%k, sample=temp/k;
    float sum=bias[filter];
    for (int channel=0; channel<c; ++channel)
        for (int ky=0; ky<r; ++ky)
            for (int kx=0; kx<r; ++kx)
                sum+=x[((sample*c+channel)*h+oy+ky)*width+ox+kx]*
                     w[((filter*c+channel)*r+ky)*r+kx];
    y[index]=sum;
}

// Cooperatively load one input tile per channel, including its halo.
// Every thread reaches both barriers, including partial edge blocks.
__global__ void shared_forward(const float* x, const float* w,
    const float* bias, float* y, int n, int c, int h, int width,
    int k, int r) {
    extern __shared__ float tile[];
    const int p=h-r+1, q=width-r+1, side=TILE+r-1;
    const int filter=blockIdx.z%k, sample=blockIdx.z/k;
    const int ox=blockIdx.x*TILE+threadIdx.x;
    const int oy=blockIdx.y*TILE+threadIdx.y;
    const int lane=threadIdx.y*TILE+threadIdx.x;
    float sum=bias[filter];
    for (int channel=0; channel<c; ++channel) {
        for (int i=lane; i<side*side; i+=TILE*TILE) {
            const int ix=blockIdx.x*TILE+i%side;
            const int iy=blockIdx.y*TILE+i/side;
            tile[i]=(ix<width && iy<h) ? x[((sample*c+channel)*h+iy)*width+ix] : 0.0f;
        }
        __syncthreads();
        if (ox<q && oy<p)
            for (int ky=0; ky<r; ++ky)
                for (int kx=0; kx<r; ++kx)
                    sum+=tile[(threadIdx.y+ky)*side+threadIdx.x+kx]*
                         w[((filter*c+channel)*r+ky)*r+kx];
        __syncthreads();
    }
    if (ox<q && oy<p) y[((sample*k+filter)*p+oy)*q+ox]=sum;
}

__global__ void weight_gradient(const float* x, const float* dy,
    float* dw, int n, int c, int h, int width, int k, int r) {
    __shared__ float partial[BLOCK];
    const int p=h-r+1, q=width-r+1;
    int temp=blockIdx.x;
    const int kx=temp%r; temp/=r;
    const int ky=temp%r; temp/=r;
    const int channel=temp%c, filter=temp/c;
    float sum=0.0f;
    for (int i=threadIdx.x; i<n*p*q; i+=BLOCK) {
        const int ox=i%q, oy=(i/q)%p, sample=i/(p*q);
        sum+=x[((sample*c+channel)*h+oy+ky)*width+ox+kx]*
             dy[((sample*k+filter)*p+oy)*q+ox];
    }
    partial[threadIdx.x]=sum;
    __syncthreads();
    for (int stride=BLOCK/2; stride>0; stride/=2) {
        if (threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) dw[blockIdx.x]=partial[0];
}

__global__ void bias_gradient(const float* dy, float* db,
    int n, int k, int p, int q) {
    __shared__ float partial[BLOCK];
    float sum=0.0f;
    for (int i=threadIdx.x; i<n*p*q; i+=BLOCK)
        sum+=dy[((i/(p*q))*k+blockIdx.x)*p*q+i%(p*q)];
    partial[threadIdx.x]=sum;
    __syncthreads();
    for (int stride=BLOCK/2; stride>0; stride/=2) {
        if (threadIdx.x<stride) partial[threadIdx.x]+=partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) db[blockIdx.x]=partial[0];
}

__global__ void input_gradient(const float* w, const float* dy,
    float* dx, int n, int c, int h, int width, int k, int r) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=n*c*h*width) return;
    const int p=h-r+1, q=width-r+1;
    int temp=index;
    const int ix=temp%width; temp/=width;
    const int iy=temp%h; temp/=h;
    const int channel=temp%c, sample=temp/c;
    float sum=0.0f;
    for (int filter=0; filter<k; ++filter)
        for (int ky=0; ky<r; ++ky)
            for (int kx=0; kx<r; ++kx) {
                const int oy=iy-ky, ox=ix-kx;
                if (oy>=0 && oy<p && ox>=0 && ox<q)
                    sum+=dy[((sample*k+filter)*p+oy)*q+ox]*
                         w[((filter*c+channel)*r+ky)*r+kx];
            }
    dx[index]=sum;
}

__global__ void empty_kernel() {}

extern "C" int conv_forward_cuda(const float* x, const float* w,
    const float* bias, float* y, int n, int c, int h, int width,
    int k, int r, int tiled, void* stream_pointer) {
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_pointer);
    const int p=h-r+1, q=width-r+1;
    if (tiled) {
        dim3 block(TILE,TILE);
        dim3 grid((q+TILE-1)/TILE,(p+TILE-1)/TILE,n*k);
        const int shared_bytes=(TILE+r-1)*(TILE+r-1)*sizeof(float);
        shared_forward<<<grid,block,shared_bytes,stream>>>(x,w,bias,y,n,c,h,width,k,r);
    } else {
        const int outputs=n*k*p*q;
        direct_forward<<<(outputs+BLOCK-1)/BLOCK,BLOCK,0,stream>>>(x,w,bias,y,n,c,h,width,k,r);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int conv_backward_cuda(const float* x, const float* w,
    const float* dy, float* dx, float* dw, float* db,
    int n, int c, int h, int width, int k, int r, void* stream_pointer) {
    cudaStream_t stream=reinterpret_cast<cudaStream_t>(stream_pointer);
    weight_gradient<<<k*c*r*r,BLOCK,0,stream>>>(x,dy,dw,n,c,h,width,k,r);
    bias_gradient<<<k,BLOCK,0,stream>>>(dy,db,n,k,h-r+1,width-r+1);
    if (dx!=nullptr) {
        const int inputs=n*c*h*width;
        input_gradient<<<(inputs+BLOCK-1)/BLOCK,BLOCK,0,stream>>>(w,dy,dx,n,c,h,width,k,r);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_empty(void* stream_pointer) {
    empty_kernel<<<1,1,0,reinterpret_cast<cudaStream_t>(stream_pointer)>>>();
    return static_cast<int>(cudaGetLastError());
}
