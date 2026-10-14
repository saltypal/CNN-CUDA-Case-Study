#include <algorithm>
#include <omp.h>

// NCHW input, KCRS weights, valid stride-one cross-correlation.
// The caller owns all memory. Each output is written by exactly one worker.
extern "C" void conv_forward_cpu(const float* x, const float* w,
    const float* bias, float* y, int n, int c, int h, int width,
    int k, int r, int threads) {
    const int p = h - r + 1;
    const int q = width - r + 1;
    const int outputs = n * k * p * q;
    #pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
    for (int index = 0; index < outputs; ++index) {
        int temp = index;
        const int ox = temp % q; temp /= q;
        const int oy = temp % p; temp /= p;
        const int filter = temp % k;
        const int sample = temp / k;
        float sum = bias[filter];
        for (int channel = 0; channel < c; ++channel) {
            for (int ky = 0; ky < r; ++ky) {
                for (int kx = 0; kx < r; ++kx) {
                    const int xi = ((sample*c+channel)*h+oy+ky)*width+ox+kx;
                    const int wi = ((filter*c+channel)*r+ky)*r+kx;
                    sum += x[xi] * w[wi];
                }
            }
        }
        y[index] = sum;
    }
}

extern "C" void conv_backward_cpu(const float* x, const float* w,
    const float* dy, float* dx, float* dw, float* db,
    int n, int c, int h, int width, int k, int r, int threads) {
    const int p = h-r+1;
    const int q = width-r+1;
    const int weights = k*c*r*r;
    // Independent reductions avoid races and atomics.
    #pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
    for (int index=0; index<weights; ++index) {
        int temp=index;
        const int kx=temp%r; temp/=r;
        const int ky=temp%r; temp/=r;
        const int channel=temp%c;
        const int filter=temp/c;
        float sum=0.0f;
        for (int sample=0; sample<n; ++sample) {
            for (int oy=0; oy<p; ++oy) {
                for (int ox=0; ox<q; ++ox) {
                    const int xi=((sample*c+channel)*h+oy+ky)*width+ox+kx;
                    const int yi=((sample*k+filter)*p+oy)*q+ox;
                    sum+=x[xi]*dy[yi];
                }
            }
        }
        dw[index]=sum;
    }
    #pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
    for (int filter=0; filter<k; ++filter) {
        float sum=0.0f;
        for (int sample=0; sample<n; ++sample)
            for (int pos=0; pos<p*q; ++pos)
                sum+=dy[(sample*k+filter)*p*q+pos];
        db[filter]=sum;
    }
    // Training the first layer does not require an input gradient.
    if (dx == nullptr) return;
    #pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
    for (int index=0; index<n*c*h*width; ++index) {
        int temp=index;
        const int ix=temp%width; temp/=width;
        const int iy=temp%h; temp/=h;
        const int channel=temp%c;
        const int sample=temp/c;
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
}
