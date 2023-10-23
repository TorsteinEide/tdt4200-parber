#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    
    *h_temp[2] = { NULL, NULL },
    *h_thermal_diffusivity,
    // TODO 1: Declare device side pointers to store host-side data.
    *d_temp,
    *d_temp_next,   
    *d_thermal_deffusivity,
    dt;
size_t size = (M+2 * N+2) * sizeof(real_t);
#define T(x,y)                      h_temp[0][(y) * (N + 2) + (x)]
#define T_next(x,y)                 h_temp[1][((y) * (N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    h_thermal_diffusivity[(y) * (N + 2) + (x)]
#define T_device(x,y)                      d_temp[(y) * (N + 2) + (x)]
#define T_device_next(x,y)                 d_temp_next[((y) * (N + 2) + (x))]
#define DEVICE_THERMAL_DIFFUSIVITY(x,y)    d_thermal_deffusivity[(y) * (N + 2) + (x)]

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void time_step (real_t* d_temp, real_t* d_temp_next, real_t* d_thermal_deffusivity, real_t dt);
void boundary_condition( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


int
main ( int argc, char **argv )
{
    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    M = options->M;
    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    cudaDeviceProp prop;
    int maxBlocksPerGrid;
    int maxThreadsPerBlock;
    cudaGetDeviceProperties(&prop, 0);
    maxThreadsPerBlock = prop.maxThreadsPerBlock;
    maxBlocksPerGrid = prop.maxGridSize[0];

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // TODO 6: Launch the time_step-kernel.
	int threadsPerBlock = maxThreadsPerBlock;
	int blocksPerGrid = (N + threadsPerBlock -1 ) / threadsPerBlock;
        time_step<<<blocksPerGrid, threadsPerBlock>>>(d_temp, d_temp_next, d_thermal_deffusivity, dt);

        // boundary_condition();

        // time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            // TODO 8: Copy data from device to host.
            cudaMemcpy(h_temp[0], d_temp, size, cudaMemcpyDeviceToHost);
            domain_save ( iteration );
        }

        // swap( &h_temp[0], &h_temp[1] );
        // TODO 7: Swap device pointers.
        swap( &d_temp, &d_temp_next );
    }

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );


    domain_finalize();

    exit ( EXIT_SUCCESS );
}


// TODO 4: Make time_step() a CUDA kernel
//         where one thread is responsible for one grid point.
__global__
void
time_step (real_t* d_temp, real_t* d_temp_next, real_t* d_thermal_deffusivity, real_t dt)
{
    real_t c, t, b, l, r, K, new_value;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    boundary_condition(x, y);

    c = T_device(x, y);

    t = T_device(x - 1, y);
    b = T_device(x + 1, y);
    l = T_device(x, y - 1);
    r = T_device(x, y + 1);
    K = DEVICE_THERMAL_DIFFUSIVITY(x, y);

    new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

    T_device_next(x, y) = new_value;


}


// TODO 5: Make boundary_condition() a device function and
//         call it from the time_step-kernel.
//         Chose appropriate threads to set the boundary values.
__device__
void
boundary_condition (int x, int y)
{
    
    T_device(x, 0) = T(x, 2);
    T_device(x, M+1) = T(x, M-1);

    T_device(0, y) = T(2, y);
    T_device(N+1, y) = T(N-1, y);
}


void
domain_init ( void )
{
    h_temp[0] = (real_t*) malloc ( (M+2)*(N+2) * sizeof(real_t) );
    h_temp[1] = (real_t*) malloc ( (M+2)*(N+2) * sizeof(real_t) );
    h_thermal_diffusivity = (real_t*) malloc ( (M+2)*(N+2) * sizeof(real_t) );

    // TODO 2: Allocate device memory.
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_temp_next, size);
    cudaMalloc(&d_thermal_deffusivity, size);

    dt = 0.1;

    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            h_temp[0][ y*(N+2) + x ] = temperature;
            h_temp[1][ y*(N+2) + x ] = temperature;
            h_thermal_diffusivity[ y*(N+2) + x ] = diffusivity;
        }
    }

    // TODO 3: Copy data from host to device.
    cudaMemcpy(d_temp, h_temp[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_next, h_temp[1], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_thermal_deffusivity, h_thermal_diffusivity, size, cudaMemcpyHostToDevice);
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    for ( int_t iter = 1; iter <= N; iter++)
    {
        fwrite( h_temp[0] + (M+2) * iter + 1, sizeof(real_t), N, out );
    }
    fclose ( out );
}


void
domain_finalize ( void )
{
    free ( h_temp[0] );
    free ( h_temp[1] );
    free ( h_thermal_diffusivity );

    // TODO 9: Free device memory.
    cudaFree(d_temp);
    cudaFree(d_temp_next);
    cudaFree(d_thermal_deffusivity);
}
