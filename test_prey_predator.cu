#include "hdf5.h"
#include "hdf5_hl.h"
#include <stdio.h>
#include <cvode/cvode.h>                  /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_cuda.h>         /* access to cuda N_Vector                      */
#include <sunmatrix/sunmatrix_cusparse.h>             /* access to cusparse SUNMatrix                  */
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>   /* acess to cuSolverSp batch QR SUNLinearSolver */
#include <sundials/sundials_types.h>     /* defs. of realtype, int              */
#include "test_sunlinsol.h"
#include <sundials/sundials_math.h>
#define BATCHSIZE 1024
#define ZERO    RCONST(0.0)
#define kb      RCONST(1.3806504e-16)
#define mh      RCONST(1.67e-24)
#define gamma   RCONST(5.0/3.0)
#define _gamma_m1 RCONST(1.0/ (gamma-1.0) )

#define GRIDSIZE 32
#define BLOCKSIZE 32

#define T0 RCONST(0.0)
#define T1 RCONST(1e10)
#define TMULT RCONST(2.0)
#define NOUT 10 
// define our datatype
typedef struct
{
    double nbins;
    double dbin;
    double idbin;
    double lb;
    double ub;

    double current_z;
    double *Ts;
    double *logTs;
    double *Tdef;
    double *dTs_ge;
    double *Tge;

    // cooling & chemical tables
    double *r_exp_growth_prey;
    double *rs_exp_growth_prey;
    double *drs_exp_growth_prey;
    double *r_natural_death_predator;
    double *rs_natural_death_predator;
    double *drs_natural_death_predator;
    double *r_predation;
    double *rs_predation;
    double *drs_predation;

    // for now, we ignore the temperature dependent Gamma
    /*
    double *g_gamma;
    double *g_dgamma_dT;
    double *gamma;
    double *dgamma_dT;
    double *_gamma_dT;
    double *g_gamma;
    double *g_dgamma_dT;
    double *gamma;
    double *dgamma_dT;
    double *_gamma_dT;
    */


} abc_data;


// Initialize a data object that stores the reaction/ cooling rate data
abc_data abc_setup_data(int *NumberOfFields, char ***FieldNames)
{

    //-----------------------------------------------------
    // Function : abc_setup_data
    // Description: Initialize a data object that stores the reaction/ cooling rate data 
    //-----------------------------------------------------


    // let's not re-scale the data yet...
    abc_data ratedata;

    ratedata.nbins = 1024;
    ratedata.dbin = (log( 100000000.0)-log(1.0)) / 1023;
    ratedata.idbin = 1.0 / ratedata.dbin;
    ratedata.lb   = log(1.0);
    ratedata.ub   = log(100000000.0);

    /* Redshift-related pieces */
    /*
    data->z_bounds[0] = 0.0;
    data->z_bounds[1] = 10.0;
    data->n_zbins = 0 - 1;
    data->d_zbin = (log(data->z_bounds[1] + 1.0) - log(data->z_bounds[0] + 1.0)) / data->n_zbins;
    data->id_zbin = 1.0L / data->d_zbin;
    */

    
    // initialize memory space for reaction rates and cooling rates
    // we use managed data, so the pointer can simultaneously be accessed from device and the host
    cudaMallocManaged(&ratedata.r_exp_growth_prey, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.rs_exp_growth_prey, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.drs_exp_growth_prey, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.r_natural_death_predator, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.rs_natural_death_predator, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.drs_natural_death_predator, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.r_predation, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.rs_predation, sizeof(double)*1024);
    cudaMallocManaged(&ratedata.drs_predation, sizeof(double)*1024);

    // Cooling Rates

    // initialize memory space for the temperature-related pieces
    cudaMallocManaged(&ratedata.Ts, sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.logTs, sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.Tdef,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.dTs_ge,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.Tge,  sizeof(double)* BATCHSIZE);

    // gamma as a function of temperature
    /*
    cudaMallocManaged(&ratedata.g_gammaH2_1, sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.g_dgammaH2_1_dT,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.gammaH2_1,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.dgamma_dTH2_1,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata._gammaH2_1_dT, sizeof(double)*BATCHSIZE);
    cudaMallocManaged(&ratedata.g_gammaH2_2, sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.g_dgammaH2_2_dT,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.gammaH2_2,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.dgamma_dTH2_2,  sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata._gammaH2_2_dT, sizeof(double)*BATCHSIZE);
    
    // maybe we can calculate the density on the fly
    // space to store the mass density
    cudaMallocManaged(&ratedata.mdensity, sizeof(double)* BATCHSIZE);
    cudaMallocManaged(&ratedata.inv_mdensity, sizeof(double)* BATCHSIZE);
    */
    // extra stuff like the density-dependent cooling rate
    
    return ratedata;
}


void abc_read_rate_tables(abc_data *data)
{
    hid_t file_id = H5Fopen("abc_tables.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    /* Allocate the correct number of rate tables */
    H5LTread_dataset_double(file_id, "/exp_growth_prey", data->r_exp_growth_prey);
    H5LTread_dataset_double(file_id, "/natural_death_predator", data->r_natural_death_predator);
    H5LTread_dataset_double(file_id, "/predation", data->r_predation);
    
    H5Fclose(file_id);
}


void abc_read_cooling_tables(abc_data *data)
{

    hid_t file_id = H5Fopen("abc_tables.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    /* Allocate the correct number of rate tables */

    H5Fclose(file_id);
}

/*
void abc_read_gamma(abc_data *data)
{

    hid_t file_id = H5Fopen("abc_tables.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
    // Allocate the correct number of rate tables
    H5LTread_dataset_double(file_id, "/gammaH2_1",
                            data->g_gammaH2_1 );
    H5LTread_dataset_double(file_id, "/dgammaH2_1_dT",
                            data->g_dgammaH2_1_dT );   
    
    H5LTread_dataset_double(file_id, "/gammaH2_2",
                            data->g_gammaH2_2 );
    H5LTread_dataset_double(file_id, "/dgammaH2_2_dT",
                            data->g_dgammaH2_2_dT );   
    

    H5Fclose(file_id);

}
 
*/


// interpolation kernel
// ideally, we should use texture to do interpolation,
// let's ignore it for now, cos i guess most time is spent in doing the matrix thingy

__global__
void linear_interpolation_kernel(abc_data data)
{
    int j = threadIdx.x + blockDim.x* blockIdx.x;
    
    int k;
    double Tdef, t1;
    double *exp_growth_prey = data.r_exp_growth_prey;
    double *rs_exp_growth_prey  = data.rs_exp_growth_prey;
    double *drs_exp_growth_prey = data.drs_exp_growth_prey;
    double *natural_death_predator = data.r_natural_death_predator;
    double *rs_natural_death_predator  = data.rs_natural_death_predator;
    double *drs_natural_death_predator = data.drs_natural_death_predator;
    double *predation = data.r_predation;
    double *rs_predation  = data.rs_predation;
    double *drs_predation = data.drs_predation;
    
    if (j < BATCHSIZE)
    {
        k    = __float2int_rz(data.idbin*data.logTs[j] - data.lb);
        t1   = data.lb + k*data.dbin;
        Tdef = (data.logTs[j] - t1) * data.idbin;
        rs_exp_growth_prey[j] = Tdef*exp_growth_prey[k+1] + (-exp_growth_prey[k]*Tdef + exp_growth_prey[k]);
        rs_natural_death_predator[j] = Tdef*natural_death_predator[k+1] + (-natural_death_predator[k]*Tdef + natural_death_predator[k]);
        rs_predation[j] = Tdef*predation[k+1] + (-predation[k]*Tdef + predation[k]);
        
    }
}


__global__
static void rhs_kernel(double y, double *ydata, double *ydotdata, abc_data data)
{
    int i = blockIdx.x* blockDim.x + threadIdx.x;

    int nchem  = 5;
    int groupi = i * nchem; 

    // get rate pointer
    double *exp_growth_prey = data.rs_exp_growth_prey;
    double *natural_death_predator = data.rs_natural_death_predator;
    double *predation = data.rs_predation;

    
    
    

    int j;
    double z, T, mdensity, inv_mdensity;

    if (i < BATCHSIZE)
    {
        T = data.Ts[i];
        z = data.current_z;

        
        
        double dead_predator = ydata[groupi+0];
        double dead_prey = ydata[groupi+1];
        double ge = ydata[groupi+2];
        double predator = ydata[groupi+3];
        double prey = ydata[groupi+4];

        mdensity     = mh*(1.0*dead_predator + 1.0*dead_prey + 1.0*predator + 1.0*prey);
        inv_mdensity = 1.0/mdensity;
        //
        // Species: dead_predator
        //
        j = 0;
        ydotdata[groupi+j] = natural_death_predator[i]*predator;
        
        j++;
        //
        // Species: dead_prey
        //
        j = 1;
        ydotdata[groupi+j] = predation[i]*predator*prey;
        
        j++;
        //
        // Species: ge
        //
        j = 2;
        ydotdata[groupi+j] = 0;
        
        ydotdata[groupi+j] *= inv_mdensity;
        
        j++;
        //
        // Species: predator
        //
        j = 3;
        ydotdata[groupi+j] = -natural_death_predator[i]*predator + 0.75*predation[i]*predator*prey;
        
        j++;
        //
        // Species: prey
        //
        j = 4;
        ydotdata[groupi+j] = exp_growth_prey[i]*prey - predation[i]*predator*prey;
        
        j++;

    }
    
    /*
    for (int ii = 0; ii < 5; ii++)
    {
        printf("from %d: ydot[%d] = %0.5g; ydata = %0.5g\n", i, ii, ydotdata[groupi+ii], ydata[groupi+ii]);
    }
    */

}


__global__
void temperature_kernel(double* ydata, abc_data data)
{
    int i = blockIdx.x* blockDim.x + threadIdx.x;
    int nchem  = 5;
    int groupi = i * nchem; 

    double *temperature = data.Ts;
    double *logTs      = data.logTs;
    double *Tge        = data.Tge;

    double gammaH2_1 = 7./5.;
    double gammaH2_2 = 7./5.;
    // as of now just do not use any "temperature-dependent" gamma
    // which simplifies my life, and not having the need to iterate it to convergence

    if (i < BATCHSIZE)
    {
        double dead_predator = ydata[groupi+0];
        double dead_prey = ydata[groupi+1];
        double ge = ydata[groupi+2];
        double predator = ydata[groupi+3];
        double prey = ydata[groupi+4];
        double density = 1.0*dead_predator + 1.0*dead_prey + 1.0*predator + 1.0*prey;
        temperature[i] = 100; //density*ge*mh/(kb*(_gamma_m1*dead_predator + _gamma_m1*dead_prey + _gamma_m1*predator + _gamma_m1*prey));
        logTs      [i] = log(temperature[i]);
        Tge        [i] = 0.0; //TODO: update this to dT_dge;
    }
}

// Function Called by the solver
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
    abc_data *udata = (abc_data *) user_data;
    double *ydata    = N_VGetDeviceArrayPointer_Cuda(y);
    double *ydotdata = N_VGetDeviceArrayPointer_Cuda(ydot);

    // calculate temperature kernel
    temperature_kernel<<<GRIDSIZE, BLOCKSIZE>>> (ydata, *udata);
    // interpolate the rates with updated temperature
    linear_interpolation_kernel<<<GRIDSIZE, BLOCKSIZE>>>(*udata);

    // update ydot with the kernel function
    rhs_kernel<<<GRIDSIZE, BLOCKSIZE>>>(t, ydata, ydotdata, *udata);

    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr,
        ">>> ERROR in f: cudaGetLastError returned %s\n",
        cudaGetErrorName(cuerr));
        return(-1);
    }


    return 0;

}


// write jacobian



/*
 * taken from cvRoberts_block_cusolversp_batchqr.cu
 *
 * Jacobian initialization routine. This sets the sparisty pattern of
 * the blocks of the Jacobian J(t,y) = df/dy. This is performed on the CPU,
 * and only occurs at the beginning of the simulation.
 */
static int blockJacInit(SUNMatrix J)
{
    int nchem = 5;
    int nnz = 5*5;
    
    int rowptrs[nchem+1];
    int colvals[nnz];

    SUNMatZero(J);
    for (int r = 0; r < nchem+1; r++)
    {
        rowptrs[r] = r*nchem;
        printf("rowptrs[%d] = %d\n", r, rowptrs[r]);
    }

    int bIdx;
    for (int c = 0; c < nnz; c++)
    {
        bIdx = c /nnz; 
        colvals[c] = bIdx*nchem + c%nchem;
        printf("colvals[%d] = %d\n", c, colvals[c]);
    }
    // copy rowptrs, colvals to the device
    SUNMatrix_cuSparse_CopyToDevice(J, NULL, rowptrs, colvals);
    cudaDeviceSynchronize();
    return (0);
}
static int JacInit(SUNMatrix J)
{
    int rowptrs[5+1];
    int colvals[5 * 5];

    /* Zero out the Jacobian */
    SUNMatZero(J);

    /* there are 5 entries per row*/
    rowptrs[0] = 0 * 5;
    rowptrs[1] = 1 * 5;
    rowptrs[2] = 2 * 5;
    rowptrs[3] = 3 * 5;
    rowptrs[4] = 4 * 5;
    
    // 0 row of block
    colvals[0] = 0;
    colvals[1] = 1;
    colvals[2] = 2;
    colvals[3] = 3;
    colvals[4] = 4;
    
    // 1 row of block
    colvals[5] = 0;
    colvals[6] = 1;
    colvals[7] = 2;
    colvals[8] = 3;
    colvals[9] = 4;
    
    // 2 row of block
    colvals[10] = 0;
    colvals[11] = 1;
    colvals[12] = 2;
    colvals[13] = 3;
    colvals[14] = 4;
    
    // 3 row of block
    colvals[15] = 0;
    colvals[16] = 1;
    colvals[17] = 2;
    colvals[18] = 3;
    colvals[19] = 4;
    
    // 4 row of block
    colvals[20] = 0;
    colvals[21] = 1;
    colvals[22] = 2;
    colvals[23] = 3;
    colvals[24] = 4;

    // copy rowptrs, colvals to the device
    SUNMatrix_cuSparse_CopyToDevice(J, NULL, rowptrs, colvals);
    cudaDeviceSynchronize();
    return (0);
}


/* Jacobian evaluation with GPU */
__global__
static void jacobian_kernel(realtype *ydata, realtype *Jdata, abc_data data)
{
    int groupj;
    int GROUPSIZE = 5;
    
    // temporary:
    int nnzper = GROUPSIZE*GROUPSIZE;
    int i;
    double *Tge = data.Tge;
    double z, T;

    
    
    groupj = blockIdx.x*blockDim.x + threadIdx.x; 
    
    T = 1000.0;
    z = 0.0;

    if (groupj < BATCHSIZE)
    {
        i = groupj;
        
        
        // pulled the species data
        double dead_predator = ydata[GROUPSIZE*groupj+0];
        double dead_prey = ydata[GROUPSIZE*groupj+1];
        double ge = ydata[GROUPSIZE*groupj+2];
        double predator = ydata[GROUPSIZE*groupj+3];
        double prey = ydata[GROUPSIZE*groupj+4];
        double mdensity = mh * (1.0*dead_predator + 1.0*dead_prey + 1.0*predator + 1.0*prey);
        double inv_mdensity = 1.0/ mdensity;
        double *exp_growth_prey = data.rs_exp_growth_prey;
        double *rexp_growth_prey= data.drs_exp_growth_prey;
        double *natural_death_predator = data.rs_natural_death_predator;
        double *rnatural_death_predator= data.drs_natural_death_predator;
        double *predation = data.rs_predation;
        double *rpredation= data.drs_predation;
        //
        // Species: dead_predator
        //
        
        
        // dead_predator by dead_predator
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 0*GROUPSIZE + 0] = ZERO;
        
        
        
 	    
        
        
        // dead_predator by dead_prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 0*GROUPSIZE + 1] = ZERO;
        
        
        
 	    
        
        
        // dead_predator by ge
        
        
        Jdata[nnzper*groupj + 0*GROUPSIZE + 2] = predator*rnatural_death_predator[i];
        
        
        
 	    
        
        Jdata[nnzper*groupj+ 0*GROUPSIZE + 2] *= Tge[i];
        
        
        // dead_predator by predator
        
        
        Jdata[nnzper*groupj + 0*GROUPSIZE + 3] = natural_death_predator[i];
        
        
        
 	    
        
        
        // dead_predator by prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 0*GROUPSIZE + 4] = ZERO;
        
        
        
 	    
        
        //
        // Species: dead_prey
        //
        
        
        // dead_prey by dead_predator
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 1*GROUPSIZE + 0] = ZERO;
        
        
        
 	    
        
        
        // dead_prey by dead_prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 1*GROUPSIZE + 1] = ZERO;
        
        
        
 	    
        
        
        // dead_prey by ge
        
        
        Jdata[nnzper*groupj + 1*GROUPSIZE + 2] = predator*prey*rpredation[i];
        
        
        
 	    
        
        Jdata[nnzper*groupj+ 1*GROUPSIZE + 2] *= Tge[i];
        
        
        // dead_prey by predator
        
        
        Jdata[nnzper*groupj + 1*GROUPSIZE + 3] = predation[i]*prey;
        
        
        
 	    
        
        
        // dead_prey by prey
        
        
        Jdata[nnzper*groupj + 1*GROUPSIZE + 4] = predation[i]*predator;
        
        
        
 	    
        
        //
        // Species: ge
        //
        
        
        // ge by dead_predator
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 0] = ZERO;
        
        
        
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 0] *= inv_mdensity;
        
 	    
        
        
        // ge by dead_prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 1] = ZERO;
        
        
        
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 1] *= inv_mdensity;
        
 	    
        
        
        // ge by ge
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 2] = ZERO;
        
        
        
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 2] *= inv_mdensity;
        
 	    
        
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 2] *= Tge[i];
        
        
        // ge by predator
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 3] = ZERO;
        
        
        
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 3] *= inv_mdensity;
        
 	    
        
        
        // ge by prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 4] = ZERO;
        
        
        
        Jdata[nnzper*groupj+ 2*GROUPSIZE + 4] *= inv_mdensity;
        
 	    
        
        //
        // Species: predator
        //
        
        
        // predator by dead_predator
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 3*GROUPSIZE + 0] = ZERO;
        
        
        
 	    
        
        
        // predator by dead_prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 3*GROUPSIZE + 1] = ZERO;
        
        
        
 	    
        
        
        // predator by ge
        
        
        Jdata[nnzper*groupj + 3*GROUPSIZE + 2] = 0.75*predator*prey*rpredation[i] - predator*rnatural_death_predator[i];
        
        
        
 	    
        
        Jdata[nnzper*groupj+ 3*GROUPSIZE + 2] *= Tge[i];
        
        
        // predator by predator
        
        
        Jdata[nnzper*groupj + 3*GROUPSIZE + 3] = -natural_death_predator[i] + 0.75*predation[i]*prey;
        
        
        
 	    
        
        
        // predator by prey
        
        
        Jdata[nnzper*groupj + 3*GROUPSIZE + 4] = 0.75*predation[i]*predator;
        
        
        
 	    
        
        //
        // Species: prey
        //
        
        
        // prey by dead_predator
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 4*GROUPSIZE + 0] = ZERO;
        
        
        
 	    
        
        
        // prey by dead_prey
        
        
        // because the Jacobian is initialized to zeros by default
        Jdata[nnzper*groupj+ 4*GROUPSIZE + 1] = ZERO;
        
        
        
 	    
        
        
        // prey by ge
        
        
        Jdata[nnzper*groupj + 4*GROUPSIZE + 2] = -predator*prey*rpredation[i] + prey*rexp_growth_prey[i];
        
        
        
 	    
        
        Jdata[nnzper*groupj+ 4*GROUPSIZE + 2] *= Tge[i];
        
        
        // prey by predator
        
        
        Jdata[nnzper*groupj + 4*GROUPSIZE + 3] = -predation[i]*prey;
        
        
        
 	    
        
        
        // prey by prey
        
        
        Jdata[nnzper*groupj + 4*GROUPSIZE + 4] = exp_growth_prey[i] - predation[i]*predator;
        
        
        
 	    
        

    }

    /*
  if (groupj < 1){
      for (int i =0; i < 25; i++){
       printf("from %d: Jdata[%d] = %0.5g\n", groupj, i, Jdata[nnzper*groupj+i]);
      }
      printf("\n");
  }
  */
}


/*
 * Jacobian routine. COmpute J(t,y) = df/dy.
 * This is done on the GPU.
 */
static int Jacobian(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
    void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
    abc_data *data = (abc_data*)user_data;

    int nnzper;
    realtype *Jdata, *ydata;
    nnzper = 5* 5;
    ydata = N_VGetDeviceArrayPointer_Cuda(y);
    Jdata = SUNMatrix_cuSparse_Data(J);

    jacobian_kernel<<<GRIDSIZE, BLOCKSIZE>>>(ydata, Jdata, *data);

    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, ">>> ERROR in Jac: cudaGetLastError returned %s\n",
                cudaGetErrorName(cuerr));
        return(-1);
    }

  return(0);

}


// now write tests kit

void test_interpolation_kernel(abc_data data)
{
    int NSYSTEM = 1024;
    // initialize temperature;
    for (int i = 0; i < NSYSTEM; i++)
    {
        data.Ts[i] = (double) 3000.0 * (i+10)/NSYSTEM;
        data.logTs[i] = log(data.Ts[i]);
    }

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int j = 0; j < 1; j++){
    linear_interpolation_kernel<<<GRIDSIZE, BLOCKSIZE>>>(data);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time to generate:  %3.1f ms \n", time);
    
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr,
        ">>> ERROR in interpolation_kernel: cudaGetLastError returned %s\n",
        cudaGetErrorName(cuerr));
    }
}

void initialize_ydata(double *ydata, int NSYSTEM)
{
    int nchem = 5;
    for (int i = 0; i < NSYSTEM; i++)
    {
        // H2I
        ydata[i*nchem]   = 1.0;
        // H2II
        ydata[i*nchem+1] = 1.0;
        // HI
        ydata[i*nchem+2] = 10.0;
        // HII
        ydata[i*nchem+3] = 1.0;
        // H-
        ydata[i*nchem+4] = 1.0;
    }
}


void test_temperature_kernel(abc_data data)
{
    int NSYSTEM = 1024;
    int nchem   = 5;
    int neq = NSYSTEM*nchem;

    N_Vector y = N_VNew_Cuda(neq);
    double *ydata;
    ydata = N_VGetHostArrayPointer_Cuda(y);
    initialize_ydata(ydata, NSYSTEM);
    N_VCopyToDevice_Cuda(y);


    ydata = N_VGetDeviceArrayPointer_Cuda(y);
    temperature_kernel<<<GRIDSIZE,BLOCKSIZE>>>(ydata, data);

    for (int i = 0; i<NSYSTEM; i++){
        printf("temperature[%d] = %0.5g\n", i, data.Ts[i]);
    }

    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr,
        ">>> ERROR in temperature kernel: cudaGetLastError returned %s\n",
        cudaGetErrorName(cuerr));
    }
}



void test_rhs_function(abc_data data)
{
    double t = 1.0;
    int NSYSTEM = 1024;
    int nchem   = 5;
    int neq = NSYSTEM*nchem;
    
    N_Vector y = N_VNew_Cuda(neq);
    double *ydata;
    ydata = N_VGetHostArrayPointer_Cuda(y);
    initialize_ydata(ydata, NSYSTEM);
    N_VCopyToDevice_Cuda(y);


    ydata = N_VGetDeviceArrayPointer_Cuda(y);
    N_Vector ydot = N_VNew_Cuda(neq);

    f(t, y, ydot, &data);
    //f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
}


void test_jacobian_function(abc_data data)
{
    double t = 1.0;
    int NSYSTEM = 1024;
    int nchem   = 5;
    int neq = NSYSTEM*nchem;
    
    N_Vector y = N_VNew_Cuda(neq);
    double *ydata;
    ydata = N_VGetHostArrayPointer_Cuda(y);
    initialize_ydata(ydata, NSYSTEM);
    N_VCopyToDevice_Cuda(y);

    ydata = N_VGetDeviceArrayPointer_Cuda(y);
    N_Vector ydot = N_VNew_Cuda(neq);
    
    // also need to initialize jacobian data space

  /* Create sparse SUNMatrix for use in linear solves */
    SUNMatrix A;
    A = NULL;

  cusparseHandle_t cusp_handle;
  cusparseCreate(&cusp_handle);
  A = SUNMatrix_cuSparse_NewBlockCSR(NSYSTEM, nchem, nchem, nchem*nchem, cusp_handle);

  /* Initialiize the Jacobian with its fixed sparsity pattern */
  JacInit(A);

    Jacobian(t, y, y, A, &data, y, y, y);
    //f(realtype t, N_Vector y, N_Vector ydot, void *user_data)
}

/*
 * Private Helper Function
 * Get and print some final statistics
 */

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *) returnvalue;
    if (*retval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
	      funcname, *retval);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
	    funcname);
    return(1); }

  return(0);
}

static void PrintFinalStats(void *cvode_mem, SUNLinearSolver LS)
{
  long int nst, nfe, nsetups, nje, nni, ncfn, netf, nge;
  size_t cuSpInternalSize, cuSpWorkSize;
  int retval;

  retval = CVodeGetNumSteps(cvode_mem, &nst);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
  retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_retval(&retval, "CVodeGetNumErrTestFails", 1);
  retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

  retval = CVodeGetNumJacEvals(cvode_mem, &nje);
  check_retval(&retval, "CVodeGetNumJacEvals", 1);

  retval = CVodeGetNumGEvals(cvode_mem, &nge);
  check_retval(&retval, "CVodeGetNumGEvals", 1);

  SUNLinSol_cuSolverSp_batchQR_GetDeviceSpace(LS, &cuSpInternalSize, &cuSpWorkSize);

  printf("\nFinal Statistics:\n");
  printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld nje = %ld\n",
	 nst, nfe, nsetups, nje);
  printf("nni = %-6ld ncfn = %-6ld netf = %-6ld    nge = %ld\n \n",
   nni, ncfn, netf, nge);
  printf("cuSolverSp numerical factorization workspace size (in bytes) = %ld\n", cuSpWorkSize);
  printf("cuSolverSp internal Q, R buffer size (in bytes) = %ld\n", cuSpInternalSize);
}


int run_solver(int argc, char *argv[])
{
    realtype reltol, t, tout;
    realtype *ydata, *abstol_data;
    N_Vector y, abstol;
    SUNMatrix A;
    SUNLinearSolver LS;
    void *cvode_mem;
    int retval, iout;
    int neq, ngroups, groupj;
    abc_data data = abc_setup_data(NULL, NULL);
    abc_read_cooling_tables( &data);
    abc_read_rate_tables( &data);

    cusparseHandle_t cusp_handle;
    cusolverSpHandle_t cusol_handle;

    y = abstol = NULL;
    A = NULL;
    LS = NULL;
    cvode_mem = NULL;

    /* Parse command line arguments */
    ngroups = BATCHSIZE;
    int GROUPSIZE = 5;
    neq = ngroups* GROUPSIZE;

    int NSYSTEM = BATCHSIZE;
    reltol = 1.0e-5;
    /* Initialize cuSOLVER and cuSPARSE handles */
    cusparseCreate(&cusp_handle);
    cusolverSpCreate(&cusol_handle);

    /* Create CUDA vector of length neq for I.C. and abstol */
    y = N_VNew_Cuda(neq);
    if (check_retval((void *)y, "N_VNew_Cuda", 0)) return(1);
    abstol = N_VNew_Cuda(neq);
    if (check_retval((void *)abstol, "N_VNew_Cuda", 0)) return(1);
    
    ydata = N_VGetHostArrayPointer_Cuda(y);
    abstol_data = N_VGetHostArrayPointer_Cuda(abstol);

    /* Initialize */
    initialize_ydata(ydata, NSYSTEM);
    for (int i = 0; i < neq; i++){
        abstol_data[i] = 1.0e-5;
    }
    N_VCopyToDevice_Cuda(y);
    N_VCopyToDevice_Cuda(abstol);

    /* Call CVodeCreate to create the solver memory and specify the
    * Backward Differentiation Formula */
    cvode_mem = CVodeCreate(CV_BDF);
    if (check_retval((void *)cvode_mem, "CVodeCreate", 0)) return(1);

    /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in y'=f(t,y), the inital time T0, and
    * the initial dependent variable vector y. */
    retval = CVodeInit(cvode_mem, f, T0, y);
    if (check_retval(&retval, "CVodeInit", 1)) return(1);

    /* Call CVodeSetUserData to attach the user data structure */
    retval = CVodeSetUserData(cvode_mem, &data);
    if (check_retval(&retval, "CVodeSetUserData", 1)) return(1);

    /* Call CVodeSVtolerances to specify the scalar relative tolerance
    * and vector absolute tolerances */
    retval = CVodeSVtolerances(cvode_mem, reltol, abstol);
    if (check_retval(&retval, "CVodeSVtolerances", 1)) return(1);

    /* Create sparse SUNMatrix for use in linear solves */
    A = SUNMatrix_cuSparse_NewBlockCSR(ngroups, GROUPSIZE, GROUPSIZE, GROUPSIZE*GROUPSIZE, cusp_handle);
    if(check_retval((void *)A, "SUNMatrix_cuSparse_NewBlockCSR", 0)) return(1);

    /* Set the sparsity pattern to be fixed so that the row pointers
    * and column indicies are not zeroed out by SUNMatZero */
    retval = SUNMatrix_cuSparse_SetFixedPattern(A, 1);

    /* Initialiize the Jacobian with its fixed sparsity pattern */
    blockJacInit(A);

    /* Create the SUNLinearSolver object for use by CVode */
    LS = SUNLinSol_cuSolverSp_batchQR(y, A, cusol_handle);
    if(check_retval((void *)LS, "SUNLinSol_cuSolverSp_batchQR", 0)) return(1);

    /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if(check_retval(&retval, "CVodeSetLinearSolver", 1)) return(1);

    /* Set the user-supplied Jacobian routine Jac */
    retval = CVodeSetJacFn(cvode_mem, Jacobian);
    if(check_retval(&retval, "CVodeSetJacFn", 1)) return(1);

    /* In loop, call CVode, print results, and test for error.
     Break out of loop when NOUT preset output times have been reached.  */
    printf(" \nGroup of independent 3-species kinetics problems\n\n");
    printf("number of groups = %d\n\n", ngroups);
    
    CVodeSetMaxNumSteps(cvode_mem, 1000);

    iout = 0;  tout = 1.0e0;
    while(1) {
        retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
        N_VCopyFromDevice_Cuda(y);
        for (groupj = 0; groupj < ngroups; groupj += 512) {
            printf("group %d: @ t = %0.5g\n", groupj, tout);
            for (int i = 0; i < GROUPSIZE; i++){
                printf("ydata[%d] = %0.5g\n", GROUPSIZE*groupj+i, ydata[GROUPSIZE*groupj+i]);
            }
            printf("\n");
        }
        
        if (check_retval(&retval, "CVode", 1)) break;
        if (retval == CV_SUCCESS) {
            iout++;
            tout *= TMULT;
        }

        if (iout == NOUT) break;
    }

  /* Print some final statistics */
  PrintFinalStats(cvode_mem, LS);

  /* Free y and abstol vectors */
  N_VDestroy(y);
  N_VDestroy(abstol);

  /* Free integrator memory */
  CVodeFree(&cvode_mem);

  /* Free the linear solver memory */
  SUNLinSolFree(LS);

  /* Free the matrix memory */
  SUNMatDestroy(A);

  /* Destroy the cuSOLVER and cuSPARSE handles */
  cusparseDestroy(cusp_handle);
  cusolverSpDestroy(cusol_handle);

  return(0);
}



int main()
{
    // read the rate data 

    cudaDeviceSynchronize();
    abc_data data = abc_setup_data(NULL, NULL);
    abc_read_cooling_tables( &data);
    abc_read_rate_tables( &data);

    // printf("rk01 = %0.5g\n", data.r_k22[213]);
    // printf("h2mheat = %0.5g\n", data.c_h2formation_h2mheat[1020]);  
    
    /*
    // test interpolation first
    test_interpolation_kernel(data);

    // test temperature kerenel
    test_temperature_kernel(data);

    // test the rhs function
    test_rhs_function(data);

    // initialize initial conditions
    // create a y_vec that holds NSYSTEM  * nchem elements
    test_jacobian_function(data);
*/
    // initialize yvec to see if we can have it print out accurate ydot

    run_solver(NULL, NULL);
    cudaDeviceSynchronize();
}


int check_vector(N_Vector X, N_Vector Y, realtype tol)
{
  int failure = 0;
  sunindextype i, local_length, maxloc;
  realtype *Xdata, *Ydata, maxerr;

  cudaDeviceSynchronize();

  N_VCopyFromDevice_Cuda(X);
  N_VCopyFromDevice_Cuda(Y);

  Xdata = N_VGetHostArrayPointer_Cuda(X);
  Ydata = N_VGetHostArrayPointer_Cuda(Y);
  local_length = N_VGetLength(X);

  /* check vector data */
  for(i=0; i < local_length; i++)
    failure += FNEQ(Xdata[i], Ydata[i], tol);

  if (failure > ZERO) {
    maxerr = ZERO;
    maxloc = -1;
    for(i=0; i < local_length; i++) {
      if (SUNRabs(Xdata[i]-Ydata[i]) >  maxerr) {
        maxerr = SUNRabs(Xdata[i]-Ydata[i]);
        maxloc = i;
      }
    }
    printf("check err failure: maxerr = %g at loc %li (tol = %g)\n",
	   maxerr, (long int) maxloc, tol);
    return(1);
  }
  else
    return(0);
}
void sync_device()
{
  cudaDeviceSynchronize();
}
