
#include <stdio.h>
#include <stdlib.h>

#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>
#include <sunmatrix/sunmatrix_cusparse.h>
#include <nvector/nvector_cuda.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include "test_sunlinsol.h"

#define nchem 5
#define nnz 25
#define batchsize 4
#define gridsize 2
#define blocksize batchsize/gridsize

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
static int blockJacInit(SUNMatrix J)
{
    
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
    
    int rowptrs[batchsize*nchem+1];
    int colvals[batchsize*nnz  ];

    SUNMatZero(J);

    for (int r = 0; r < batchsize*nchem+1; r++)
    {
        rowptrs[r] = r*nchem;
        printf("rowptrs[%d] = %d\n", r, rowptrs[r]);
    }

    int bIdx;
    for (int c = 0; c < batchsize*nnz; c++)
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

__global__
static void jacobian_kernel(realtype *Jdata)
{
    int groupj = blockIdx.x*blockDim.x + threadIdx.x; 
    if (groupj < batchsize)
    {
        for (int i = 0; i < nnz; i++)
        {
            if (i%2 == 0)
            {
                Jdata[groupj*nnz+i] = 1;
            }else{
                Jdata[groupj*nnz+i] = 0;
            }
        }
    }
}

static int Jacobian(SUNMatrix J)
{
    realtype *Jdata;
    Jdata = SUNMatrix_cuSparse_Data(J);
    jacobian_kernel<<<gridsize, blocksize>>>(Jdata);
    
    cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, ">>> ERROR in Jac: cudaGetLastError returned %s\n",
                cudaGetErrorName(cuerr));
        return(-1);
    }

  return(0);
}


int main()
{

  SUNLinearSolver LS;                 /* linear solver object          */
  cusparseStatus_t cusp_status;
  cusolverStatus_t cusol_status;
  cusparseHandle_t cusp_handle;
  cusolverSpHandle_t cusol_handle;
  /* Initialize cuSPARSE */
  cusp_status = cusparseCreate(&cusp_handle);
  if (cusp_status != CUSPARSE_STATUS_SUCCESS) {
    printf("ERROR: could not create cuSPARSE handle\n");
    return(-1);
  }

  /* Initialize cuSOLVER */
  cusol_status = cusolverSpCreate(&cusol_handle);
  if (cusol_status != CUSOLVER_STATUS_SUCCESS) {
    printf("ERROR: could not create cuSOLVER handle\n");
    return(-1);
  }


  int N = nchem*batchsize;
  N_Vector d_x, d_b;
  d_x = N_VNew_Cuda(N);
  d_b = N_VNew_Cuda(N);

  realtype *xdata, *bdata;
  xdata = N_VGetHostArrayPointer_Cuda(d_x);
  bdata = N_VGetHostArrayPointer_Cuda(d_b);
  for (int i=0; i<N; i++)
  {
      xdata[i] = i % 2;
      bdata[i] = 1.0 + i%2;
  }
  N_VCopyToDevice_Cuda(d_x);
  N_VCopyToDevice_Cuda(d_b);


  /* Create the device matrix */
  SUNMatrix J;
  //J = SUNMatrix_cuSparse_NewCSR(N, N, N*nchem, cusp_handle);
  // JacInit(J);
  // Jacobian(J);
  // Instead of using the CSR, we use BCSR
  J = SUNMatrix_cuSparse_NewBlockCSR(batchsize, nchem, nchem, nchem*nchem, cusp_handle);
  if(check_retval((void *)J, "SUNMatrix_cuSparse_NewBlockCSR", 0)) return(1);
  SUNMatrix_cuSparse_SetFixedPattern(J, 1);
  JacInit(J);
  Jacobian(J);


  /*
  // get the rows and col in the Sparse matrix
  int M = SUNMatrix_cuSparse_Rows(J);
  int N = SUNMatrix_cuPsarse_Columns(J);
  int nz = SUNMatrix_cuSparse_NNZ(J);
*/


  // create an empty host array to store it and print it
  SUNMatrix Jhost;
  Jhost = SUNSparseMatrix(N,N, N*nchem, CSR_MAT);

    SUNMatrix_cuSparse_CopyFromDevice(J,
            SUNSparseMatrix_Data(Jhost),
            SUNSparseMatrix_IndexPointers(Jhost),
            SUNSparseMatrix_IndexValues(Jhost));

    printf("\nJhost =\n");
    SUNSparseMatrix_Print(Jhost,stdout);

    // create a linear solver object
    // from J


    LS = SUNLinSol_cuSolverSp_batchQR(d_x, J, cusol_handle);


  if (LS == NULL) {
    printf("FAIL: SUNLinSol_cuSolverSp_batchQR returned NULL\n");
    return(1);
  }

  // need to first initialize sunlinsol
  SUNLinSolInitialize(LS);
  sync_device();
  // first we need a linsolsetup;
  int failure;
  failure = SUNLinSolSetup(LS, J);
  sync_device();

  N_Vector tmp;
  tmp = N_VClone(d_x);

  // perform solve
  failure = SUNLinSolSolve(LS, J, d_x, d_b, 0.001);
  sync_device();


    N_VCopyFromDevice_Cuda(d_x); /* copy solution from device */
    printf("x (computed)\n");
    N_VPrint_Cuda(d_x);

    N_VCopyFromDevice_Cuda(d_b);
    printf("\nb = Ax (reference)\n");
    N_VPrint_Cuda(d_b);

}


/* ----------------------------------------------------------------------
 * Implementation-specific 'check' routines
 * --------------------------------------------------------------------*/
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
