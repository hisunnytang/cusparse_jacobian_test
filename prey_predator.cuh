#include "hdf5.h"
#include "hdf5_hl.h"
#include <stdio.h>
#include <cvode/cvode.h>                  /* prototypes for CVODE fcts., consts.          */
#include <nvector/nvector_cuda.h>         /* access to cuda N_Vector                      */
#include <sunmatrix/sunmatrix_cusparse.h>             /* access to cusparse SUNMatrix                  */
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>   /* acess to cuSolverSp batch QR SUNLinearSolver */
#include <sundials/sundials_types.h>     /* defs. of realtype, int              */

#define BATCHSIZE 1
#define ZERO    RCONST(0.0)
#define kb      RCONST(1.3806504e-16)
#define mh      RCONST(1.67e-24)
#define gamma   RCONST(5.0/3.0)
#define _gamma_m1 RCONST(1.0/ (gamma-1.0) )

#define GRIDSIZE 1
#define BLOCKSIZE 1

#define T0 RCONST(0.0)
#define T1 RCONST(1e10)
#define TMULT RCONST(10.0)
#define NOUT 12
