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

abc_data abc_setup_data(int *NumberOfFields, char ***FieldNames);
void abc_read_rate_tables(abc_data *data);
void abc_read_cooling_tables(abc_data *data);
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int JacInit(SUNMatrix J);

static int Jacobian(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

void test_interpolation_kernel(abc_data data);
void test_temperature_kernel(abc_data data);
void test_rhs_function(abc_data data);

void test_jacobian_function(abc_data data);
static int check_retval(void *returnvalue, const char *funcname, int opt);
static void PrintFinalStats(void *cvode_mem, SUNLinearSolver LS);
int run_solver(int argc, char *argv[]);

