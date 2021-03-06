# -*- mode: makefile -*-
# -----------------------------------------------------------------
# Programmer: Slaven Peles, Cody Balos @ LLNL
# -----------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2020, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# -----------------------------------------------------------------
# Makefile for  CUDA examples
#
# This file is generated from a template using various variables
# set at configuration time. It can be used as a template for
# other user Makefiles.
# -----------------------------------------------------------------

SHELL = sh

prefix       = /home/kwoksun2/dengo-merge/cvode-5.3.0/instdir
exec_prefix  = /home/kwoksun2/dengo-merge/cvode-5.3.0/instdir
includedir   = /home/kwoksun2/dengo-merge/cvode-5.3.0/instdir/include
libdir       = /home/kwoksun2/dengo-merge/cvode-5.3.0/instdir/lib64

CC          = /usr/bin/cc
CFLAGS      = -fPIC
CXX         = /usr/bin/c++
CXXFLAGS    = -fPIC
NVCC        = /usr/local/cuda/bin/nvcc
NVCCFLAGS   = -ccbin=${CXX} -std=c++11  -arch sm_30
LD          = ${NVCC}
LDFLAGS     =  ${NVCCFLAGS} -Xcompiler \"-Wl,-rpath,${libdir}\"
LIBS        =  -lm /usr/lib64/librt.so -lcusolver -lcusparse -lhdf5 -lhdf5_hl

TMP_INCS = ${includedir}
INCLUDES = $(addprefix -I, ${TMP_INCS})

TMP_LIBDIRS  = ${libdir}
LIBDIRS      = $(addprefix -L, ${TMP_LIBDIRS})

TMP_SUNDIALSLIBS = sundials_cvode  sundials_nvecserial sundials_nveccuda sundials_sunmatrixdense sundials_sunmatrixsparse sundials_sunmatrixcusparse sundials_sunlinsolcusolversp
SUNDIALSLIBS     = $(addprefix -l, ${TMP_SUNDIALSLIBS})
LIBRARIES = ${SUNDIALSLIBS} ${LIBS}

EXAMPLES = prey_predator test_prey_predator test_jacobian  test_sunlinsol_cusolversp_batchqr
EXAMPLES_DEPENDENCIES =  test_sunlinsol sundials_nvector

OBJECTS = ${EXAMPLES:=.o}
OBJECTS_DEPENDENCIES = ${EXAMPLES_DEPENDENCIES:=.o}

# -----------------------------------------------------------------------------------------

.SUFFIXES : .o .cu

.c.o :
	${CC} ${CFLAGS} ${INCLUDES} -c $<

.cu.o :
	${NVCC} ${NVCCFLAGS} ${INCLUDES} -c $<

# -----------------------------------------------------------------------------------------

all: ${OBJECTS}
	@for i in ${EXAMPLES} ; do \
	  echo "${NVCC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${INCLUDES} ${LIBDIRS} ${LIBRARIES} ${LDFLAGS}"; \
	  ${NVCC} -o $${i} $${i}.o ${OBJECTS_DEPENDENCIES} ${INCLUDES} ${LIBDIRS} ${LIBRARIES} ${LDFLAGS}; \
	done

${OBJECTS}: ${OBJECTS_DEPENDENCIES}

clean:
	rm -f ${OBJECTS_DEPENDENCIES}
	rm -f ${OBJECTS}
	rm -f ${EXAMPLES}

# -----------------------------------------------------------------------------------------

