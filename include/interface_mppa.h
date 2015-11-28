//-------------------------------------------------------------------------------
// Copyright (c) 2015, Márcio B. Castro <marcio.castro@ufsc.br>
//					   
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//-------------------------------------------------------------------------------

#ifndef __INTERFACE_MPPA_H
#define __INTERFACE_MPPA_H

#include <mppaipc.h>
#include <inttypes.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*
 * GLOBAL CONSTANTS
 */

#define BARRIER_SYNC_MASTER "/mppa/sync/128:1"
#define BARRIER_SYNC_SLAVE "/mppa/sync/[0..15]:2"

#define TRUE 1
#define FALSE 0

#define IO_NODE_RANK 128
#define MAX_CLUSTERS 16
#define MAX_THREADS_PER_CLUSTER 16
#define MPPA_FREQUENCY 400

/*
 * INTERNAL STRUCTURES
 */

typedef enum {
  BARRIER_MASTER,
  BARRIER_SLAVE
} barrier_mode_t;

typedef struct {
  int file_descriptor;
  mppa_aiocb_t aiocb;
} portal_t;

typedef struct {
  int sync_fd_master;
  int sync_fd_slave;
  barrier_mode_t mode;
  int nb_clusters;
} barrier_t;

/*
 * FUNCTIONS
 */

void set_path_name(char *path, char *template_path, int rx, int tag);

portal_t *mppa_create_read_portal (char *path, void* buffer, unsigned long buffer_size, int trigger, void (*function)(mppa_sigval_t));
portal_t *mppa_create_write_portal (char *path, void* buffer, unsigned long buffer_size, int receiver_rank);
void mppa_write_portal (portal_t *portal, void *buffer, int buffer_size, int offset);
void mppa_async_write_portal (portal_t *portal, void *buffer, int buffer_size, int offset);
void mppa_async_write_wait_portal(portal_t *portal);
void mppa_async_read_wait_portal(portal_t *portal);
void mppa_close_portal (portal_t *portal);

barrier_t *mppa_create_master_barrier (char *path_master, char *path_slave, int clusters);
barrier_t *mppa_create_slave_barrier (char *path_master, char *path_slave);
void mppa_barrier_wait (barrier_t *barrier);
void mppa_close_barrier (barrier_t *barrier);

void mppa_init_time(void);
inline uint64_t mppa_get_time(void);
inline uint64_t mppa_diff_time(uint64_t t1, uint64_t t2);

#endif // __INTERFACE_MPPA_H
