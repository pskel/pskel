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

#include <mppa/osconfig.h>
#include "interface_mppa.h"

void set_path_name(char *path, char *template_path, int rx, int tag) {
  sprintf(path, template_path, rx, tag);
}

/**************************************
 * PORTAL COMMUNICATION
 **************************************/

portal_t *mppa_create_read_portal (char *path, void* buffer, unsigned long buffer_size, int trigger, void (*function)(mppa_sigval_t)) {
  portal_t *ret = (portal_t*) malloc (sizeof(portal_t));
  int status;
  ret->file_descriptor = mppa_open(path, O_RDONLY);
  assert(ret->file_descriptor != -1);

  mppa_aiocb_ctor(&ret->aiocb, ret->file_descriptor, buffer, buffer_size);

  if (trigger > -1) {
    mppa_aiocb_set_trigger(&ret->aiocb, trigger);
  }

  // Attention: we can't use callbacks with trigger/mppa_aio_wait (bug?)
  if (function)
    mppa_aiocb_set_callback(&ret->aiocb, function);

  status = mppa_aio_read(&ret->aiocb);
  assert(status == 0);

  return ret;
}

portal_t *mppa_create_write_portal (char *path, void* buffer, unsigned long buffer_size, int receiver_rank) {
  portal_t *ret = (portal_t*) malloc (sizeof(portal_t));
  ret->file_descriptor = mppa_open(path, O_WRONLY);
  assert(ret->file_descriptor != -1);

  // Tell mppa_io_write to wait for resources when sending a asynchronous message
  assert(mppa_ioctl(ret->file_descriptor, MPPA_TX_WAIT_RESOURCE_ON) == 0);

  // Select the DMA interface according to the receiver's rank.
  // This is only possible on the IO-node!
  if (__k1_get_cluster_id() == 128)
    assert(mppa_ioctl(ret->file_descriptor, MPPA_TX_SET_IFACE, receiver_rank % 4) == 0);

  // We need to initialize an aiocb for asynchronous writes.
  // It seems that the buffer and buffer size parameters are not important here,
  // because we're going to specify them with mppa_aiocb_set_pwrite()
  // before calling mppa_aio_write()
  assert(mppa_aiocb_ctor(&ret->aiocb, ret->file_descriptor, buffer, buffer_size) == &ret->aiocb);

  return ret;
}

void mppa_async_read_wait_portal(portal_t *portal) {
  int status;
  status = mppa_aio_rearm(&portal->aiocb);
  assert(status != -1);
}

void mppa_async_write_wait_portal(portal_t *portal) {
  int status;
  while(mppa_aio_error(&portal->aiocb) == EINPROGRESS);
  status = mppa_aio_return(&portal->aiocb);
  assert(status != -1);
}

void mppa_close_portal (portal_t *portal) {
  assert(mppa_close(portal->file_descriptor) != -1);
  free (portal);
}

void mppa_write_portal (portal_t *portal, void *buffer, int buffer_size, int offset) {
  int status;
  status = mppa_pwrite(portal->file_descriptor, buffer, buffer_size, offset);
  assert(status == buffer_size);
}

void mppa_async_write_portal (portal_t *portal, void *buffer, int buffer_size, int offset) {
  int status;
  mppa_aiocb_set_pwrite(&portal->aiocb, buffer, buffer_size, offset);
  status = mppa_aio_write(&portal->aiocb);
  assert(status == 0);
}

void mppa_async_write_stride_portal (portal_t *portal, void *buffer, int buffer_size, int ecount, int sstride, int tstride, int offset) {
  int status;

  // mppa_pwrites(portal->file_descriptor, buffer, buffer_size, ecount, sstride, tstride, offset);
  mppa_aiocb_set_pwrite(&portal->aiocb, buffer, buffer_size, offset);
  mppa_aiocb_set_strides(&portal->aiocb, ecount, sstride, tstride);
  status = mppa_aio_write(&portal->aiocb);
  assert(status == 0);
}

/**************************************
 * BARRIER
 **************************************/

barrier_t *mppa_create_master_barrier (char *path_master, char *path_slave, int clusters) {
  int status, i;
  int ranks[clusters];
  long long match;

  barrier_t *ret = (barrier_t*) malloc (sizeof (barrier_t));

  ret->sync_fd_master = mppa_open(path_master, O_RDONLY);
  assert(ret->sync_fd_master != -1);

  ret->sync_fd_slave = mppa_open(path_slave, O_WRONLY);
  assert(ret->sync_fd_slave != -1);

  // set all bits to 1 except the less significative "cluster" bits (those ones are set to 0).
  // when the IO receives messagens from the clusters, they will set their correspoding bit to 1.
  // the mppa_read() on the IO will return when match = 11111...1111
  match = (long long) - (1 << clusters);
  status = mppa_ioctl(ret->sync_fd_master, MPPA_RX_SET_MATCH, match);
  assert(status == 0);

  for (i = 0; i < clusters; i++)
    ranks[i] = i;

  // configure the sync connector to receive message from "ranks"
  status = mppa_ioctl(ret->sync_fd_slave, MPPA_TX_SET_RX_RANKS, clusters, ranks);
  assert(status == 0);

  ret->mode = BARRIER_MASTER;

  return ret;
}

barrier_t *mppa_create_slave_barrier (char *path_master, char *path_slave) {
  int status;

  barrier_t *ret = (barrier_t*) malloc (sizeof (barrier_t));

  ret->sync_fd_master = mppa_open(path_master, O_WRONLY);
  assert(ret->sync_fd_master != -1);

  ret->sync_fd_slave = mppa_open(path_slave, O_RDONLY);
  assert(ret->sync_fd_slave != -1);

  // set match to 0000...000.
  // the IO will send a massage containing 1111...11111, so it will allow mppa_read() to return
  status = mppa_ioctl(ret->sync_fd_slave, MPPA_RX_SET_MATCH, (long long) 0);
  assert(status == 0);

  ret->mode = BARRIER_SLAVE;

  return ret;
}

void mppa_barrier_wait(barrier_t *barrier) {
  int status;
  long long dummy;

  if(barrier->mode == BARRIER_MASTER) {
    dummy = -1;
    long long match;

    // the IO waits for a message from each of the clusters involved in the barrier
    // each cluster will set its correspoding bit on the IO (variable match) to 1
    // when match = 11111...1111 the following mppa_read() returns
    status = mppa_read(barrier->sync_fd_master, &match, sizeof(match));
    assert(status == sizeof(match));

    // the IO sends a message (dummy) containing 1111...1111 to all slaves involved in the barrier
    // this will unblock their mppa_read()
    status = mppa_write(barrier->sync_fd_slave, &dummy, sizeof(long long));
    assert(status == sizeof(long long));
  }
  else {
    dummy = 0;
    long long mask;

    // the cluster sets its corresponding bit to 1
    mask = 0;
    mask |= 1 << __k1_get_cluster_id();

    // the cluster sends the mask to the IO
    status = mppa_write(barrier->sync_fd_master, &mask, sizeof(mask));
    assert(status == sizeof(mask));

    // the cluster waits for a message containing 1111...111 from the IO to unblock
    status = mppa_read(barrier->sync_fd_slave, &dummy, sizeof(long long));
    assert(status == sizeof(long long));
  }
}

void mppa_close_barrier (barrier_t *barrier) {
  assert(mppa_close(barrier->sync_fd_master) != -1);
  assert(mppa_close(barrier->sync_fd_slave) != -1);
  free(barrier);
}

/**************************************
 * TIME
 **************************************/

//static uint64_t residual_error = 0;

struct timeval mppa_master_get_time(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  //std::cout << tv.tv_sec << "|" << tv.tv_usec << "|" << tv.tv_sec+((double)(tv.tv_usec/1000000)) << std::endl;
  return tv; //tv.tv_sec+((double)(tv.tv_usec/1000000));
  // return tv;
  // get_time_of_day();
  // residual_error = t2 - t1;

}

struct timespec mppa_slave_get_time(void) {
  struct timespec request;

  clock_gettime(CLOCK_REALTIME, &request);
  return request;
  // uint64_t t1, t2;
  // t1 = clock_gettime();
  // t2 = clock_gettime();
  // residual_error = t2 - t1;
}

double mppa_master_diff_time(struct timeval begin, struct timeval end) {
    std::cout << begin.tv_sec << ", " << begin.tv_usec << std::endl;
    std::cout << end.tv_sec << ", " << end.tv_usec << std::endl;
    std::cout << (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) << std::endl;
  return (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
}
double mppa_slave_diff_time(struct timespec begin, struct timespec end) {
  return (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec)/BILLION);
}
