#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <mppaipc.h>
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "hr_time.h"

#define ARGC_SLAVE 11
#define DEBUG
#define TIMER

using namespace std;


struct Arguments
{
	int externCircle;
	int internCircle;
	float power;
};


int CalcSize(int level){
	if (level == 1) {
    		return 3;
  	}
  	if (level >= 1) {
    		return CalcSize(level-1) + 2;
  	}
	return 0;
}



int main(int argc, char **argv){ 
	int i;
	int cluster_id;
	int pid;
	
	int nb_clusters = 1;
	#ifdef TIMER
	hr_timer_t totTimerMaster;
	//timer
  	hrt_start(&totTimerMaster);
  	#endif
	char **argv_slave = (char**) malloc(sizeof (char*) * ARGC_SLAVE);
	for (i = 0; i < ARGC_SLAVE - 1; i++)
	  argv_slave[i] = (char*) malloc (sizeof (char) * 11);
	argv_slave[10] = NULL;

	  
	// Spawn slave processes
	for (cluster_id = 0; cluster_id < nb_clusters; cluster_id++) {
	    pid = mppa_spawn(cluster_id, NULL, "slave", (const char **)argv_slave, NULL);
	    assert(pid >= 0);
	}
	for (i = 0; i < ARGC_SLAVE; i++)
		free(argv_slave[i]);
	free(argv_slave);

	#ifdef DEBUG
		cout << "Clusters spawned!" << endl;
	#endif

	for (pid = 0; pid < nb_clusters; pid++) {
    		mppa_waitpid(pid, NULL, 0);
	}
	#ifdef TIMER
  	hrt_stop(&totTimerMaster);
  	cout<<"Master Time: " << hrt_elapsed_time(&totTimerMaster) << endl;
  	#endif



	mppa_exit(0);
}
