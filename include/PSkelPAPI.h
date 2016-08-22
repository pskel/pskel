//*******************************************************************************************
// PSKEL PAPI
//*******************************************************************************************

#define NUM_EVENTS_RAPL 6
#define NUM_EVENTS_NVML 1

#define NUM_GROUPS_GPU 1
#define MAX_EVENTS_GPU 7

#ifdef QUADRO
	#define NUM_GROUPS_CPU 6
	#define MAX_EVENTS_CPU 5
#else
	#define NUM_GROUPS_CPU 7
	#define MAX_EVENTS_CPU 6
#endif

#define NUM_COMPONENTS 4

#ifdef PSKEL_PAPI
	#include <papi.h>
#endif
#include <iostream>

using namespace std;

namespace PSkel{
	
namespace PSkelPAPI{
	const int CPU = 0;
	const int GPU = 1;
	const int RAPL = 2;
	const int NVML = 3;
	
	bool papi_init = false; //Checks if PAPI_library_init has been called  
	int retval;
	//typedef enum:short{CPU=0,GPU,RAPL,NVML}Component;
	char const *Components[NUM_COMPONENTS] = {"CPU","GPU","RAPL","NVML"};
	int EventSetRAPL = PAPI_NULL;
	int EventSetCPU[NUM_GROUPS_CPU];
	int EventSetGPU = PAPI_NULL; //{/*,PAPI_NULL,PAPI_NULL,PAPI_NULL,PAPI_NULL,PAPI_NULL,PAPI_NULL*/};
	int EventSetNVML = PAPI_NULL;
	
	long long values_rapl[NUM_EVENTS_RAPL] = {0};
	long long values_gpu[MAX_EVENTS_GPU] = {0};
	long long values_nvml[NUM_EVENTS_NVML] = {0};
	long long values_cpu[NUM_GROUPS_CPU][MAX_EVENTS_CPU] = {0};
	
	//PAPI Event Names
    	char const *EventNameRAPL[NUM_EVENTS_RAPL] = {"rapl:::PACKAGE_ENERGY:PACKAGE0","rapl:::PP0_ENERGY:PACKAGE0","rapl:::DRAM_ENERGY:PACKAGE0",
						      "rapl:::PACKAGE_ENERGY:PACKAGE1","rapl:::PP0_ENERGY:PACKAGE1","rapl:::DRAM_ENERGY:PACKAGE1"};

	//Events for Quadro 2000
	#ifdef QUADRO
	char const *EventNameCPU[NUM_GROUPS_CPU][MAX_EVENTS_CPU] = {
								{"PAPI_TOT_INS", "PAPI_FP_INS", "PAPI_VEC_SP", "PAPI_LD_INS", "PAPI_SR_INS"},
								{"PAPI_TOT_INS", "PAPI_BR_INS", "PAPI_BR_CN",  "PAPI_BR_TKN", "PAPI_BR_MSP"},
								{"PAPI_L2_DCA",  "PAPI_L2_DCM", "PAPI_L2_LDM", "PAPI_L2_STM", ""},
								{"PAPI_L2_DCR",  "PAPI_L2_DCW", "PAPI_L3_DCR", "PAPI_L3_DCW", ""},
								{"PAPI_L3_TCA",  "PAPI_L3_TCM", "PAPI_L3_LDM", "",            ""},
								{"PAPI_TOT_CYC", "PAPI_TOT_INS","PAPI_RES_STL","PAPI_REF_CYC","PAPI_FP_OPS"}
							};
	#elif XEON
	char const *EventNameCPU[NUM_GROUPS_CPU][MAX_EVENTS_CPU] = {
								{"PAPI_TOT_INS", "PAPI_VEC_SP", "PAPI_LD_INS", "PAPI_SR_INS", "",            ""},
								{"PAPI_TOT_INS", "PAPI_FP_INS", "PAPI_FDV_INS","",            "",            ""},
								{"PAPI_TOT_INS", "PAPI_BR_INS", "PAPI_BR_CN",  "PAPI_BR_TKN", "PAPI_BR_MSP", "PAPI_BR_PRC"},
								{"PAPI_L2_DCA",  "PAPI_L2_DCM", "PAPI_L2_TCA", "PAPI_L2_TCM", "",            ""},
								{"PAPI_L2_DCR",  "PAPI_L2_DCW", "PAPI_L2_TCR", "PAPI_L2_TCW", "",             ""},
								{"PAPI_L3_TCA",  "PAPI_L3_TCM", "PAPI_L3_DCR", "PAPI_L3_DCW", "PAPI_L3_TCR", "PAPI_L3_TCW"},
								{"PAPI_TOT_CYC", "PAPI_TOT_INS","PAPI_STL_ICY","PAPI_REF_CYC","",             ""}
							};
	#else
	char const *EventNameCPU[NUM_GROUPS_CPU][MAX_EVENTS_CPU] = {
								{"PAPI_TOT_INS", "PAPI_VEC_SP", "",            "",            "",            ""},
								{"PAPI_TOT_INS", "PAPI_FP_INS", "PAPI_FDV_INS","",            "",            ""},
								{"PAPI_TOT_INS", "PAPI_BR_INS", "PAPI_BR_CN",  "PAPI_BR_TKN", "PAPI_BR_MSP", "PAPI_BR_PRC"},
								{"PAPI_L2_DCA",  "PAPI_L2_DCM", "PAPI_L2_TCA", "PAPI_L2_TCM", "",            ""},
								{"PAPI_L2_DCR",  "PAPI_L2_DCW", "PAPI_L2_TCR", "PAPI_L2_TCW", "",             ""},
								{"PAPI_L3_TCA",  "PAPI_L3_TCM", "PAPI_L3_DCR", "PAPI_L3_DCW", "PAPI_L3_TCR", "PAPI_L3_TCW"},
								{"PAPI_TOT_CYC", "PAPI_TOT_INS","PAPI_STL_ICY","PAPI_REF_CYC","",             ""}
							};
	#endif
	
	char const *EventNameGPU[MAX_EVENTS_GPU] =	{"cuda:::device:0:inst_executed","cuda:::device:0:branch", "cuda:::device:0:divergent_branch","cuda:::device:0:l1_global_load_hit", "cuda:::device:0:l1_global_load_miss","cuda:::device:0:elapsed_cycles_sm","cuda:::device:0:active_cycles"};
								
								/*{"","","","","",""}
								 ""
								* cuda:::device:0:warps_launched
								* cuda:::device:0:threads_launched
								* cuda:::device:0:active_warps
								* cuda:::device:0:l1_local_load_hit 
								* cuda:::device:0:l1_local_load_miss		 
								* cuda:::device:0:l1_local_store_hit 
								* cuda:::device:0:l1_local_store_miss
								* cuda:::device:0:thread_inst_executed_1
								* cuda:::device:0:thread_inst_executed_2
								* cuda:::device:0:thread_inst_executed_3
								* cuda:::device:0:thread_inst_executed_4
								* cuda:::device:0:l1_shared_bank_conflict
								* cuda:::device:0:sm_cta_launched
								*/
							
	
	char const *EventNameNVML[NUM_EVENTS_NVML] = {"nvml:::Tesla_K20m:power"};

	//Holds PAPI CODES
	int events_rapl[NUM_EVENTS_RAPL];
	int events_cpu[NUM_GROUPS_CPU][MAX_EVENTS_CPU] = {0}; 
	int events_gpu[MAX_EVENTS_GPU] = {0}; 
	int events_nvml[NUM_EVENTS_NVML];
	
	//Holds the number of events detected
	int eventCountRAPL = 0;
	int eventCountCPU[NUM_GROUPS_CPU] = {0};
	int eventCountGPU = 0;
	int eventCountNVML = 0;
	
	//Timing variables
	long long before_time[NUM_COMPONENTS][NUM_GROUPS_CPU] = {0};
	long long after_time[NUM_COMPONENTS][NUM_GROUPS_CPU] = {0};
    	double elapsed_time[NUM_COMPONENTS][NUM_GROUPS_CPU] = {0.0};

	void NVML_init(){
	
	}
	
	/* convert PAPI RAPL native events to PAPI code */
	void RAPL_init(){
		#ifdef PSKEL_PAPI_DEBUG
			printf("\nConverting rapl group events name to code...\n");
		#endif
		
		for( int i = 0; i < NUM_EVENTS_RAPL; i++ ){
			retval = PAPI_event_name_to_code( (char *) EventNameRAPL[i], &events_rapl[i] );
			if( retval != PAPI_OK ) {
				fprintf( stderr, "PAPI_event_name_to_code failed\n" );
				continue;
			}
			eventCountRAPL++;
			#ifdef PSKEL_PAPI_DEBUG
				printf( "Name %s --- Code: %#x\n", EventNameRAPL[i], events_rapl[i] );
			#endif
			/*	
			retval = PAPI_get_event_info(events[i],&evinfo);
			if (retval != PAPI_OK) {
				fprintf( stderr, "PAPI_get_event_info failed\n" );
			}
			
			strncpy(units[i],evinfo.units,sizeof(units[0])-1);
			// buffer must be null terminated to safely use strstr operation on it below
			units[i][sizeof(units[0])-1] = '\0';

			data_type[i] = evinfo.data_type;
			*/
		}
		
		/* if we did not find any valid events, just report test failed. */
		if (eventCountRAPL == 0) {
				printf( "Test FAILED: no valid RAPL events found.\n");
				//exit(-1);
		}
		else{
			/* Create RAPL EventSet */	
			retval = PAPI_create_eventset( &EventSetRAPL );
			if( retval != PAPI_OK )
				fprintf( stderr, "PAPI_create_eventset RAPL failed with return value %d\n",retval );
			
			/* Add events to EventSet */
			retval = PAPI_add_events( EventSetRAPL, events_rapl, eventCountRAPL );
			if( retval != PAPI_OK ){
				fprintf( stderr, "PAPI_add_events RAPL failed with return value %d\n",retval );		
			}
		}
	}
	
	/* Initialize PAPI CPU counters */
	void CPU_init(){
		for(int i=0;i<NUM_GROUPS_CPU;i++){
			EventSetCPU[i] = PAPI_NULL;
		}
		
		/* convert PAPI CPU native events to PAPI code */
		#ifdef PSKEL_PAPI_DEBUG
			printf("\nConverting cpu group events name to code...\n");
		#endif
		
		for(int g = 0; g < NUM_GROUPS_CPU; g++){
			for( int i = 0; i < MAX_EVENTS_CPU; i++ ){
				if(strcmp(EventNameCPU[g][i],"") != 0){
					retval = PAPI_event_name_to_code( (char *)EventNameCPU[g][i], &events_cpu[g][i] );
					if( retval != PAPI_OK ) {
						fprintf( stderr, "PAPI_event_name_to_code failed\n" );
					continue;
					}
					eventCountCPU[g]++;
					#ifdef PSKEL_PAPI_DEBUG
						printf( "Name %s --- Code: %#x\n", EventNameCPU[g][i], events_cpu[g][i] );
					#endif
				}
			}
			#ifdef PSKEL_PAPI_DEBUG
				printf("# Events for group %d = %d\n\n",g,eventCountCPU[g]);
			#endif
		}		
		
		for(int g = 0; g < NUM_GROUPS_CPU; g++){
			if (eventCountCPU[g] == 0)
				fprintf( stderr, "Test FAILED: no valid PAPI CPU events found in group %d.\n",g);
				//exit(-1);
		}
		
		/* Create PAPI CPU EventSet */
		#ifdef PSKEL_PAPI_DEBUG
			printf("Creating CPU EventSet...\n");
		#endif	
					
		for(int g=0;g<NUM_GROUPS_CPU;g++){
			if(eventCountCPU[g]!= 0){
				retval = PAPI_create_eventset( &EventSetCPU[g] );
				if( retval != PAPI_OK )
					fprintf( stderr, "PAPI_create_eventset CPU failed for group %d with return value %d\n",g,retval );
			}
		}
		
		#ifdef PSKEL_PAPI_DEBUG
			printf("Adding events to CPU EventSet...\n");
		#endif
		
		for(int g=0;g<NUM_GROUPS_CPU;g++){
			if(eventCountCPU[g]!= 0){
				retval = PAPI_add_events( EventSetCPU[g], events_cpu[g], eventCountCPU[g] );
				if( retval != PAPI_OK )
					fprintf( stderr, "PAPI_add_events CPU failed for group %d with return value %d\n",g,retval );
			}
		}
	}
		
	/* Initialize PAPI GPU counters */
	void GPU_init(){
		/* convert PAPI CPU native events to PAPI code */
		#ifdef PSKEL_PAPI_DEBUG
			printf("\nConverting GPU group events name to code...\n");
		#endif
		
		//for(int g = 0; g < NUM_GROUPS_GPU; g++){
			for( int i = 0; i < MAX_EVENTS_GPU; i++ ){
				if(strcmp(EventNameGPU[i],"") != 0){
					retval = PAPI_event_name_to_code( (char*)EventNameGPU[i], &events_gpu[i] );
					if( retval != PAPI_OK ) {
						fprintf( stderr, "PAPI_event_name_to_code failed\n" );
					continue;
					}
					eventCountGPU++;
					#ifdef PSKEL_PAPI_DEBUG
						printf( "Name %s --- Code: %#x\n", EventNameGPU[i], events_gpu[i] );
					#endif
				}
			}
			#ifdef PSKEL_PAPI_DEBUG
				printf("# Events for GPU = %d\n\n",eventCountGPU);
			#endif
		//}		
		
		//for(int g = 0; g < NUM_GROUPS_GPU; g++){
		if (eventCountGPU == 0){
			fprintf( stderr, "Test FAILED: no valid GPU events found.\n");
		}
		else{
				//exit(-1);
			//}
			
			/* Create PAPI GPU EventSet */
			#ifdef PSKEL_PAPI_DEBUG
				printf("Creating GPU EventSet...\n");
			#endif	
						
			//for(int g=0;g<NUM_GROUPS_GPU;g++){
				//if(eventCountGPU[g]!= 0){
					retval = PAPI_create_eventset( &EventSetGPU );
					if( retval != PAPI_OK )
						fprintf( stderr, "PAPI_create_eventset GPU failed for with return value %d\n",retval );
				//}
			//}
			
			#ifdef PSKEL_PAPI_DEBUG
				printf("Adding events to GPU EventSet...\n");
			#endif
			
			//for(int g=0;g<NUM_GROUPS_GPU;g++){
				//if(eventCountGPU[g]!= 0){
					retval = PAPI_add_events( EventSetGPU, events_gpu, eventCountGPU );
					if( retval != PAPI_OK )
						fprintf( stderr, "PAPI_add_events GPU failed with return value %d\n",retval );
				//}
			//}
			}
	}

	/* Initializes PSkel PAPI */
	void init(int comp){
		if(!papi_init){
			/* PAPI Initialization */
			retval = PAPI_library_init( PAPI_VER_CURRENT );
			if( retval != PAPI_VER_CURRENT )
				fprintf( stderr, "PAPI_library_init failed\n" );
			
			#ifdef PSKEL_PAPI_DEBUG
				printf( "PAPI_VERSION     : %4d %6d %7d\n",
					PAPI_VERSION_MAJOR( PAPI_VERSION ),
					PAPI_VERSION_MINOR( PAPI_VERSION ),
					PAPI_VERSION_REVISION( PAPI_VERSION ) );
			#endif
			
			retval = PAPI_thread_init((unsigned long (*)(void))(omp_get_thread_num));
			if( retval != PAPI_OK )
				fprintf( stderr, "PAPI_thread_init failed\n" );
				
			papi_init=true;
		}
		
		switch(comp){
			case CPU:
				CPU_init();
				break;
			case GPU:
				GPU_init();
				break;
			case RAPL:
				RAPL_init();
				break;
			case NVML:
				NVML_init();
				break;
			default:
				fprintf(stderr, "Wrong component code %s\n",Components[comp]);
		}
	}

	inline void papi_start(int comp, unsigned int group){
		#ifdef PSKEL_PAPI_DEBUG
			printf("Start profiling for group %d of component %s...\n",group,Components[comp]);
		#endif
		
		before_time[comp][group] = PAPI_get_real_nsec();
		
		switch(comp){
			case CPU: retval = PAPI_start( EventSetCPU[group] ); break;
			case GPU: retval = PAPI_start( EventSetGPU ); break;
			case RAPL: retval = PAPI_start( EventSetRAPL ); break;
			case NVML: retval = PAPI_start( EventSetNVML ); break;
			default: fprintf( stderr, "Invalid component code %d",comp);retval=!PAPI_OK;break;
		}
		
		if( retval != PAPI_OK )
			fprintf( stderr, "PAPI_start failed for group %d of component %s with return value %d\n",group,Components[comp],retval );
	}
	
	inline void papi_stop(int comp, unsigned int group){
		after_time[comp][group] = PAPI_get_real_nsec();
		
		switch(comp){
			case CPU:  retval = PAPI_stop( EventSetCPU[group] , values_cpu[group] ); break;
			case GPU:  retval = PAPI_stop( EventSetGPU , values_gpu ); break;
			case RAPL: retval = PAPI_stop( EventSetRAPL       , values_rapl); break;
			case NVML: retval = PAPI_stop( EventSetNVML       , values_nvml); break;
			default: fprintf( stderr, "Invalid component code %d",comp); break;
		}
		
		if( retval != PAPI_OK )	
			fprintf( stderr, "PAPI_stop failed for group %d of componenet %s with return value %d\n",group,Components[comp],retval );
		
		elapsed_time[comp][group] = ((double)(after_time[comp][group] - before_time[comp][group]))/1.0e9;
		
		#ifdef PSKEL_PAPI_DEBUG
			printf("Stop profiling for group %d of component %s. Took %.3fs...\n",group,Components[comp],elapsed_time[comp][group]);
		#endif
	}
	inline void print_profile_values(int comp, unsigned int papi_group){
		printf("\n--------------------------------------------------------------------------------\n");
		switch (comp){
			case CPU:
				printf("PROFILE VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				//for(int g=0;g<NUM_GROUPS_CPU;g++){
					for( int i = 0; i < eventCountCPU[papi_group]; i++ ){
						printf("%s\t\t\t\t%12lld\n",EventNameCPU[papi_group][i],values_cpu[papi_group][i]);
					}
					printf("--------------------------------------------------------------------------------\n");
				//}
				printf("PERCENTUAL VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				
				#ifdef QUADRO
					switch(papi_group){
						case 0:
							printf("FP_INS  = %.3f\n",(100.0*((double)values_cpu[0][1]/values_cpu[0][0])));
							printf("VEC_SP  = %.3f\n",(100.0*((double)values_cpu[0][2]/values_cpu[0][0])));
							printf("LD_INS  = %.3f\n",(100.0*((double)values_cpu[0][3]/values_cpu[0][0])));
							printf("SR_INS  = %.3f\n",(100.0*((double)values_cpu[0][4]/values_cpu[0][0])));
							
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 1:					
							printf("BR_INS  = %.2f\n",(100.0*((double)values_cpu[1][1]/values_cpu[1][0])));
							printf("BR_CN   = %.2f\n",(100.0*((double)values_cpu[1][2]/values_cpu[1][1])));
							printf("BR_TKN  = %.2f\n",(100.0*((double)values_cpu[1][3]/values_cpu[1][1])));
							printf("BR_MSP  = %.2f\n",(100.0*((double)values_cpu[1][4]/values_cpu[1][1])));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 2:
							printf("L2_DCM  = %.2f\n",(100.0*((double)values_cpu[2][1]/values_cpu[2][0])));
							printf("L2_LDM  = %.2f\n",(100.0*((double)values_cpu[2][2]/values_cpu[2][1])));
							printf("L2_STM  = %.2f\n",(100.0*((double)values_cpu[2][3]/values_cpu[2][1])));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 3:
							printf("L2_DCR  = %.2f\n",(100.0*((double)values_cpu[3][0]/(values_cpu[3][0]+ values_cpu[3][1]))));
							printf("L3_DCR  = %.2f\n",(100.0*((double)values_cpu[3][2]/(values_cpu[3][2]+ values_cpu[3][3]))));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 4:
							printf("L3_TCM  = %.2f\n",(100.0*((double)values_cpu[4][1]/values_cpu[4][0])));
							printf("L3_LDM  = %.2f\n",(100.0*((double)values_cpu[4][2]/values_cpu[4][1])));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 5:
							printf("INS_CYC = %.2f\n",(double)values_cpu[5][1]/values_cpu[5][0]);
							printf("FPO_CYC = %.2f\n",(double)values_cpu[5][4]/values_cpu[5][0]);
							printf("STL_ICY = %.2f\n",(100.0*((double)values_cpu[5][2]/values_cpu[5][0])));
							printf("REF_CYC = %.2f\n",((double)values_cpu[5][0]/values_cpu[5][3]));
							printf("--------------------------------------------------------------------------------\n");
							break;
						default:
							cout<< "ERROR: invalid papi group" <<endl;
							break;
					}
				#else
					switch(papi_group){
						case 0:
							printf("FP_INS  = %.3f\n",(100.0*((double)values_cpu[1][1]/values_cpu[1][0])));	
							printf("FDV_INS = %.3f\n",(100.0*((double)values_cpu[1][2]/values_cpu[1][0])));
							printf("VEC_SP  = %.3f\n",(100.0*((double)values_cpu[0][1]/values_cpu[0][0])));
							
							#ifdef XEON
							printf("LD_INS  = %.3f\n",(100.0*((double)values_cpu[0][2]/values_cpu[0][0])));
							printf("SR_INS  = %.3f\n",(100.0*((double)values_cpu[0][3]/values_cpu[0][0])));	
							#endif						
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 1:						
							printf("BR_INS  = %.2f\n",(100.0*((double)values_cpu[2][1]/values_cpu[2][0])));
							printf("BR_CN   = %.2f\n",(100.0*((double)values_cpu[2][2]/values_cpu[2][1])));
							printf("BR_TKN  = %.2f\n",(100.0*((double)values_cpu[2][3]/values_cpu[2][1])));
							printf("BR_MSP  = %.2f\n",(100.0*((double)values_cpu[2][4]/values_cpu[2][1])));
							//printf("BR_UCN  = %.2f\n",(100.0*((double)values_cpu[2][3]/values_cpu[2][1])));
							//printf("BR_NTK  = %.2f\n",(100.0*((double)values_cpu[2][4]/values_cpu[2][2])));
							//printf("BR_PRC  = %.2f\n",(100.0*((double)values_cpu[2][6]/values_cpu[2][1])));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 3:
							printf("L2_DCM  = %.2f\n",(100.0*((double)values_cpu[3][1]/values_cpu[3][0])));
							printf("L2_TCM  = %.2f\n",(100.0*((double)values_cpu[3][3]/values_cpu[3][2])));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 4:
							printf("L2_DCR  = %.2f\n",(100.0*((double)values_cpu[4][0]/(values_cpu[4][0] + values_cpu[4][1]))));
							printf("L2_DCW  = %.2f\n",(100.0*((double)values_cpu[4][1]/(values_cpu[4][0] + values_cpu[4][1]))));
							printf("L2_TCR  = %.2f\n",(100.0*((double)values_cpu[4][2]/(values_cpu[4][2] + values_cpu[4][3]))));
							printf("L2_TCW  = %.2f\n",(100.0*((double)values_cpu[4][3]/(values_cpu[4][2] + values_cpu[4][3]))));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 5:
							printf("L3_TCM  = %.2f\n",(100.0*((double)values_cpu[5][1]/values_cpu[5][0])));
							printf("L3_DCR  = %.2f\n",(100.0*((double)values_cpu[5][2]/(values_cpu[5][2] + values_cpu[5][3]))));
							printf("L3_DCW  = %.2f\n",(100.0*((double)values_cpu[5][3]/(values_cpu[5][2] + values_cpu[5][3]))));
							printf("L3_TCR  = %.2f\n",(100.0*((double)values_cpu[5][4]/(values_cpu[5][4] + values_cpu[5][5]))));
							printf("L3_TCW  = %.2f\n",(100.0*((double)values_cpu[5][5]/(values_cpu[5][4] + values_cpu[5][5]))));
							printf("--------------------------------------------------------------------------------\n");
							break;
						case 6:
							printf("INS_CYC = %.2f\n",(double)values_cpu[6][1]/values_cpu[6][0]);
							printf("STL_ICY = %.2f\n",(100.0*((double)values_cpu[6][2]/values_cpu[6][0])));
							printf("REF_CYC = %.2f\n",((double)values_cpu[6][0]/values_cpu[6][3]));
							printf("--------------------------------------------------------------------------------\n");
							break;
						default:cout<< "ERROR: invalid papi group" <<endl;
							break;
					}
				#endif	
				break;
			case GPU:
				printf("PROFILE VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				for( int i = 0; i < eventCountGPU; i++ ){
					printf("%s\t\t%12lld\n",EventNameGPU[i],values_gpu[i]);
				}
				printf("--------------------------------------------------------------------------------\n");
				
				printf("PERCENTUAL VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				printf("BR_INS  = %.2f\n",(100.0*((double)values_gpu[1]/values_gpu[0])));
				printf("BR_DIV  = %.2f\n",(100.0*((double)values_gpu[2]/values_gpu[1])));
				printf("GLB_M   = %.2f\n",(100.0*((double)values_gpu[4]/(values_gpu[3]+values_gpu[4]))));
				printf("ACT_CYC = %.2f\n",(100.0*((double)values_gpu[6]/values_gpu[5])));
				printf("INS_CYC = %.2f\n",(double)values_gpu[0]/values_gpu[5]);
				
				printf("--------------------------------------------------------------------------------\n");
				break;
			case RAPL:
				for( int i = 0; i < eventCountRAPL; i++ ){
						//printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
						//if (strstr(units[i],"nJ")) {
					printf("%-40s%12.6f J\t(Average Power %.1fW)\n",
						EventNameRAPL[i],
						(double)values_rapl[i]/1.0e9,
						((double)values_rapl[i]/1.0e9)/elapsed_time[RAPL][0]);
				}
				break;
			case NVML:
				for( int i = 0; i < eventCountNVML; i++ ){
					printf("%s\t\t\t\t%12lld\n",EventNameNVML[i],values_nvml[i]);
				}
				break;
			default:
				fprintf(stderr, "Wrong component %s informed",Components[comp]);
		}
	}
	
	inline void print_profile_values(int comp){
		printf("\n--------------------------------------------------------------------------------\n");
		switch (comp){
			case CPU:
				printf("PROFILE VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				for(int g=0;g<NUM_GROUPS_CPU;g++){
					for( int i = 0; i < eventCountCPU[g]; i++ ){
						printf("%s\t\t\t\t%12lld\n",EventNameCPU[g][i],values_cpu[g][i]);
					}
					printf("--------------------------------------------------------------------------------\n");
				}
				#ifdef QUADRO
					printf("PERCENTUAL VALUES\n");
					printf("--------------------------------------------------------------------------------\n");
					printf("FP_INS  = %.3f\n",(100.0*((double)values_cpu[0][1]/values_cpu[0][0])));
					printf("VEC_SP  = %.3f\n",(100.0*((double)values_cpu[0][2]/values_cpu[0][0])));
					printf("LD_INS  = %.3f\n",(100.0*((double)values_cpu[0][3]/values_cpu[0][0])));
					printf("SR_INS  = %.3f\n",(100.0*((double)values_cpu[0][4]/values_cpu[0][0])));
					
					printf("--------------------------------------------------------------------------------\n");
					
					printf("BR_INS  = %.2f\n",(100.0*((double)values_cpu[1][1]/values_cpu[1][0])));
					printf("BR_CN   = %.2f\n",(100.0*((double)values_cpu[1][2]/values_cpu[1][1])));
					printf("BR_TKN  = %.2f\n",(100.0*((double)values_cpu[1][3]/values_cpu[1][1])));
					printf("BR_MSP  = %.2f\n",(100.0*((double)values_cpu[1][4]/values_cpu[1][1])));
					
					printf("--------------------------------------------------------------------------------\n");
					printf("L2_DCM  = %.2f\n",(100.0*((double)values_cpu[2][1]/values_cpu[2][0])));
					printf("L2_LDM  = %.2f\n",(100.0*((double)values_cpu[2][2]/values_cpu[2][1])));
					printf("L2_STM  = %.2f\n",(100.0*((double)values_cpu[2][3]/values_cpu[2][1])));
					printf("L2_DCR  = %.2f\n",(100.0*((double)values_cpu[3][0]/(values_cpu[3][0]+ values_cpu[3][1]))));
					
					printf("--------------------------------------------------------------------------------\n");
					
					printf("L3_TCM  = %.2f\n",(100.0*((double)values_cpu[4][1]/values_cpu[4][0])));
					printf("L3_LDM  = %.2f\n",(100.0*((double)values_cpu[4][2]/values_cpu[4][1])));
					printf("L3_DCR  = %.2f\n",(100.0*((double)values_cpu[3][2]/(values_cpu[3][2]+ values_cpu[3][3]))));
					
					printf("--------------------------------------------------------------------------------\n");
					
					printf("INS_CYC = %.2f\n",(double)values_cpu[5][1]/values_cpu[5][0]);
					printf("FPO_CYC = %.2f\n",(double)values_cpu[5][4]/values_cpu[5][0]);
					printf("STL_ICY = %.2f\n",(100.0*((double)values_cpu[5][2]/values_cpu[5][0])));
					printf("REF_CYC = %.2f\n",((double)values_cpu[5][0]/values_cpu[5][3]));
					
					printf("--------------------------------------------------------------------------------\n");
				#else
					printf("PERCENTUAL VALUES\n");
					printf("--------------------------------------------------------------------------------\n");
					printf("FP_INS  = %.3f\n",(100.0*((double)values_cpu[1][1]/values_cpu[1][0])));	
					printf("FDV_INS = %.3f\n",(100.0*((double)values_cpu[1][2]/values_cpu[1][0])));
					printf("VEC_SP  = %.3f\n",(100.0*((double)values_cpu[0][1]/values_cpu[0][0])));
					
					#ifdef XEON
					printf("LD_INS  = %.3f\n",(100.0*((double)values_cpu[0][2]/values_cpu[0][0])));
					printf("SR_INS  = %.3f\n",(100.0*((double)values_cpu[0][3]/values_cpu[0][0])));	
					#endif
					
					printf("--------------------------------------------------------------------------------\n");
					
					printf("BR_INS  = %.2f\n",(100.0*((double)values_cpu[2][1]/values_cpu[2][0])));
					printf("BR_CN   = %.2f\n",(100.0*((double)values_cpu[2][2]/values_cpu[2][1])));
					printf("BR_TKN  = %.2f\n",(100.0*((double)values_cpu[2][3]/values_cpu[2][1])));
					printf("BR_MSP  = %.2f\n",(100.0*((double)values_cpu[2][4]/values_cpu[2][1])));
					//printf("BR_UCN  = %.2f\n",(100.0*((double)values_cpu[2][3]/values_cpu[2][1])));
					//printf("BR_NTK  = %.2f\n",(100.0*((double)values_cpu[2][4]/values_cpu[2][2])));
					//printf("BR_PRC  = %.2f\n",(100.0*((double)values_cpu[2][6]/values_cpu[2][1])));
					
					printf("--------------------------------------------------------------------------------\n");
					
					printf("L2_DCM  = %.2f\n",(100.0*((double)values_cpu[3][1]/values_cpu[3][0])));
					printf("L2_TCM  = %.2f\n",(100.0*((double)values_cpu[3][3]/values_cpu[3][2])));

					printf("L2_DCR  = %.2f\n",(100.0*((double)values_cpu[4][0]/(values_cpu[4][0] + values_cpu[4][1]))));
					printf("L2_DCW  = %.2f\n",(100.0*((double)values_cpu[4][1]/(values_cpu[4][0] + values_cpu[4][1]))));
					printf("L2_TCR  = %.2f\n",(100.0*((double)values_cpu[4][2]/(values_cpu[4][2] + values_cpu[4][3]))));
					printf("L2_TCW  = %.2f\n",(100.0*((double)values_cpu[4][3]/(values_cpu[4][2] + values_cpu[4][3]))));
					
					printf("--------------------------------------------------------------------------------\n");
					
					printf("L3_TCM  = %.2f\n",(100.0*((double)values_cpu[5][1]/values_cpu[5][0])));
					printf("L3_DCR  = %.2f\n",(100.0*((double)values_cpu[5][2]/(values_cpu[5][2] + values_cpu[5][3]))));
					printf("L3_DCW  = %.2f\n",(100.0*((double)values_cpu[5][3]/(values_cpu[5][2] + values_cpu[5][3]))));
					printf("L3_TCR  = %.2f\n",(100.0*((double)values_cpu[5][4]/(values_cpu[5][4] + values_cpu[5][5]))));
					printf("L3_TCW  = %.2f\n",(100.0*((double)values_cpu[5][5]/(values_cpu[5][4] + values_cpu[5][5]))));
										
					printf("--------------------------------------------------------------------------------\n");
					
					printf("INS_CYC = %.2f\n",(double)values_cpu[6][1]/values_cpu[6][0]);
					printf("STL_ICY = %.2f\n",(100.0*((double)values_cpu[6][2]/values_cpu[6][0])));
					printf("REF_CYC = %.2f\n",((double)values_cpu[6][0]/values_cpu[6][3]));
					
					printf("--------------------------------------------------------------------------------\n");
				#endif	
				break;
			case GPU:
				printf("PROFILE VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				for( int i = 0; i < eventCountGPU; i++ ){
					printf("%s\t\t%12lld\n",EventNameGPU[i],values_gpu[i]);
				}
				printf("--------------------------------------------------------------------------------\n");
				
				printf("PERCENTUAL VALUES\n");
				printf("--------------------------------------------------------------------------------\n");
				printf("BR_INS  = %.2f\n",(100.0*((double)values_gpu[1]/values_gpu[0])));
				printf("BR_DIV  = %.2f\n",(100.0*((double)values_gpu[2]/values_gpu[1])));
				printf("GLB_M   = %.2f\n",(100.0*((double)values_gpu[4]/(values_gpu[3]+values_gpu[4]))));
				printf("ACT_CYC = %.2f\n",(100.0*((double)values_gpu[6]/values_gpu[5])));
				printf("INS_CYC = %.2f\n",(double)values_gpu[0]/values_gpu[5]);
				
				printf("--------------------------------------------------------------------------------\n");
				break;
			case RAPL:
				for( int i = 0; i < eventCountRAPL; i++ ){
						//printf( "%12lld \t\t --> %s \n", values[i], EventName[i] );
						//if (strstr(units[i],"nJ")) {
					printf("%-40s%12.6f J\t(Average Power %.1fW)\n",
						EventNameRAPL[i],
						(double)values_rapl[i]/1.0e9,
						((double)values_rapl[i]/1.0e9)/elapsed_time[RAPL][0]);
				}
				break;
			case NVML:
				for( int i = 0; i < eventCountNVML; i++ ){
					printf("%s\t\t\t\t%12lld\n",EventNameNVML[i],values_nvml[i]);
				}
				break;
			default:
				fprintf(stderr, "Wrong component %s informed",Components[comp]);
		}
	}
	
	void shutdown(){		
		if(eventCountCPU[0] != 0){
			for(int g=0;g<NUM_GROUPS_CPU;g++){
				retval = PAPI_cleanup_eventset(EventSetCPU[g]);
				if( retval != PAPI_OK )
					fprintf(stderr, "PAPI_cleanup_eventset failed for CPU on group %d\n",g);
			}
		
			for(int g=0;g<NUM_GROUPS_CPU;g++){
				retval = PAPI_destroy_eventset(&EventSetCPU[g]);
				if (retval != PAPI_OK)
					fprintf(stderr, "PAPI_destroy_eventset failed for CPU on group %d\n",g);
			}
		}
		
		if(eventCountGPU != 0){
				retval = PAPI_cleanup_eventset(EventSetGPU);
				if( retval != PAPI_OK )
					fprintf(stderr, "PAPI_cleanup_eventset failed for GPU\n");
		
			//for(int g=0;g<NUM_GROUPS_GPU;g++){
				retval = PAPI_destroy_eventset(&EventSetGPU);
				if (retval != PAPI_OK)
					fprintf(stderr, "PAPI_destroy_eventset failed for GPU\n");
			//}
		}
		
		if(eventCountRAPL != 0){
				retval = PAPI_cleanup_eventset(EventSetRAPL);
				if( retval != PAPI_OK )
					fprintf(stderr, "PAPI_cleanup_eventset failed for RAPL\n");
		
				retval = PAPI_destroy_eventset(&EventSetRAPL);
				if (retval != PAPI_OK)
					fprintf(stderr, "PAPI_destroy_eventset failed for RAPL\n");
		}
		
		if(eventCountNVML != 0){
				retval = PAPI_cleanup_eventset(EventSetNVML);
				if( retval != PAPI_OK )
					fprintf(stderr, "PAPI_cleanup_eventset failed for NVML\n");
		
				retval = PAPI_destroy_eventset(&EventSetNVML);
				if (retval != PAPI_OK)
					fprintf(stderr, "PAPI_destroy_eventset failed for NVML\n");
		}
		
		PAPI_shutdown();
	}
	
}; //end PSkelPAPI
} //end namespace	
