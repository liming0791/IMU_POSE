#include "stdio.h"
#include "stdlib.h"
#include "../include/MadgwickAHRS.h"
#include "../include/MahonyAHRS.h"

#define PI 3.1415926


int main(int argc, char** argv)
{


    if(argc!=10)
    {
        printf("arguments illegal!\n");
        return 0;
    }

    float v[9];
    for(int i = 0; i < 9; i++){
        v[i] = atof(argv[i+1]);
    }

    float q[4] = { 1, 0, 0, 0 };
    int freq = 100;

    printf("==================test values====================\n");
    printf("%f %f %f %f %f %f %f %f %f\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);

    MadgwickAHRS::updateIMU(v[3]/180*PI, v[4]/180*PI, v[5]/180*PI, v[0], v[1], v[2], 100, q[0], q[1], q[2], q[3]);
    printf("==================test MadgwickAHRS====================\n");
    printf("%f %f %f %f\n", q[0], q[1], q[2], q[3]);

    MahonyAHRS::updateIMU(v[3]/180*PI, v[4]/180*PI, v[5]/180*PI, v[0], v[1], v[2], 100,  q[0], q[1], q[2], q[3]);
    printf("===================test MahonyAHRS====================\n");
    printf("%f %f %f %f\n", q[0], q[1], q[2], q[3]);



    return 0;
}
