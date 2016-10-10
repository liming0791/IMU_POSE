//=====================================================================================================
// MahonyAHRS.h
//=====================================================================================================
//
// Madgwick's implementation of Mayhony's AHRS algorithm.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//
// Date			Author			Notes
// 29/09/2011	SOH Madgwick    Initial release
// 02/10/2011	SOH Madgwick	Optimised for reduced CPU load
//
//=====================================================================================================
#ifndef MahonyAHRS_h
#define MahonyAHRS_h


// Function declarations

class MahonyAHRS
{
    public:
        static void update(float gx, float gy, float gz, 
                float ax, float ay, float az, 
                float mx, float my, float mz,
                float sampleFreq,
                float& q0, float& q1, float& q2, float& q3);
        static void updateIMU(float gx, float gy, float gz, 
                float ax, float ay, float az,
                float sampleFreq,
                float& q0, float& q1, float& q2, float& q3);
        static void reset();

    private:
        static float integralFBx, integralFBy, integralFBz;
        static float twoKi, twoKp;
        static float invSqrt(float x);
        
};

#endif
//=====================================================================================================
// End of file
//=====================================================================================================
