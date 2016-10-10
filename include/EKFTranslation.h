#ifndef EKFTRANSLATION_H
#define EKFTRANSLATION_H

#include "stdio.h"
#include "stdlib.h"
#include "math.h"

#include <vector>
#include <mutex>

#include "Eigen/Dense"

class EKFTranslation
{
    public:
        static void predictEKF(double a0, double a1, double a2, double dt);
        static void updateEKF(double p0, double p1, double p2);
        static void updatePartialEKF(double a0, double a1, double a2, double dt);
        static void updateNoEKF(double a0, double a1, double a2, double dt);
        static void updateAcc(double a0, double a1, double a2);
        static void getTranslation(std::vector<float> &T);
        static void getTranslation(double *T);
        static void getTranslation(double &t0, double &t1, double &t2);
        static void setTranslation(std::vector<float> &T);
        static void reset();
        static void getStatus(std::vector<float> &S);
        static void getInitR(float* R);
        static void getInitQ(std::vector<float> &Q);
        static void getAcc(std::vector<float> &A);

        static double a[3];
        static double v[3];
        static double p[3];
        static Eigen::MatrixXd F, P, G, H, R, Q ;
        static Eigen::MatrixXd PF, PP, PG, PH, PR, PQ ;
        static std::mutex mMuteEKF;
};

#endif
