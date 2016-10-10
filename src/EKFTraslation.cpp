#include "EKFTranslation.h"

std::mutex EKFTranslation::mMuteEKF;

double EKFTranslation::a[] = {0, 0, 0};
double EKFTranslation::v[] = {0, 0, 0};
double EKFTranslation::p[] = {0, 0, 0};

// Full EKF
Eigen::MatrixXd EKFTranslation::F = Eigen::MatrixXd::Identity(9,9);
Eigen::MatrixXd EKFTranslation::P = Eigen::MatrixXd::Identity(9,9);
Eigen::MatrixXd EKFTranslation::G = Eigen::MatrixXd::Zero(9,3);
Eigen::MatrixXd EKFTranslation::H(
        (Eigen::MatrixXd(3,9) << 
         1, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0).finished() ); 
Eigen::MatrixXd EKFTranslation::R(
        (Eigen::MatrixXd(3,3) << 
         0.0001, 0, 0,
         0, 0.0001, 0,
         0, 0, 0.0001).finished() );
Eigen::MatrixXd EKFTranslation::Q(
        (Eigen::MatrixXd(9,9) << 
         0.001, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0.001, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0.001, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0.002, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0.002, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0.002, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0.003, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0.003, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0.003).finished() ); 

// Partial EKF
Eigen::MatrixXd EKFTranslation::PF = Eigen::MatrixXd::Identity(3,3);
Eigen::MatrixXd EKFTranslation::PP = Eigen::MatrixXd::Identity(3,3);
Eigen::MatrixXd EKFTranslation::PG = Eigen::MatrixXd::Zero(3,3);
Eigen::MatrixXd EKFTranslation::PH(
        (Eigen::MatrixXd(3,3) << 
                1, 0, 0,
                0, 1, 0,
                0, 0, 1).finished() ); 
Eigen::MatrixXd EKFTranslation::PR(
        (Eigen::MatrixXd(3,3) << 
                5, 0, 0,
                0, 5, 0,
                0, 0, 5).finished() );
Eigen::MatrixXd EKFTranslation::PQ(
        (Eigen::MatrixXd(3,3) << 
                0.01, 0, 0,
                0, 0.01, 0,
                0, 0, 0.01).finished() ); 

void EKFTranslation::predictEKF(double a0, double a1, double a2, double dt)
{
    double dt2_2 = dt*dt/2;

    {
        std::lock_guard<std::mutex> lock(mMuteEKF);
        // Construct F
        F(0, 3) = F(1, 4) = F(2, 5) = F(3, 6) = F(4, 7) = F(5, 8) = dt;
        F(0, 6) = F(1, 7) = F(2, 8) = -dt2_2;

        // Predict
        p[0] = p[0] + v[0]*dt + (a0 - a[0])*dt2_2; 
        p[1] = p[1] + v[1]*dt + (a1 - a[1])*dt2_2;
        p[2] = p[2] + v[2]*dt + (a2 - a[2])*dt2_2;

        v[0] = v[0] + (a0 - a[0])*dt;
        v[1] = v[1] + (a1 - a[1])*dt;
        v[2] = v[2] + (a2 - a[2])*dt;

        P = F*P*F.transpose() + Q;
    }
}

void EKFTranslation::updateEKF(double p0, double p1, double p2)
{
    // Update
    {
        std::lock_guard<std::mutex> lock(mMuteEKF);

        if (F(0, 3) == 0) {
            printf("would not update, because have not predict yet !");
            return;
        }

        G =  P*H.transpose()*(H*P*H.transpose() + R).inverse();
        P = (Eigen::MatrixXd::Identity(9,9) - G*H)*P;
        Eigen::MatrixXd h(3,1);
        h << p0 - p[0],
             p1 - p[1],
             p2 - p[2];
        Eigen::MatrixXd g(9,1);
        g = G*h;

        p[0] += g(0, 0);
        p[1] += g(1, 0); 
        p[2] += g(2, 0);

        v[0] += g(3, 0);
        v[1] += g(4, 0); 
        v[2] += g(5, 0);

        a[0] += g(6, 0);
        a[1] += g(7, 0); 
        a[2] += g(8, 0);
    }
}

void EKFTranslation::updateAcc(double a0, double a1, double a2)
{
//    // EKF only on acce
//    // Predict
//    PP = PF * PP * (PF.transpose());
//
//    // Update
//    PG =  PP * ((PH.transpose()) * ( PH * PP * (PH.transpose()) + PR).inverse());
//    PP = (Eigen::MatrixXd::Identity(3,3) - PG*PH)*PP;
//    Eigen::MatrixXd h(3,1);
//    h << a0 - a[0],
//         a1 - a[1],
//         a2 - a[2];
//    Eigen::MatrixXd g(3,1);
//    g = PG*h;
//
//    a[0] += g(0, 0);
//    a[1] += g(1, 0); 
//    a[2] += g(2, 0);

    a[0] = (a0 + a[0])/2;
    a[1] = (a1 + a[1])/2;
    a[2] = (a2 + a[2])/2;
}

void EKFTranslation::updatePartialEKF(double a0, double a1, double a2, double dt)
{

    // Update Position and Velocity
    double v0 = v[0] + a[0]*dt;
    double v1 = v[1] + a[1]*dt;
    double v2 = v[2] + a[2]*dt;

    p[0] = p[0] + (v[0] + v0)/2*dt; 
    p[1] = p[1] + (v[1] + v1)/2*dt;
    p[2] = p[2] + (v[2] + v2)/2*dt;

    v[0] = v0;
    v[1] = v1;
    v[2] = v2;

    printf("\n===Dt: %f v0: %f p0: %f\n\n", dt, v[0], p[0]);

    // EKF only on acce
    // Predict
    PP = PF*PP*PF.transpose();

    // Update
    PG =  PP*PH.transpose()*(PH*PP*PH.transpose() + PR).inverse();
    PP = (Eigen::MatrixXd::Identity(3,3) - PG*PH)*PP;
    Eigen::MatrixXd h(3,1);
    h << a0 - a[0],
         a1 - a[1],
         a2 - a[2];
    Eigen::MatrixXd g(3,1);
    g = PG*h;

    a[0] += g(0, 0);
    a[1] += g(1, 0); 
    a[2] += g(2, 0);

}

void EKFTranslation::updateNoEKF(double a0, double a1, double a2, double dt)
{
    double dt2_2 = dt*dt/2;
    p[0] = p[0] + v[0]*dt + a[0]*dt2_2; 
    p[1] = p[1] + v[1]*dt + a[1]*dt2_2;
    p[2] = p[2] + v[2]*dt + a[2]*dt2_2;

    v[0] = v[0] + a[0]*dt;
    v[1] = v[1] + a[1]*dt;
    v[2] = v[2] + a[2]*dt;

    a[0] = a0;
    a[1] = a1;
    a[2] = a2;
}

void EKFTranslation::getTranslation(std::vector<float> &T)
{
    if(T.size()!=3)
        T.resize(3);
    T[0] = p[0];
    T[1] = p[1];
    T[2] = p[2];
}

void EKFTranslation::getTranslation(double *T)
{
    T[0] = p[0];
    T[1] = p[1];
    T[2] = p[2];
}

void EKFTranslation::getTranslation(double &t0, double &t1, double &t2)
{
    t0 = p[0];
    t1 = p[1];
    t2 = p[2];
}

void EKFTranslation::setTranslation(std::vector<float> &T)
{
    p[0] = T[0];
    p[1] = T[1];
    p[2] = T[2];
}

void EKFTranslation::reset()
{
   a[0] = a[1] = a[2] = 0; 
   v[0] = v[1] = v[2] = 0;
   a[0] = a[1] = a[2] = 0;

   P = Eigen::MatrixXd::Identity(9,9);
   PP = Eigen::MatrixXd::Identity(3,3);
}

void EKFTranslation::getStatus(std::vector<float> &S)
{
    if(S.size() != 9)
        S.resize(9);
    S[0] = p[0];
    S[1] = p[1];
    S[2] = p[2];
    S[3] = v[0];
    S[4] = v[1];
    S[5] = v[2];
    S[6] = a[0];
    S[7] = a[1];
    S[8] = a[2];
}

void EKFTranslation::getInitR(float* R)
{
    double r = atan2(a[1], a[2]);
    double p = atan2(a[0], (sqrt(a[2]*a[2]+a[1]*a[1])));
    double y = 0;

    float sy = sin(y), cy = cos(y),
          sp = sin(p), cp = cos(p),
          sr = sin(r), cr = cos(r);

    R[0]=cp*cy,          R[1]=cp*sy,          R[2]=-sp,
    R[3]=sr*sp*cy-cr*sy, R[4]=sr*sp*sy+cr*cy, R[5]=sr*cp,
    R[6]=cr*sp*cy+sr*sy, R[7]=cr*sp*sy-sr*cy, R[8]=cr*cp;

}

void EKFTranslation::getInitQ(std::vector<float> &Q)
{
    float n[3];
    n[0] = a[1];
    n[1] = -a[0];
    n[2] = 0;
    float len = sqrt(a[1]*a[1] + a[0]*a[0]);
    n[0] /= len;
    n[1] /= len;

    float cos_angle = a[2]/
            sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
    float angle_2 = acos(cos_angle)/2;

    printf("\n===a:%f %f %f\n", a[0], a[1], a[2]);
    printf("\n===n:%f %f %f\n", n[0], n[1], n[2]);
    printf("\n===cos_a:%f\n", cos_angle);
    printf("\n===angle_2:%f\n", angle_2);

    if(Q.empty())
        Q.resize(4);
    Q[0] = cos(angle_2);
    Q[1] = n[0] * sin(angle_2);
    Q[2] = n[1] * sin(angle_2);
    Q[3] = n[2] * sin(angle_2);
}

void EKFTranslation::getAcc(std::vector<float> &A)
{
    if(A.empty())
        A.resize(3);
    A[0] = a[0];
    A[1] = a[1];
    A[2] = a[2];
}
