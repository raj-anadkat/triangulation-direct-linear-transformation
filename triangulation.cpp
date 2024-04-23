#include <Eigen/Dense>
#include <iostream>
#include <cmath>

using namespace Eigen;
using namespace std;

const float PI = 3.141592653589793;

int main() {
    // Set up the constants, camera centers, and pixel coordinates (pixels)
    double f = 270;
    double cx = 320;
    double cy = 240; 

    Vector3d d1(400, 220, 1);  // First pixel coordinates
    Vector3d d2(120, 300, 1);  // Second pixel coordinates

    Vector3d p1(2, 2, 1);  // First camera center
    Vector3d p2(6, 3, 2);  // Second camera center

    double theta1_deg = -10;  // Heading angle for the first camera
    double theta2_deg = 40;  // Heading angle for the second camera

    double theta1_rad = theta1_deg * PI / 180.0;  // Convert to radians
    double theta2_rad = theta2_deg * PI / 180.0;  // Convert to radians

    // Camera intrinsic matrix
    Matrix3d K;
    K << f, 0, cx,
         0, f, cy,
         0, 0, 1;

    // Rotation matrices
    Matrix3d R1;
    R1 << cos(theta1_rad), -sin(theta1_rad), 0,
          sin(theta1_rad), cos(theta1_rad), 0,
          0, 0, 1;

    Matrix3d R2;
    R2 << cos(theta2_rad), -sin(theta2_rad), 0,
          sin(theta2_rad), cos(theta2_rad), 0,
          0, 0, 1;

    // Projection matrices
    Matrix<double, 3, 4> P1;
    P1.block<3, 3>(0, 0) = K * R1;
    P1.block<3, 1>(0, 3) = K * p1;

    Matrix<double, 3, 4> P2;
    P2.block<3, 3>(0, 0) = K * R2;
    P2.block<3, 1>(0, 3) = K * p2;

    // Set up the linear system with pixel coordinates from d1 and d2
    Matrix<double, 4, 3> A;  // 4x3 matrix, 4 equations and 3 unknowns
    Vector4d b;  // 4-element vector for constants

    // First camera equations (horizontal and vertical components)
    A.row(0) = (d1(0) * P1.row(2) - P1.row(0)).head(3);
    A.row(1) = (d1(1) * P1.row(2) - P1.row(1)).head(3);

    // Second camera equations (horizontal and vertical components)
    A.row(2) = (d2(0) * P2.row(2) - P2.row(0)).head(3);
    A.row(3) = (d2(1) * P2.row(2) - P2.row(1)).head(3);

    // Construct the constant vector 'b'
    b(0) = P1(0, 3) - d1(0) * P1(2, 3);
    b(1) = P1(1, 3) - d1(1) * P1(2, 3);
    b(2) = P2(0, 3) - d2(0) * P2(2, 3);
    b(3) = P2(1, 3) - d2(1) * P2(2, 3);

    // Use SVD to find the least-squares solution to Ax = B
    JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV); // Use thin SVD for smaller matrices
    Vector3d x = svd.solve(b); // Obtain the solution

    cout<<"Real World Coordinates of Object (m): "<<x<<endl;

    // Transform x into homogeneous coordinates by appending a 1 at the end
    Vector4d x_homogeneous;
    x_homogeneous << x, 1;


    // Take the dot product of P1 and x_homogeneous
    Vector3d P1_dot_x = P1 * x_homogeneous;

    // Divide the resulting vector P1_dot_x by its last value (scale)
    double scale = P1_dot_x(2);
    P1_dot_x /= scale;

    cout<<"Actual Pixel Coords: "<<d1<<endl;
    cout<<"Reprojected Pixel Coords: "<<P1_dot_x<<endl;


    // Calculate the reprojection error (Euclidean distance between d1 and P1_dot_x)
    double reprojection_error = (d1.head<2>() - P1_dot_x.head<2>()).norm();

    cout << "Reprojection error between d1 and P1_dot_x: " << reprojection_error << endl;

    return 0;
}
