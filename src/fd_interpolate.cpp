#include "fd_interpolate.h"
#include <iostream>
using namespace std;

void fd_interpolate(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const Eigen::RowVector3d & corner,
  const Eigen::MatrixXd & P,
  Eigen::SparseMatrix<double> & W)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  // W[8*P.rows(), i + j*ny + k*nx*ny], t'th interpolation
  // We have to end up with W1, W2, W3 of size 8*nx*ny*nz x nx*ny*nz
  typedef Eigen::Triplet<double> T;
  std::vector<T> list;
  list.reserve(P.rows()*8);
  for(int r = 0; r < P.rows(); r++) {
    int x = floor((P(r, 0) - corner(0))/h);
    int y = floor((P(r, 1) - corner(1))/h);
    int z = floor((P(r, 2) - corner(2))/h);
    double dx = (P(r, 0) - corner(0))/h - x;
    double dy = (P(r, 1) - corner(1))/h - y;
    double dz = (P(r, 2) - corner(2))/h - z;
    list.push_back(T(r, x+y*nx+z*nx*ny, (1-dx)*(1-dy)*(1-dz)));
    list.push_back(T(r, (x+1)+y*nx+z*nx*ny, dx*(1-dy)*(1-dz)));
    list.push_back(T(r, x+(y+1)*nx+z*nx*ny, (1-dx)*dy*(1-dz)));
    list.push_back(T(r, (x+1)+(y+1)*nx+z*nx*ny, dx*dy*(1-dz)));
    list.push_back(T(r, x+y*nx+(z+1)*nx*ny, (1-dx)*(1-dy)*dz));
    list.push_back(T(r, (x+1)+y*nx+(z+1)*nx*ny, dx*(1-dy)*dz));
    list.push_back(T(r, x+(y+1)*nx+(z+1)*nx*ny, (1-dx)*dy*dz));
    list.push_back(T(r, (x+1)+(y+1)*nx+(z+1)*nx*ny, dx*dy*dz));
  }
  W.resize(P.rows(), nx*ny*nz);
  W.setFromTriplets(list.begin(), list.end());
  ////////////////////////////////////////////////////////////////////////////
}
