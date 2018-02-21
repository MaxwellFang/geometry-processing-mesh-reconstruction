#include "fd_partial_derivative.h"
#include <iostream>

void fd_partial_derivative(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  const int dir,
  Eigen::SparseMatrix<double> & D)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  int dx = 0;
  int dy = 0;
  int dz = 0;
  if (dir == 0) dx = 1;
  if (dir == 1) dy = 1;
  if (dir == 2) dz = 1;
  D.resize((nx-dx)*(ny-dy)*(nz-dz), nx*ny*nz);
  typedef Eigen::Triplet<double> T;
  std::vector<T> list;
  list.reserve((nx-dx)*(ny-dy)*(nz-dz)*2);
  for(int i=0; i<nx-dx; i++) {
    for(int j=0; j<ny-dy; j++) {
      for(int k=0; k<nz-dz; k++) {
        list.push_back(T(i+j*(nx-dx)+k*(ny-dy)*(nx-dx), (i+dx)+(j+dy)*nx+(k+dz)*nx*ny, 1));
        list.push_back(T(i+j*(nx-dx)+k*(ny-dy)*(nx-dx), i+j*nx+k*nx*ny, -1));
      }
    }
  }
  D.setFromTriplets(list.begin(), list.end());

  ////////////////////////////////////////////////////////////////////////////
}
