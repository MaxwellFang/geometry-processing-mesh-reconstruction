#include "fd_grad.h"
#include <iostream>
#include "igl/cat.h"
#include "fd_partial_derivative.h"
using namespace std;

void fd_grad(
  const int nx,
  const int ny,
  const int nz,
  const double h,
  Eigen::SparseMatrix<double> & G)
{
  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  Eigen::SparseMatrix<double> Dx;
  Eigen::SparseMatrix<double> Dy;
  Eigen::SparseMatrix<double> Dz;
  fd_partial_derivative(nx, ny, nz, h, 0, Dx);
  fd_partial_derivative(nx, ny, nz, h, 1, Dy);
  fd_partial_derivative(nx, ny, nz, h, 2, Dz);
  G = igl::cat(1, Dx, Dy);
  G = igl::cat(1, G, Dz);
  ////////////////////////////////////////////////////////////////////////////
}
