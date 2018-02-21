#include "poisson_surface_reconstruction.h"
#include <igl/copyleft/marching_cubes.h>
#include <algorithm>
#include <iostream>
#include "fd_grad.h"
#include "fd_interpolate.h"
#include <Eigen/IterativeLinearSolvers>
using namespace std;

void poisson_surface_reconstruction(
    const Eigen::MatrixXd & P,
    const Eigen::MatrixXd & N,
    Eigen::MatrixXd & V,
    Eigen::MatrixXi & F)
{
  ////////////////////////////////////////////////////////////////////////////
  // Construct FD grid, CONGRATULATIONS! You get this for free!
  ////////////////////////////////////////////////////////////////////////////
  // number of input points
  const int n = P.rows();
  // Grid dimensions
  int nx, ny, nz;
  // Maximum extent (side length of bounding box) of points
  double max_extent =
    (P.colwise().maxCoeff()-P.colwise().minCoeff()).maxCoeff();
  // padding: number of cells beyond bounding box of input points
  const double pad = 8;
  // choose grid spacing (h) so that shortest side gets 30+2*pad samples
  double h  = max_extent/double(30+2*pad);
  // Place bottom-left-front corner of grid at minimum of points minus padding
  Eigen::RowVector3d corner = P.colwise().minCoeff().array()-pad*h;
  // Grid dimensions should be at least 3
  nx = std::max((P.col(0).maxCoeff()-P.col(0).minCoeff()+(2.*pad)*h)/h,3.);
  ny = std::max((P.col(1).maxCoeff()-P.col(1).minCoeff()+(2.*pad)*h)/h,3.);
  nz = std::max((P.col(2).maxCoeff()-P.col(2).minCoeff()+(2.*pad)*h)/h,3.);
  // Compute positions of grid nodes
  Eigen::MatrixXd x(nx*ny*nz, 3);
  for(int i = 0; i < nx; i++)
  {
    for(int j = 0; j < ny; j++)
    {
      for(int k = 0; k < nz; k++)
      {
         // Convert subscript to index
         const auto ind = i + nx*(j + k * ny);
         x.row(ind) = corner + h*Eigen::RowVector3d(i,j,k);
      }
    }
  }
  Eigen::VectorXd g = Eigen::VectorXd::Zero(nx*ny*nz);

  ////////////////////////////////////////////////////////////////////////////
  // Add your code here
  ////////////////////////////////////////////////////////////////////////////

  Eigen::SparseMatrix<double> Wx;
  Eigen::SparseMatrix<double> Wy;
  Eigen::SparseMatrix<double> Wz;
  Eigen::RowVector3d stage_x(h/2, 0, 0);
  Eigen::RowVector3d stage_y(0, h/2, 0);
  Eigen::RowVector3d stage_z(0, 0, h/2);
  Eigen::RowVector3d corner_x = corner + stage_x;
  Eigen::RowVector3d corner_y = corner + stage_y;
  Eigen::RowVector3d corner_z = corner + stage_z;
  fd_interpolate(nx-1, ny, nz, h, corner_x, P, Wx);
  fd_interpolate(nx, ny-1, nz, h, corner_y, P, Wy);
  fd_interpolate(nx, ny, nz-1, h, corner_z, P, Wz);
  Eigen::VectorXd vx = Wx.transpose() * N.col(0);
  Eigen::VectorXd vy = Wy.transpose() * N.col(1);
  Eigen::VectorXd vz = Wz.transpose() * N.col(2);
  Eigen::VectorXd v(vx.rows()+vy.rows()+vz.rows());
  v << vx, vy, vz;

  Eigen::SparseMatrix<double> G;
  fd_grad(nx, ny, nz, h, G);

  Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
  solver.compute(G.transpose() * G);
  g = solver.solve(G.transpose() * v);

  Eigen::SparseMatrix<double> W;
  fd_interpolate(nx, ny, nz, h, corner, P, W);
  Eigen::VectorXd surface = W * g;
  double sigma = surface.sum() / W.rows();
  g.array() = g.array() - sigma;
  ////////////////////////////////////////////////////////////////////////////
  // Run black box algorithm to compute mesh from implicit function: this
  // function always extracts g=0, so "pre-shift" your g values by -sigma
  ////////////////////////////////////////////////////////////////////////////
  igl::copyleft::marching_cubes(g, x, nx, ny, nz, V, F);
}
