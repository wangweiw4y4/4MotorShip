#ifndef SIM_HPP
#define SIM_HPP

// C++
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

typedef std::vector<double> state_type;
typedef struct ret_type{
  double array[10];
} ret_struct, *ret_pointer;
class Sim
{
private:
  std::vector<double> state;
  double simStep = 0.0001;

  /** default system dynamics parameters **/
  double d11 = 6;    // drag coff in the x direction
  double d22 = 8;    // drag coff in y direction
  double d33 = 0.6;  // drag torque coff
  double m11 = 12;   // mass plus added mass in the x direction
  double m22 = 24;   // mass plus added mass in the y direction
  double m33 = 1.5;  // moment of inertia plus added mass around the z axis
  double aa = 0.9;   // robot length 
  double bb = 0.45;  // robot width

  runge_kutta4<state_type> stepper;
  
public:
  void operator()(const state_type& x, state_type& dxdt, const double /* t */);
  state_type integrate(state_type& x, double time);
  //* 新定义了一个包裹的my_integrate
  ret_pointer my_integrate(double * x, double time);
};

#endif