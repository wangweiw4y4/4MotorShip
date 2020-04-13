// C++
#include <boost/numeric/odeint.hpp>
#include <boost/ref.hpp>
#include "math.h"
#include <random>
#include "Sim.hpp"

using namespace boost::numeric::odeint;

void Sim::operator()(const state_type &x, state_type &dxdt, const double /* t */)
{
  dxdt[0] = cos(x[2]) * x[3] - sin(x[2]) * x[4];
  dxdt[1] = sin(x[2]) * x[3] + cos(x[2]) * x[4];
  dxdt[2] = x[5];
  dxdt[3] = -d11 / m11 * x[3] + x[6] / m11 + x[7] / m11;
  dxdt[4] = -d22 / m22 * x[4] + x[8] / m22 + x[9] / m22;
  dxdt[5] = -d33 / m33 * x[5] + aa / (2 * m33) * x[6] - aa / (2 * m33) * x[7] + bb / (2 * m33) * x[8] - bb / (2 * m33) * x[9];
  dxdt[6] = 0;
  dxdt[7] = 0;
  dxdt[8] = 0;
  dxdt[9] = 0;
}

state_type Sim::integrate(state_type &x, double time)
{
  //std::cout<<"function integrate invoked"<<std::endl;
  integrate_const(stepper, boost::ref(*this), x, 0.0, time, simStep);
  //std::cout<<"integrate calc finished"<<std::endl;
  return x;
}

ret_pointer Sim::my_integrate(double * x, double time){
  // 输入array转vector
  state_type vx = std::vector<double>();
  for(int i = 0 ; i < 10 ; i ++){
    vx.push_back(x[i]);
  }

  // vector输入传给 integrate
  state_type resultVec = integrate(vx, time);
  
  // 创建结构体并赋值
  ret_pointer  p = (ret_pointer)malloc(sizeof(ret_struct)); 
  for(int i = 0 ; i < 10 ; i ++){
    p->array[i] = resultVec[i];
  }

  // 返回结构体指针（而不是结构体）
  return p;
}
extern "C"
{
  Sim obj;
  state_type integrate(state_type &px, double ptime)
  {
    return obj.integrate(px, ptime);
  }
  ret_pointer my_integrate(double* x, double ptime)
  {
    return obj.my_integrate(x, ptime);
  }
}