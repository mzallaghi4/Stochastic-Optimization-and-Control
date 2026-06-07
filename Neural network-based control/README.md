# Neural Network-based Control Solver
This repository implements neural network-based methods to solve control problems, using frameworks:

- **Physics-Informed Neural Networks (PINNs)**  
- **Deep Neural Operator Framework (DeepONet)**  
- **Physics-Informed Neural Operators (PINO)**

Supports:
- Optimal Control
- Stochastic Control
- Hamilton-Jacobi-Bellman (HJB) Equations
- Operator Learning
- Scientific Machine Learning

Built with PyTorch.


### Physics-Informed Neural Networks

PINNs approximate the value function and control policy using neural networks while satisfying the HJB PDE constraints and the stochastic dynamics. 


- PINNs implementation
- 
PINN_Control.ipynb


### Deep Neural Operator Framework
DeepONet Architecture
- Maps control function u(x) to system state y(x)

Learns operators rather than individual solutions.


### Physics-Informed Neural Operators

PINO combines: Operator Learning with Physics Constraints








