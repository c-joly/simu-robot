# simu-robot
Simulation of joint robot

## Commands of the simulator

  - `F`: toggle free space display
  - `A`: starts automatic scanning of the free space
  - `T`: shows the precomputed trajectory (open loop with good inputs / initial condition)
  - `C`: shows a closed loop trajectory starting from the current robot position (take some time to compute the sol of `solve_ivp`)

  Note: the 3 options for displaying trajectory are mutually exclusive. Activating one with deactivate the current one. 

