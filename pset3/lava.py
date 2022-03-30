"""This script simulates a viscous lava flow down a slope.

This simulation makes the following assumptions:
    - no-slip boundary condition at the slope-lava interface
    - no-stress boundary condition at the air-lava interface
    - hydrostatic equilibrium perpendicular to the slope
    - incompressible flow
    - no pressure gradient along the flow

The problem is solved using a 1-dimensional diffusion code that uses an
implicit scheme to handle diffusion. Gravity is applied as a constant
forcing term at each integration.

Note that the numerical solution slightly overshoots the steady-state solution
at the top of the flow near the end of the simulation. This is partially due
to making a compromise on the time step size for reduced compute time (i.e.
the error at the end decreases as the step size is made smaller), but could
potentially be partially due to numerical errors in solving the matrix
equation to actually perform the update.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants, units

if __name__ == "__main__":
    # Set simulation parameters.
    flow_height = 5  # cm
    viscosity = 2.5e3  # cm^2/s
    inclination = 10 * units.deg.to("rad")
    g_eff = constants.g0.to("cm/s^2").value * np.sin(inclination)
    grid_size = 200
    n_steps = 1000
    grid_resolution = flow_height / (grid_size - 1)
    max_courant_number = 0.001
    max_speed = g_eff * flow_height**2 / (2 * viscosity)
    time_step = max_courant_number * grid_resolution / max_speed
    beta = viscosity * time_step / grid_resolution**2

    # Construct the weight matrix.
    weights = (
        (1 + 2*beta) * np.eye(grid_size)
        - beta * np.eye(grid_size, k=1)
        - beta * np.eye(grid_size, k=-1)
    )
    weights[0,0] = 1  # No-slip condition.
    weights[0,1] = 0
    weights[-1,-1] = 1 + beta  # No-stress condition.

    # Initialize the flow and write down the steady state solution.
    velocity_field = np.zeros(grid_size, dtype=float)
    grid = np.arange(grid_size, dtype=float) * grid_resolution
    steady_state_flow = max_speed * (
        2 * grid / flow_height - (grid / flow_height)**2
    )

    # Initialize the animation.
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(6,6), dpi=150)
    ax.plot(grid, steady_state_flow, "k:", label="steady state")
    plot, = ax.plot(grid, velocity_field, label="numerical solution")
    ax.legend(loc="upper left")
    time_info = ax.text(
        0.01, 0.8, "Time Elapsed: 0.000 ms",
        ha="left", va="bottom", transform=ax.transAxes,
    )
    ax.set_xlabel("Height Above Plane (cm)")
    ax.set_ylabel("Flow Velocity (cm/s)")
    plt.pause(1e-6)

    # Start the time evolution.
    for step in range(n_steps):
        # Enforce the no-slip boundary condition.
        velocity_field[0] = 0

        # Apply diffusion.
        velocity_field = np.linalg.solve(weights, velocity_field)

        # Apply gravity.
        velocity_field += g_eff * time_step

        # Update the animation.
        plot.set_ydata(velocity_field)
        time_info.set_text(f"Time Elapsed: {step * time_step * 1e3:.3f} ms")
        plt.pause(1e-6)
