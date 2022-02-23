"""A script for simulating vortex ring interactions in two dimensions.

This script models each vortex as a line vortex (i.e. azimuthally-symmetric
flow with a 1/r dependence around each vortex). Constant circulation is
assumed throughout each ring, with the parity flipped between the "upper"
and "lower" cross-sections. The rings are situated to move from left-to-right.

The vortex rings are assumed to exist in an inviscid, incompressible fluid,
so the system evolves purely through advection. The rings are also assumed
to be rotationally-symmetric about the x-axis.

Note: Some of the generality in my original solution has been omitted here
to make the solution easier to understand. Some python tricks are also
omitted for the same reason.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
plt.ion()


def calculate_velocity(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    vortex_positions: np.ndarray,
    vortex_circulations: np.ndarray,
    mask_radius: Optional[float] = None,
    fill_value: Optional[float] = np.nan,
) -> np.ndarray:
    """Calculate the velocity at the provided locations due to the vortices.

    Parameters
    ----------
    x_coordinates
        x-value for each place we want to calculate the flow velocity.
    y_coordinates
        y-value for each place we want to calculate the flow velocity.
    vortex_positions
        Where each vortex is located. Rows correspond to vortices, columns
        correspond to x-/y-components.
    vortex_circulations
        Value of the "k" parameter for each vortex.
    mask_radius
        How big of a region around each vortex should be masked.
    fill_value
        What to insert into the masked region.
    
    Returns
    -------
    x_velocities
        Velocity in the x-direction at each point.
    y_velocities
        Velocity in the y-direction at each point.

    Notes
    -----
    In Cartesian coordinates (x,y), a line vortex at the origin creates a flow
        u_x = -k * y / r^2,
        u_y = k * x / r^2,
    where r is the distance to the point (x,y). Also note that the circulation
    of a line vortex is 2*pi times the k parameter.
    """
    x_velocities = np.zeros_like(x_coordinates, dtype=float)
    y_velocities = np.zeros_like(y_coordinates, dtype=float)
    k_values = vortex_circulations / (2 * np.pi)
    for vortex_position, k_value in zip(vortex_positions, k_values):
        # Calculate the relative positions.
        delta_x = x_coordinates - vortex_position[0]
        delta_y = y_coordinates - vortex_position[1]
        squared_distance = delta_x**2 + delta_y**2

        # Use a filter to avoid divide-by-zero issues.
        ok_points = squared_distance != 0
        x_velocities[ok_points] -= (
            delta_y[ok_points] * k_value / squared_distance[ok_points]
        )
        y_velocities[ok_points] += (
            delta_x[ok_points] * k_value / squared_distance[ok_points]
        )
      
        # Apply the mask, if requested.
        if mask_radius is not None:
            mask = squared_distance <= mask_radius**2
            x_velocities[mask] = fill_value
            y_velocities[mask] = fill_value

    return x_velocities, y_velocities


# Actually run the script if we're in the right namespace.
if __name__ == "__main__":
    # Set up the initial vortex parameters.
    vortex_positions = np.array(
        [
            [-150, 50],  # upper-left
            [-150, -50],  # lower-left
            [-130, 50],  # upper-right
            [-130, -50],  # lower-right
        ],
        dtype=float,
    )
    circulation_amp = 2 * np.pi
    circulations = circulation_amp * np.array([1, -1, 1, -1], dtype=float)

    # Set up the mesh.
    n_pix_per_side = 200
    grid_resolution = 1  # cm
    base_grid = np.arange(
        -n_pix_per_side, n_pix_per_side+1, dtype=float
    ) * grid_resolution
    x_mesh, y_mesh = np.meshgrid(base_grid, base_grid, indexing="xy")

    # Set up the integration parameters.
    # See the following Wiki for details on how I'm choosing the time step:
    # https://en.wikipedia.org/wiki/Courant%E2%80%93Friedrichs%E2%80%93Lewy_condition
    max_courant_number = 100  # Typically want this to be 1, but we're safe here.
    approx_max_flow_velocity = np.abs(circulations).max() / grid_resolution
    time_step = max_courant_number * grid_resolution / approx_max_flow_velocity
    n_steps = 100

    # Choose a masking scale for the streamline visualization.
    r_mask = 5  # cm
    
    # Calculate the initial velocity field.
    x_velocity, y_velocity = calculate_velocity(
        x_coordinates=x_mesh,
        y_coordinates=y_mesh,
        vortex_positions=vortex_positions,
        vortex_circulations=circulations,
        mask_radius=r_mask,
        fill_value=np.nan,
    )
    flow_speed = np.sqrt(x_velocity**2 + y_velocity**2)

    # Setup the plot.
    fig, ax = plt.subplots(1, 1, figsize=(8,8), dpi=150)
    ax.set_xlim(x_mesh.min(), x_mesh.max())
    ax.set_ylim(y_mesh.min(), y_mesh.max())
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Make an initial plot of the vortices and streamlines.
    plot, = ax.plot(
        vortex_positions[:,0], vortex_positions[:,1], "k+", markersize=10
    )
    colors = np.log10(flow_speed)
    ax.streamplot(
        x_mesh, y_mesh, x_velocity, y_velocity, density=[1,1], color=colors
    )

    # Evolve the system in time.
    for _ in range(n_steps):
        # Calculate the velocity at each vortex position.
        vortex_velocities = calculate_velocity(
            x_coordinates=vortex_positions[:,0],
            y_coordinates=vortex_positions[:,1],
            vortex_positions=vortex_positions,
            vortex_circulations=circulations,
            mask_radius=None,
        )

        # Update the vortex positions using a linear approximation.
        vortex_positions[:,0] += vortex_velocities[0] * time_step
        vortex_positions[:,1] += vortex_velocities[1] * time_step

        # Calculate the new velocity field.
        x_velocity, y_velocity = calculate_velocity(
            x_coordinates=x_mesh,
            y_coordinates=y_mesh,
            vortex_positions=vortex_positions,
            vortex_circulations=circulations,
            mask_radius=r_mask,
            fill_value=np.nan,
        )
        flow_speed = np.sqrt(x_velocity**2 + y_velocity**2)

        # Clear the old plot.
        ax.collections = []
        ax.patches = []

        # Update the plot with the new state.
        plot.set_xdata(vortex_positions[:,0])
        plot.set_ydata(vortex_positions[:,1])
        colors = np.log10(flow_speed)
        ax.streamplot(
            x_mesh, y_mesh, x_velocity, y_velocity, density=[1,1], color=colors
        )
        plt.pause(1e-6)  # Wait a little bit before doing the next step.
