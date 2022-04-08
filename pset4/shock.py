"""
This script simulates the evolution of a strong, adiabatic shock in 1-D.
This Command Line Interface (CLI) allows for setting simulation parameters
at runtime, without the need to update the code to try different settings.

The simulation is intended to be for a strong, adiabatic shock, in the
absence of external forces and viscous forces. The shock evolution is
calculated by integrating the mass, momentum, and energy conservation
equations numerically. Integration is performed via donor cell advection,
followed by source application. Weak reflective boundary conditions are used.

Some things to note:
    The ratio of pre-shock and post-shock density for an adiabatic, monatomic
    gas should be 1/4. We don't quite get this numerically (there seems to be
    a bit of overshoot in my solutions), but I suspect that's mainly due to
    errors associated with having a very steep gradient at the shock front.
    
    The shock width is primarily set by the numerical viscosity in the
    simulation. You can test this by varying dx and dt (but make sure that
    your choices give a stable integration scheme!). I have provided some
    plots that show this behavior.

    Since we're working in the pre-shock fluid's rest frame, the pressure is
    initially determined solely from the total energy density. In order to
    simulate a strong shock, we need to ensure that the pressure in the
    perturbed region greatly exceeds the ambient pressure. In other words,
    we just need to make sure the perturbation amplitude is at least ~100
    times bigger than the ambient energy density.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

def advect(field, adv_velocity, dt, dx):
    # Define the flux at the field cell boundaries.
    flux = np.zeros_like(adv_velocity)

    # Now fill it in accounting for directionality of the flow.
    flux[1:-1] = np.where(
        adv_velocity[1:-1] > 0,  # When the flow is to the right,
        adv_velocity[1:-1] * field[:-1],  # the flux is from left to right,
        adv_velocity[1:-1] * field[1:],  # otherwise, it's from right to left.
    )
    
    # Now update the field using the flux at each field cell.
    return field - (dt/dx)*(flux[1:] - flux[:-1])


def calc_velocity(density, momentum_density):
    return momentum_density / density


def calc_pressure(density, momentum_density, energy_density, gamma):
    """Calculate the pressure of an adiabatic fluid."""
    kinetic_energy_density = 0.5 * momentum_density**2 / density
    return (energy_density - kinetic_energy_density) * (gamma-1) / gamma


def calc_gradient(field, dx):
    """
    Calculate the gradient of a field.

    Notes
    -----
    Here, it is assumed that the gradient is to be used as a source term in a
    scheme that enforces reflective boundary conditions. Following the second
    set of numerical methods notes, the terms at the boundary need to have an
    extra factor of two divided out.
    """
    gradient = np.zeros_like(field)
    gradient[1:-1] = field[2:] - field[:-2]  # Central difference
    gradient[0] = field[1] - field[0]  # Right difference.
    gradient[-1] = field[-1] - field[-2]  # Left difference.
    return 0.5 * gradient / dx

parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
parser.add_argument(
    "--n_cells", type=int, default=150, help="Number of spatial points."
)
parser.add_argument(
    "--n_steps", type=int, default=1500, help="Number of integrations."
)
parser.add_argument(
    "--dt", type=float, default=0.007, help="Time elapsed per integration."
)
parser.add_argument(
    "--dx", type=float, default=1.0, help="Length of each cell."
)
parser.add_argument(
    "--gamma", type=float, default=5/3, help="Adiabatic index."
)
parser.add_argument(
    "--background_density", type=float, default=1, help="Pre-shock density."
)
parser.add_argument(
    "--pert_amp", type=float, default=1e3,
    help="Amplitude of energy density perturbation.",
)
parser.add_argument(
    "--snapshot_fn", type=str, default=None, help="Where to write snapshot."
)

if __name__ == "__main__":
    args = parser.parse_args()
    # Setup the basic simulation parameters.
    n_cells = args.n_cells
    n_steps = args.n_steps
    dt = args.dt
    dx = args.dx
    gamma = args.gamma

    # Initialize the relevant fields.
    background_density = args.background_density
    density = background_density * np.ones(n_cells, dtype=float)
    momentum_density = np.zeros(n_cells, dtype=float)
    adv_velocity = np.zeros(n_cells + 1, dtype=float)
    velocity = np.zeros(n_cells, dtype=float)

    # Use a Gaussian perturbation to initialize the energy density.
    grid = np.arange(n_cells, dtype=float) * dx
    amp = args.pert_amp
    x0 = np.mean(grid)
    var = 0.5 * x0
    energy_density = amp * np.exp(-0.5 * (grid-x0)**2 / var)
    plot_grid = grid - np.mean(grid)
    n_ticks = 11
    x_ticks = np.around(
        np.linspace(plot_grid[0], plot_grid[-1], n_ticks), 0
    )

    # Calculate the initial Mach number.
    pressure = calc_pressure(density, momentum_density, energy_density, gamma)
    sound_speed = np.sqrt(gamma * pressure / density)
    velocity = calc_velocity(density, momentum_density)
    mach_number = velocity / sound_speed

    # Initialize the plots.
    plt.ion()
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7,6),
        sharex=True,
        dpi=150,
        facecolor="white",
        gridspec_kw=dict(hspace=0.075),
    )
    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("Mach Number")
    axes[1].set_xlabel("Position")
    axes[0].set_title("Time Elapsed: 0")
    axes[0].set_ylim(0, 5 * background_density)
    axes[1].set_ylim(-4, 4)
    axes[0].axhline(
        background_density * (gamma+1) / (gamma-1),
        color="gray",
        ls=":",
        label=r"$\frac{\gamma+1}{\gamma-1}\rho_0$",
    )
    axes[0].legend(loc="upper center")
    
    if args.snapshot_fn is not None:
        info = f"dx = {dx:.2f}\ndt = {dt:.4f}\n"
        info += r"$P_0/P_{\rm bg}$ =" + f" {amp:.2e}"
        axes[1].text(
            0.95,
            0.05,
            info,
            va="bottom",
            ha="right",
            transform=axes[1].transAxes,
            bbox=dict(facecolor="w", alpha=1),
        )

    for ax in axes:
        ax.set_xticks(x_ticks)
        ax.grid(True)

    density_plot, = axes[0].plot(plot_grid, density, "k.", ms=2)
    mach_plot, = axes[1].plot(plot_grid, mach_number, "k.", ms=2)
    mach_text = axes[1].text(
        0.95,
        0.95,
        r"${\cal M}_{\rm max}$ = " + f"{mach_number.max():.2f}",
        ha="right",
        va="top",
        transform=axes[1].transAxes,
    )

    # Start integrating.
    elapsed_time = 0
    for _ in range(n_steps):
        # Update the advection velocity.
        adv_velocity[1:-1] = 0.5 * (velocity[1:] + velocity[:-1])
        
        # Advect the density and momentum density.
        density = advect(density, adv_velocity, dt, dx)
        momentum_density = advect(momentum_density, adv_velocity, dt, dx)

        # Update the pressure, then use it as a source for momentum density.
        pressure = calc_pressure(
            density, momentum_density, energy_density, gamma
        )
        momentum_density -= dt * calc_gradient(pressure, dx)

        # Update the velocity and advection velocity.
        velocity = calc_velocity(density, momentum_density)
        adv_velocity[1:-1] = 0.5 * (velocity[1:] + velocity[:-1])
        
        # Advect the energy density.
        energy_density = advect(energy_density, adv_velocity, dt, dx)

        # Update the pressure and use it as a source for energy density.
        pressure = calc_pressure(
            density, momentum_density, energy_density, gamma
        )
        energy_density -= dt * calc_gradient(pressure * velocity, dx)

        # Calculate the new Mach number.
        pressure = calc_pressure(
            density, momentum_density, energy_density, gamma
        )
        sound_speed = np.sqrt(gamma * pressure / density)
        mach_number = velocity / sound_speed
        
        # Update the plots.
        elapsed_time += dt
        axes[0].set_title(f"Elapsed Time: {elapsed_time:.2f}")
        density_plot.set_ydata(density)
        mach_plot.set_ydata(mach_number)
        mach_text.set_text(
            r"${\cal M}_{\rm max}$ = " + f"{mach_number.max():.2f}"
        )
        plt.pause(0.0001)

        # Figure out whether or not to take a snapshot.
        if args.snapshot_fn is not None:
            peak_loc = np.argmax(density[plot_grid >= 0])
            # Ensure different snapshots are taken at nearly the same place.
            if np.abs(peak_loc - n_cells/3) < 1:
                fig.savefig(args.snapshot_fn, dpi=150, bbox_inches="tight")
                args.snapshot_fn = None
