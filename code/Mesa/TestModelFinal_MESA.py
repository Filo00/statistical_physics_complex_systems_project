import math
import numpy as np
import solara
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)

class FractalAgent(Agent):
    """An agent representing a particle in fractal growth"""

    def __init__(self, unique_id, model, agent_type="particle"):
        super().__init__(model)
        self.agent_type = agent_type  # "particle", "seed", "walker"
        self.stuck = False
        self.unique_id = unique_id

    def step(self):
        if self.agent_type == "walker" and not self.stuck:
            self.random_walk()
            self.check_aggregation()

    def random_walk(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        # Move only if within bounds
        if (0 <= new_position[0] < self.model.grid.width and
                0 <= new_position[1] < self.model.grid.height):
            self.model.grid.move_agent(self, new_position)

    def check_aggregation(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        for neighbor in neighbors:
            if neighbor.agent_type in ["seed", "particle"] and neighbor.stuck:
                self.stuck = True
                self.agent_type = "particle"
                self.model.aggregated_particles += 1
                break

class FractalModel(Model):
    """Model for fractal growth and dimension estimation"""

    def __init__(self, width=51, height=51, initial_particles=1,
                 walker_rate=0.1, max_walkers=10, fractal_type="DLA"):
        super().__init__()
        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, True)
        self.fractal_type = fractal_type
        self.walker_rate = walker_rate
        self.max_walkers = max_walkers
        self.aggregated_particles = 0
        self.step_count = 0
        self.agent_count = 0

        if fractal_type == "DLA":
            self.create_initial_seed()
        elif fractal_type == "percolation":
            self.create_percolation_sites()
        elif fractal_type == "sierpinski_carpet":
            self.create_sierpinski_carpet()
        elif fractal_type == "sierpinski_triangle":
            self.create_sierpinski_triangle()
        elif fractal_type == "cantor_dust":
            self.create_cantor_dust()
        elif fractal_type == "random_cantor_dust":
            self.create_random_cantor_dust()

        self.datacollector = DataCollector(
            model_reporters={
                "Aggregated_Particles": "aggregated_particles",
                "Box_Counting_Fractal_Dimension": self.calculate_fractal_dimension,
                "Mass_Radius_Fractal_Dimension": self.calculate_mass_radius_dimension,
                "Correlation_Fractal_Dimension": self.calculate_correlation_dimension,
                "Radius_of_Gyration": self.calculate_radius_of_gyration,
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def create_initial_seed(self):
        center_x, center_y = self.width // 2, self.height // 2
        seed_agent = FractalAgent(self.agent_count, self, "seed")
        seed_agent.stuck = True
        self.grid.place_agent(seed_agent, (center_x, center_y))
        self.aggregated_particles = 1
        self.agent_count += 1

    def create_percolation_sites(self, site_probability=0.3):
        for x in range(self.width):
            for y in range(self.height):
                if self.random.random() < site_probability:
                    agent = FractalAgent(self.agent_count, self, "particle")
                    agent.stuck = True
                    self.grid.place_agent(agent, (x, y))
                    self.agent_count += 1
                    self.aggregated_particles += 1

    def create_sierpinski_carpet(self):
        """Create a Sierpinski carpet pattern (deterministic fractal)"""
        def is_carpet(x, y, size):
            while size > 1:
                if (x // (size // 3)) % 3 == 1 and (y // (size // 3)) % 3 == 1:
                    return False
                size //= 3
            return True
        grid_size = min(self.width, self.height)
        for x in range(grid_size):
            for y in range(grid_size):
                if is_carpet(x, y, grid_size):
                    agent = FractalAgent(self.agent_count, self, "particle")
                    agent.stuck = True
                    self.grid.place_agent(agent, (x, y))
                    self.agent_count += 1
                    self.aggregated_particles += 1

    def create_sierpinski_triangle(self):
        """Create a Sierpinski triangle pattern (deterministic fractal)"""
        def is_triangle(x, y, size):
            # Only works for size = 2^n+1 (odd)
            while size > 1:
                if (x & y) != 0:
                    return False
                size //= 2
            return True
        grid_size = min(self.width, self.height)
        for y in range(grid_size):
            for x in range(grid_size):
                # Sierpinski triangle fits in the bottom half
                if y >= x and is_triangle(x, y, grid_size):
                    agent = FractalAgent(self.agent_count, self, "particle")
                    agent.stuck = True
                    self.grid.place_agent(agent, (x, y))
                    self.agent_count += 1
                    self.aggregated_particles += 1

    def create_cantor_dust(self):
        """Create 2D Cantor dust (deterministic, product of two 1D Cantor sets)"""
        def is_cantor(n, size):
            # n: coordinate, size: total size (must be power of 3)
            while size > 1:
                if (n // (size // 3)) % 3 == 1:
                    return False
                size //= 3
            return True
        grid_size = min(self.width, self.height)
        for x in range(grid_size):
            for y in range(grid_size):
                if is_cantor(x, grid_size) and is_cantor(y, grid_size):
                    agent = FractalAgent(self.agent_count, self, "particle")
                    agent.stuck = True
                    self.grid.place_agent(agent, (x, y))
                    self.agent_count += 1
                    self.aggregated_particles += 1

    def create_random_cantor_dust(self, probability=0.7):
        """Create a random 2D Cantor dust (stochastic)"""
        grid_size = min(self.width, self.height)
        for x in range(grid_size):
            for y in range(grid_size):
                if self.random.random() < probability:
                    agent = FractalAgent(self.agent_count, self, "particle")
                    agent.stuck = True
                    self.grid.place_agent(agent, (x, y))
                    self.agent_count += 1
                    self.aggregated_particles += 1

    def step(self):
        self.step_count += 1
        if self.fractal_type == "DLA":
            self.dla_step()
            for agent in list(self.schedule_agents()):
                agent.step()
            self.remove_distant_walkers()
        self.datacollector.collect(self)

    def schedule_agents(self):
        agents = [a for cell in self.grid.coord_iter() for a in (cell[0] if len(cell) == 3 else cell[0])]
        self.random.shuffle(agents)
        return agents

    def get_cluster_radius(self):
        positions = self.get_stuck_positions()
        if not positions:
            return 0
        center_x, center_y = self.width // 2, self.height // 2
        return max(math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in positions)

    def add_random_walker(self):
        center_x, center_y = self.width // 2, self.height // 2
        r = self.get_cluster_radius()
        margin = 5
        max_possible = min(center_x, center_y, self.width - center_x - 1, self.height - center_y - 1)
        radius = int(min(r + margin, max_possible))
        if radius < 1:
            radius = 1
        theta = self.random.uniform(0, 2 * np.pi)
        x = int(round(center_x + radius * np.cos(theta)))
        y = int(round(center_y + radius * np.sin(theta)))
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        walker = FractalAgent(self.agent_count, self, "walker")
        self.grid.place_agent(walker, (x, y))
        self.agent_count += 1

    def remove_distant_walkers(self, margin=8):
        center_x, center_y = self.width // 2, self.height // 2
        cluster_radius = self.get_cluster_radius()
        max_distance = cluster_radius + margin
        to_remove = []
        for agent in self.schedule_agents():
            if agent.agent_type == "walker":
                x, y = agent.pos
                distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance > max_distance:
                    to_remove.append(agent)
        for agent in to_remove:
            self.grid.remove_agent(agent)

    def dla_step(self):
        current_walkers = sum(
            1 for agent in self.schedule_agents() if agent.agent_type == "walker"
        )
        if current_walkers < self.max_walkers:
            if self.random.random() < self.walker_rate:
                self.add_random_walker()

    def get_stuck_positions(self):
        positions = []
        for cell in self.grid.coord_iter():
            if len(cell) == 3:
                cell_content, x, y = cell
            else:
                cell_content, pos = cell
                x, y = pos
            for agent in cell_content:
                if agent.stuck:
                    positions.append((x, y))
        return positions

    def calculate_fractal_dimension(self):
        """Box-Counting (Minkowski–Bouligand) dimension"""
        positions = self.get_stuck_positions()
        N = len(positions)
        if N < 10:
            return 0
        xs, ys = zip(*positions)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        L = max(max_x - min_x + 1, max_y - min_y + 1)
        box_sizes = []
        s = L
        while s >= 2:
            box_sizes.append(s)
            s = s // 2
        if 1 not in box_sizes:
            box_sizes.append(1)
        box_sizes = np.array(box_sizes)
        counts = []
        for box_size in box_sizes:
            boxes = set()
            for x, y in positions:
                bx = (x - min_x) // box_size
                by = (y - min_y) // box_size
                boxes.add((bx, by))
            counts.append(len(boxes))
        box_sizes = np.array(box_sizes)
        counts = np.array(counts)
        mask = (counts > 0) & (box_sizes > 0)
        if np.sum(mask) < 2:
            return 0
        log_box_sizes = np.log(1.0 / box_sizes[mask])
        log_counts = np.log(counts[mask])
        slope, intercept = np.polyfit(log_box_sizes, log_counts, 1)
        return slope

    def calculate_mass_radius_dimension(self):
        """Mass-radius (sandbox) dimension"""
        positions = self.get_stuck_positions()
        N = len(positions)
        if N < 10:
            return 0
        center_x = sum(p[0] for p in positions) / N
        center_y = sum(p[1] for p in positions) / N
        dists = [math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) for x, y in positions]
        max_r = max(dists)
        num_radii = 6
        radii = np.geomspace(1, max_r, num=num_radii)
        masses = []
        for r in radii:
            mass = sum(1 for d in dists if d <= r)
            masses.append(mass)
        radii = np.array(radii)
        masses = np.array(masses)
        mask = (masses > 1)
        if np.sum(mask) < 2:
            return 0
        log_r = np.log(radii[mask])
        log_m = np.log(masses[mask])
        slope, intercept = np.polyfit(log_r, log_m, 1)
        return slope

    def calculate_correlation_dimension(self):
        """Correlation dimension (Grassberger–Procaccia)"""
        positions = self.get_stuck_positions()
        N = len(positions)
        if N < 10:
            return 0
        positions = np.array(positions)
        dmat = np.sqrt(np.sum((positions[None, :, :] - positions[:, None, :]) ** 2, axis=-1))
        dmat = dmat + np.eye(N) * 1e10
        maxdist = np.max(dmat[dmat < 1e9])
        num_radii = 8
        radii = np.geomspace(1, maxdist, num=num_radii)
        corr_sum = []
        for r in radii:
            count = np.sum(dmat <= r)
            corr_sum.append(count / (N * (N - 1)))
        radii = np.array(radii)
        corr_sum = np.array(corr_sum)
        mask = (corr_sum > 0)
        if np.sum(mask) < 2:
            return 0
        log_r = np.log(radii[mask])
        log_C = np.log(corr_sum[mask])
        slope, intercept = np.polyfit(log_r, log_C, 1)
        return slope

    def calculate_radius_of_gyration(self):
        positions = self.get_stuck_positions()
        if len(positions) < 2:
            return 0
        center_x = sum(pos[0] for pos in positions) / len(positions)
        center_y = sum(pos[1] for pos in positions) / len(positions)
        sum_distances_sq = sum((pos[0] - center_x) ** 2 + (pos[1] - center_y) ** 2
                               for pos in positions)
        return math.sqrt(sum_distances_sq / len(positions))

def agent_portrayal(agent):
    if agent.agent_type == "seed":
        return {"color": "black", "size": 15, "marker": "*"}
    elif agent.agent_type == "particle" and agent.stuck:
        return {"color": "red", "size": 10, "marker": "*"}
    elif agent.agent_type == "walker":
        return {"color": "blue", "size": 5, "marker": "*"}
    else:
        return {"color": "gray", "size": 8, "marker": "*"}

def get_fractal_info(model):
    box_dim = model.calculate_fractal_dimension()
    mass_dim = model.calculate_mass_radius_dimension()
    corr_dim = model.calculate_correlation_dimension()
    rg = model.calculate_radius_of_gyration()
    particles = model.aggregated_particles

    box_dim_text = f"{box_dim:.3f}" if box_dim > 0 else "N/A"
    mass_dim_text = f"{mass_dim:.3f}" if mass_dim > 0 else "N/A"
    corr_dim_text = f"{corr_dim:.3f}" if corr_dim > 0 else "N/A"
    rg_text = f"{rg:.2f}" if rg > 0 else "N/A"

    return solara.Markdown(
        f"**Box-counting fractal dimension:** {box_dim_text}<br>"
        f"**Mass-radius fractal dimension:** {mass_dim_text}<br>"
        f"**Correlation fractal dimension:** {corr_dim_text}<br>"
        f"**Radius of Gyration:** {rg_text}<br>"
        f"**Aggregated Particles:** {particles}<br>"
        f"**Step:** {model.step_count}"
    )

model_params = {
    "width": Slider(
        label="Grid Width",
        value=51,
        min=27,
        max=81,
        step=3,
    ),
    "height": Slider(
        label="Grid Height",
        value=51,
        min=27,
        max=81,
        step=3,
    ),
    "fractal_type": {
        "type": "Select",
        "value": "DLA",
        "values": [
            "DLA",
            "percolation",
            "sierpinski_carpet",
            "sierpinski_triangle",
            "cantor_dust",
            "random_cantor_dust",
        ],
    },
    "walker_rate": Slider(
        label="Walker Generation Rate",
        value=0.4,
        min=0.01,
        max=0.5,
        step=0.01,
    ),
    "max_walkers": Slider(
        label="Max Walkers",
        value=10,
        min=1,
        max=50,
        step=1,
    ),
}

def post_process_plot(ax):
    ax.set_ylim(ymin=0)
    ax.set_ylabel("Count/Value")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

SpacePlot = make_space_component(agent_portrayal)
MetricsPlot = make_plot_component(
    {
        "Aggregated_Particles": "red",
        "Box_Counting_Fractal_Dimension": "blue",
        "Mass_Radius_Fractal_Dimension": "cyan",
        "Correlation_Fractal_Dimension": "magenta",
        "Radius_of_Gyration": "green"
    },
    post_process=post_process_plot,
)

model1 = FractalModel()

page = SolaraViz(
    model1,
    components=[
        SpacePlot,
        MetricsPlot,
        get_fractal_info,
    ],
    model_params=model_params,
    name="Fractal Dimension Estimation",
)

# To run: solara run FractalDimensionGUI.py