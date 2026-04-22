#!/usr/bin/env python3
"""
SCION Simulation Platform CLI

Main command-line interface for running simulations.
"""

import typer
import yaml
from pathlib import Path
from typing import Optional, List
import time
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .topology.brite_cfg_gen import BRITEConfigGenerator, BRITERunner
from .topology.brite2scion_converter import BRITE2SCIONConverter
from .link_annotation.capacity_delay_builder import CapacityDelayBuilder
from .traffic.traffic_engine import TrafficEngine, LinkMetricBuilder
from .beacon.beacon_sim_v2 import CorrectedBeaconSimulator
from .harness.algo_harness import (
    AlgorithmHarness, ShortestPathAlgorithm, 
    LowestLatencyAlgorithm, RandomAlgorithm
)
from .visualization.topology_visualizer import (
    TopologyVisualizer, create_topology_report
)

app = typer.Typer(help="SCION AS-Level Simulation Platform")
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@app.command()
def generate(
    n_ases: int = typer.Option(100, help="Number of ASes"),
    base_dir: Path = typer.Option("./experiments", help="Base output directory"),
    config: Path = typer.Option(None, help="Configuration file"),
    name: str = typer.Option(None, help="Experiment name (defaults to timestamp)")
):
    """Generate SCION topology"""
    
    # Create timestamped directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if name:
        dir_name = f"{timestamp}_{name}_n{n_ases}"
    else:
        dir_name = f"{timestamp}_topology_n{n_ases}"
    
    output_dir = base_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'n_ases': n_ases,
        'name': name or 'topology',
        'command': 'generate'
    }
    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Load config
        if config:
            with open(config) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}
            
        # Generate BRITE config
        task = progress.add_task("Generating BRITE configuration...", total=1)
        brite_gen = BRITEConfigGenerator()
        brite_conf = brite_gen.generate(
            output_dir / "topology.conf",
            n_nodes=n_ases
        )
        progress.advance(task)
        
        # Run BRITE
        task = progress.add_task("Running BRITE topology generator...", total=1)
        runner = BRITERunner()
        brite_files = runner.run_parallel(
            [brite_conf], output_dir, n_jobs=1
        )
        progress.advance(task)
        
        # Convert to SCION
        task = progress.add_task("Converting to SCION topology...", total=1)
        converter = BRITE2SCIONConverter(
            n_isds=cfg.get('n_isds', 3),
            core_ratio=cfg.get('core_ratio', 0.1)
        )
        topology = converter.convert(
            brite_files[0], 
            output_dir / "topology.pkl"
        )
        progress.advance(task)
        
        # Annotate links
        task = progress.add_task("Annotating link capacities and delays...", total=1)
        builder = CapacityDelayBuilder()
        link_table = builder.annotate(
            output_dir / "topology.pkl",
            output_dir / "link_table.parquet"
        )
        progress.advance(task)
        
    # Print summary
    table = Table(title="Topology Generation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total ASes", str(topology['metadata']['n_nodes']))
    table.add_row("Total Links", str(topology['metadata']['n_edges']))
    table.add_row("ISDs", str(topology['metadata']['n_isds']))
    table.add_row("Core ASes", str(topology['metadata']['n_core_ases']))
    
    console.print(table)


@app.command()
def simulate(
    topology_dir: Path = typer.Option(None, help="Topology directory (latest if not specified)"),
    base_dir: Path = typer.Option("./experiments", help="Base experiments directory"),
    horizon_days: int = typer.Option(30, help="Simulation horizon in days"),
    slot_minutes: int = typer.Option(5, help="Time slot duration in minutes")
):
    """Run full simulation pipeline"""
    
    # Find topology directory if not specified
    if topology_dir is None:
        # Find most recent topology directory
        topology_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and 'topology' in d.name])
        if not topology_dirs:
            console.print("[red]No topology directories found. Run 'generate' first.[/red]")
            raise typer.Abort()
        topology_dir = topology_dirs[-1]
        console.print(f"Using topology: {topology_dir.name}")
    
    # Create results subdirectory
    output_dir = topology_dir / "simulation_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Generate traffic
        task = progress.add_task("Generating traffic matrices...", total=1)
        traffic_engine = TrafficEngine(
            slot_duration_min=slot_minutes,
            horizon_days=horizon_days
        )
        traffic_path = output_dir / "traffic_TM.memmap"
        traffic_matrix = traffic_engine.generate(
            topology_dir / "topology.pkl",
            traffic_path
        )
        progress.advance(task)
        
        # Build link metrics
        task = progress.add_task("Computing link metrics...", total=1)
        metric_builder = LinkMetricBuilder()
        metrics_path = output_dir / "link_metrics.memmap"
        
        n_slots = (horizon_days * 24 * 60) // slot_minutes
        with open(topology_dir / "topology.pkl", 'rb') as f:
            import pickle
            topo = pickle.load(f)
        n_nodes = topo['metadata']['n_nodes']
        
        link_metrics = metric_builder.build(
            traffic_path,
            topology_dir / "link_table.parquet",
            topology_dir / "topology.pkl",
            metrics_path
        )
        progress.advance(task)
        
        # Run beaconing (v2 simulator writes segment pickles under output_dir)
        task = progress.add_task("Simulating SCION beaconing...", total=1)
        segment_dir = output_dir / "segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        beacon_sim = CorrectedBeaconSimulator()
        _segment_store, stats = beacon_sim.simulate(
            topology_dir / "topology.pkl",
            segment_dir,
        )
        progress.advance(task)
        
    # Print summary
    table = Table(title="Simulation Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Traffic Generation", "✓ Complete")
    table.add_row("Link Metrics", "✓ Complete")
    table.add_row("Beaconing", "✓ Complete")
    table.add_row(
        "Core Segments",
        str(stats.get("totals", {}).get("core_segments", stats.get("core_segments", 0))),
    )
    
    console.print(table)


@app.command()
def benchmark(
    topology_dir: Path = typer.Option("./output", help="Topology directory"),
    results_dir: Path = typer.Option("./results", help="Results directory"),
    num_flows: int = typer.Option(1000, help="Number of flows to simulate"),
    algorithms: List[str] = typer.Option(["shortest_path", "lowest_latency"], help="Algorithms to test"),
    num_seeds: int = typer.Option(10, help="Number of random seeds")
):
    """Benchmark path selection algorithms"""
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics shape
    import pickle
    with open(topology_dir / "topology.pkl", 'rb') as f:
        topo = pickle.load(f)
    n_nodes = topo['metadata']['n_nodes']
    
    # Infer metrics shape
    import pandas as pd
    link_table = pd.read_parquet(topology_dir / "link_table.parquet")
    n_edges = len(link_table)
    n_slots = 8640  # 30 days * 24 hours * 60 min / 5 min
    metrics_shape = (n_slots, n_edges, 3)
    
    # Initialize harness
    harness = AlgorithmHarness(
        topology_dir / "topology.pkl",
        results_dir / "segments",
        topology_dir / "link_table.parquet",
        results_dir / "link_metrics.memmap",
        metrics_shape,
        results_dir / "benchmark"
    )
    
    # Register algorithms
    if "shortest_path" in algorithms:
        harness.register_algorithm(ShortestPathAlgorithm())
    if "lowest_latency" in algorithms:
        harness.register_algorithm(LowestLatencyAlgorithm())
    if "random" in algorithms:
        harness.register_algorithm(RandomAlgorithm())
        
    # Generate flow trace
    console.print("Generating flow trace...")
    flows = harness.generate_flow_trace(
        topology_dir / "topology.pkl",
        num_flows,
        n_slots
    )
    
    # Run experiment
    with console.status("Running benchmarks..."):
        results_df = harness.run_experiment(
            flows, algorithms, num_seeds
        )
        
    # Compute metrics
    metrics = harness.compute_metrics(results_df)
    
    # Display results
    table = Table(title="Algorithm Performance")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Success Rate", style="green")
    table.add_column("P50 Latency (ms)", style="yellow")
    table.add_column("P95 Latency (ms)", style="yellow")
    table.add_column("Avg Decision Time (ms)", style="magenta")
    
    for algo, m in metrics.items():
        table.add_row(
            algo,
            f"{m['success_rate']:.2%}",
            f"{m['p50_latency_ms']:.1f}",
            f"{m['p95_latency_ms']:.1f}",
            f"{m['avg_decision_time_ms']:.2f}"
        )
        
    console.print(table)


@app.command()
def clean(
    output_dir: Path = typer.Option("./output", help="Output directory"),
    results_dir: Path = typer.Option("./results", help="Results directory"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation")
):
    """Clean generated files"""
    
    dirs_to_clean = [output_dir, results_dir]
    
    if not force:
        confirm = typer.confirm(
            f"This will delete {output_dir} and {results_dir}. Continue?"
        )
        if not confirm:
            raise typer.Abort()
            
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            import shutil
            shutil.rmtree(dir_path)
            console.print(f"[red]Deleted {dir_path}")
            
    console.print("[green]Cleanup complete!")


@app.command()
def visualize(
    topology_dir: Path = typer.Option(None, help="Topology directory (latest if not specified)"),
    base_dir: Path = typer.Option("./experiments", help="Base experiments directory"),
    output_format: str = typer.Option("png", help="Output format (png, pdf, svg)")
):
    """Create visualization of topology"""
    
    # Find topology directory if not specified
    if topology_dir is None:
        topology_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and 'topology' in d.name])
        if not topology_dirs:
            console.print("[red]No topology directories found. Run 'generate' first.[/red]")
            raise typer.Abort()
        topology_dir = topology_dirs[-1]
        console.print(f"Using topology: {topology_dir.name}")
    
    # Check if topology file exists
    topology_file = topology_dir / "topology.pkl"
    if not topology_file.exists():
        console.print(f"[red]Topology file not found: {topology_file}[/red]")
        raise typer.Abort()
        
    # Create visualizations
    vis_dir = topology_dir / "visualizations"
    
    with console.status("Creating visualizations..."):
        create_topology_report(topology_file, vis_dir)
        
    # List created files
    console.print(f"\n[green]Visualizations created in {vis_dir}:[/green]")
    for file in sorted(vis_dir.iterdir()):
        console.print(f"  - {file.name}")


if __name__ == "__main__":
    app()