"""
Algorithm harness for benchmarking path selection algorithms

Supports plugin-based algorithm loading and Monte Carlo evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from abc import ABC, abstractmethod
import pickle
import time
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import importlib
import yaml

from ..path_services.pathfinder_v2 import PathFinderV2 as PathFinder
from ..path_services.pathfinder_v2 import SCIONPath
from ..path_services.pathprobe import PathProbe, PathMetrics


@dataclass
class FlowRequest:
    """Traffic flow request"""
    src: int
    dst: int
    bandwidth_mbps: float
    start_time: int
    duration: int


@dataclass
class FlowResult:
    """Result of scheduling a flow"""
    flow_id: int
    path: Optional[SCIONPath]
    metrics: Optional[PathMetrics]
    success: bool
    algorithm: str
    decision_time_ms: float


class PathSelectionAlgorithm(ABC):
    """Base class for path selection algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def select_path(self, 
                   src: int, 
                   dst: int,
                   available_paths: List[SCIONPath],
                   path_metrics: List[PathMetrics],
                   flow_request: FlowRequest) -> Optional[SCIONPath]:
        """
        Select best path for a flow
        
        Args:
            src: Source AS
            dst: Destination AS
            available_paths: List of feasible paths
            path_metrics: Current metrics for each path
            flow_request: Flow requirements
            
        Returns:
            Selected path or None if no suitable path
        """
        pass
    
    def initialize(self, config: Dict[str, Any]):
        """Initialize algorithm with configuration"""
        pass


class AlgorithmHarness:
    """Harness for running and evaluating algorithms"""
    
    def __init__(self,
                 topology_path: Path,
                 segment_dir: Path,
                 link_table_path: Path,
                 link_metrics_path: Path,
                 metrics_shape: tuple,
                 output_dir: Path):
        """
        Args:
            topology_path: Path to topology pickle
            segment_dir: Segment store directory
            link_table_path: Path to link table
            link_metrics_path: Path to link metrics memmap
            metrics_shape: Shape of metrics array
            output_dir: Output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        seg_path = Path(segment_dir)
        if seg_path.is_dir():
            cand = seg_path / "segments_corrected.pkl"
            seg_path = cand if cand.is_file() else seg_path

        # Initialize path services
        self.path_finder = PathFinder(topology_path, seg_path, link_table_path)
        self.path_probe = PathProbe(link_metrics_path, link_table_path, metrics_shape)
        
        # Load registered algorithms
        self.algorithms = {}
        
    def register_algorithm(self, algorithm: PathSelectionAlgorithm):
        """Register an algorithm for evaluation"""
        self.algorithms[algorithm.name] = algorithm
        
    def load_algorithm_plugin(self, module_path: str, class_name: str, 
                            config: Dict = None):
        """Load algorithm from Python module"""
        module = importlib.import_module(module_path)
        algo_class = getattr(module, class_name)
        algorithm = algo_class()
        
        if config:
            algorithm.initialize(config)
            
        self.register_algorithm(algorithm)
        
    def run_experiment(self, 
                      flow_trace: List[FlowRequest],
                      algorithm_names: List[str],
                      num_seeds: int = 10,
                      n_jobs: int = -1) -> pd.DataFrame:
        """
        Run Monte Carlo experiment
        
        Args:
            flow_trace: List of flow requests
            algorithm_names: Algorithms to evaluate
            num_seeds: Number of random seeds
            n_jobs: Parallel jobs
            
        Returns:
            Results DataFrame
        """
        if n_jobs == -1:
            n_jobs = min(num_seeds, ProcessPoolExecutor()._max_workers)
            
        # Prepare work items
        work_items = []
        for seed in range(num_seeds):
            for algo_name in algorithm_names:
                if algo_name in self.algorithms:
                    work_items.append((seed, algo_name, flow_trace))
                    
        # Run in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(
                self._run_single_experiment, work_items
            ))
            
        # Combine results
        all_results = []
        for result_list in results:
            all_results.extend(result_list)
            
        # Create DataFrame
        df = pd.DataFrame([asdict(r) for r in all_results])
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"results_{timestamp}.parquet"
        df.to_parquet(output_path, index=False)
        
        return df
    
    def _run_single_experiment(self, work_item: tuple) -> List[FlowResult]:
        """Run single experiment instance"""
        seed, algo_name, flow_trace = work_item
        
        # Set random seed
        np.random.seed(seed)
        
        algorithm = self.algorithms[algo_name]
        results = []
        
        for i, flow in enumerate(flow_trace):
            # Find available paths
            start_time = time.time()
            paths = self.path_finder.get_paths(
                flow.src, flow.dst, k=16, policy="min-lat"
            )
            
            if not paths:
                # No paths available
                results.append(FlowResult(
                    flow_id=i,
                    path=None,
                    metrics=None,
                    success=False,
                    algorithm=algo_name,
                    decision_time_ms=0.0
                ))
                continue
                
            # Get current metrics
            metrics = self.path_probe.probe_batch(
                paths, flow.start_time, noisy=True
            )
            
            # Filter by bandwidth requirement
            feasible_paths = []
            feasible_metrics = []
            for path, metric in zip(paths, metrics):
                if metric.bandwidth_mbps >= flow.bandwidth_mbps:
                    feasible_paths.append(path)
                    feasible_metrics.append(metric)
                    
            # Select path
            selected_path = algorithm.select_path(
                flow.src, flow.dst,
                feasible_paths, feasible_metrics,
                flow
            )
            
            decision_time = (time.time() - start_time) * 1000  # ms
            
            if selected_path:
                # Get final metrics
                final_metrics = self.path_probe.probe(
                    selected_path, flow.start_time, noisy=True
                )
                success = final_metrics.bandwidth_mbps >= flow.bandwidth_mbps
            else:
                final_metrics = None
                success = False
                
            results.append(FlowResult(
                flow_id=i,
                path=selected_path,
                metrics=final_metrics,
                success=success,
                algorithm=algo_name,
                decision_time_ms=decision_time
            ))
            
        return results
    
    def generate_flow_trace(self, 
                           topology_path: Path,
                           num_flows: int,
                           time_slots: int) -> List[FlowRequest]:
        """Generate synthetic flow trace"""
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            
        node_df = topology['nodes']
        as_ids = node_df['as_id'].values
        
        flows = []
        for i in range(num_flows):
            src = np.random.choice(as_ids)
            dst = np.random.choice(as_ids)
            
            while dst == src:
                dst = np.random.choice(as_ids)
                
            # Log-normal bandwidth demand
            bandwidth = np.random.lognormal(3.0, 1.5)  # ~20 Mbps median
            bandwidth = np.clip(bandwidth, 1.0, 1000.0)
            
            # Random start time
            start_time = np.random.randint(0, time_slots - 100)
            
            # Exponential duration
            duration = int(np.random.exponential(20))  # ~20 slots average
            duration = np.clip(duration, 1, 100)
            
            flows.append(FlowRequest(
                src=int(src),
                dst=int(dst),
                bandwidth_mbps=float(bandwidth),
                start_time=int(start_time),
                duration=int(duration)
            ))
            
        return flows
    
    def compute_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Compute aggregate performance metrics"""
        metrics = {}
        
        for algo in results_df['algorithm'].unique():
            algo_df = results_df[results_df['algorithm'] == algo]
            
            # Success rate
            success_rate = algo_df['success'].mean()
            
            # Latency stats (for successful flows)
            successful = algo_df[algo_df['success']]
            if len(successful) > 0:
                latencies = [m.latency_ms for m in successful['metrics'] 
                           if m is not None]
                if latencies:
                    p50_latency = np.percentile(latencies, 50)
                    p95_latency = np.percentile(latencies, 95)
                    p99_latency = np.percentile(latencies, 99)
                else:
                    p50_latency = p95_latency = p99_latency = np.nan
            else:
                p50_latency = p95_latency = p99_latency = np.nan
                
            # Decision time
            avg_decision_time = algo_df['decision_time_ms'].mean()
            
            metrics[algo] = {
                'success_rate': success_rate,
                'p50_latency_ms': p50_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'avg_decision_time_ms': avg_decision_time
            }
            
        return metrics


# Example baseline algorithms that can be imported
class ShortestPathAlgorithm(PathSelectionAlgorithm):
    """Select path with minimum hop count"""
    
    def __init__(self):
        super().__init__("shortest_path")
        
    def select_path(self, src, dst, available_paths, path_metrics, flow_request):
        if not available_paths:
            return None
            
        # Sort by hop count
        return min(available_paths, key=lambda p: p.total_hops)


class LowestLatencyAlgorithm(PathSelectionAlgorithm):
    """Select path with lowest latency"""
    
    def __init__(self):
        super().__init__("lowest_latency")
        
    def select_path(self, src, dst, available_paths, path_metrics, flow_request):
        if not available_paths:
            return None
            
        # Sort by latency
        best_idx = np.argmin([m.latency_ms for m in path_metrics])
        return available_paths[best_idx]


class RandomAlgorithm(PathSelectionAlgorithm):
    """Random path selection"""
    
    def __init__(self):
        super().__init__("random")
        
    def select_path(self, src, dst, available_paths, path_metrics, flow_request):
        if not available_paths:
            return None
            
        return np.random.choice(available_paths)