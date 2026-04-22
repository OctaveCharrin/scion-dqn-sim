"""
BRITE configuration generator

Emits BRITE 2.x Java configuration files (numeric model codes and BeginOutput flags).
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ModelConstants.java (BRITE 2.0)
RT_WAXMAN = 1
AS_WAXMAN = 3
AS_BARABASI = 4
RT_BARABASI2 = 9
AS_BARABASI2 = 10


class BRITEConfigGenerator:
    """Generate BRITE configuration files from YAML templates or defaults."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "model_name": AS_BARABASI,  # AS-level Barabasi-Albert
        "n_nodes": 100,
        "hs": 1000,
        "ls": 100,
        "node_placement": 1,
        "m": 2,
        "bw_dist": 1,
        "bw_min": 10.0,
        "bw_max": 100.0,
        "p": 0.45,
        "q": 0.2,
    }

    def __init__(self, template_path: Optional[Path] = None):
        self.template_path = template_path
        self.config = self.DEFAULT_CONFIG.copy()
        if template_path and template_path.exists():
            with open(template_path) as f:
                user_config = yaml.safe_load(f) or {}
            self.config.update(user_config)

    def generate(self, output_path: Path, **kwargs) -> Path:
        """
        Generate a BRITE .conf file.

        Common kwargs: n_nodes (or legacy alias num_as), model_name, hs, ls, m,
        bw_min, bw_max, bw_dist, p, q (required for AS Barabasi-Albert 2).
        """
        config = self.config.copy()
        config.update(kwargs)
        if "num_as" in config:
            config.setdefault("n_nodes", config["num_as"])

        conf_content = self._format_brite_config(config)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(conf_content)
        return output_path

    def _format_brite_config(self, config: Dict[str, Any]) -> str:
        """BRITE Java parser expects numeric fields (see configs/brite_templates)."""
        model_name = int(config["model_name"])
        n = int(config["n_nodes"])
        hs = int(config["hs"])
        ls = int(config["ls"])
        np_ = int(config["node_placement"])
        m = int(config["m"])
        bw_dist = int(config["bw_dist"])
        bw_min = float(config["bw_min"])
        bw_max = float(config["bw_max"])

        lines = [
            "BriteConfig",
            "",
            "BeginModel",
            f"\tName = {model_name}",
            f"\tN = {n}",
            f"\tHS = {hs}",
            f"\tLS = {ls}",
            f"\tNodePlacement = {np_}",
            f"\tm = {m}",
            f"\tBWDist = {bw_dist}",
            f"\tBWMin = {bw_min}",
            f"\tBWMax = {bw_max}",
        ]
        if model_name in (RT_BARABASI2, AS_BARABASI2):
            lines.append(f"\tp = {float(config['p'])}")
            lines.append(f"\tq = {float(config['q'])}")
        if model_name in (RT_WAXMAN, AS_WAXMAN):
            lines.append(f"\talpha = {float(config.get('alpha', 0.15))}")
            lines.append(f"\tbeta = {float(config.get('beta', 0.2))}")
            lines.append(f"\tGrowthType = {int(config.get('growth_type', 1))}")
        lines.extend(
            [
                "EndModel",
                "",
                "BeginOutput",
                "\tBRITE = 1",
                "\tOTTER = 0",
                "\tDML = 0",
                "\tNS = 0",
                "\tJavasim = 0",
                "EndOutput",
                "",
            ]
        )
        return "\n".join(lines)


try:
    from src.topology.brite_wrapper import BRITEWrapper

    class BRITERunner(BRITEWrapper):
        """Extended BRITE runner with parallel execution support."""

        def run_parallel(self, config_files: list, output_dir: Path, n_jobs: int = -1):
            from joblib import Parallel, delayed
            import multiprocessing

            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()

            def run_single(config_path, output_dir):
                output_name = Path(config_path).stem
                return self.generate_topology(
                    n_nodes=None,
                    model_type=None,
                    output_dir=output_dir,
                    output_name=output_name,
                    config_file=str(config_path),
                )

            return Parallel(n_jobs=n_jobs)(
                delayed(run_single)(cfg, output_dir) for cfg in config_files
            )

except ImportError:

    class BRITERunner:
        """Run BRITE via the bundled JAR (Main.Brite requires config, output stem, seed file)."""

        def __init__(self, brite_path: Optional[Path] = None):
            self.brite_path = Path(brite_path or "external/brite")

        def run_parallel(
            self, config_files: List[Path], output_dir: Path, n_jobs: int = -1
        ) -> List[Path]:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            jar = self.brite_path / "Java" / "Brite.jar"
            seed = self.brite_path / "Java" / "seed_file"
            if not jar.is_file():
                raise FileNotFoundError(f"BRITE jar missing: {jar} (run ./setup_brite.sh)")
            if not seed.is_file():
                raise FileNotFoundError(f"BRITE seed file missing: {seed}")

            results: List[Path] = []
            for cfg in config_files:
                cfg = Path(cfg).resolve()
                out_stem = (output_dir / cfg.stem).resolve()
                cmd = [
                    "java",
                    "-jar",
                    str(jar.resolve()),
                    str(cfg),
                    str(out_stem),
                    str(seed.resolve()),
                ]
                subprocess.run(cmd, cwd=str(self.brite_path.resolve()), check=True)
                results.append(Path(str(out_stem) + ".brite"))
            return results
