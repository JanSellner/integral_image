import subprocess
from pathlib import Path

import pandas as pd
from rich.progress import Progress


def build_code(script_dir: Path):
    # Compile the latest version
    res = subprocess.run(f"cmake --build {script_dir.parent / 'build'} --config Release -j 34", shell=True)
    assert res.returncode == 0, "Build failed"


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    build_code(script_dir)

    n_repeats = 10
    scalings = [4, 6, 8]
    algorithms = ["serial", "parallel", "parallel2", "opencv", "torch"]

    # Perform the benchmark
    with Progress(auto_refresh=False) as progress:
        task = progress.add_task("Configurations...", total=len(scalings) * len(algorithms))

        results = []
        for scale in scalings:
            for algorithm in algorithms:
                times = []
                for _ in range(n_repeats):
                    res = subprocess.run(
                        ["./integral_image", str(scale), algorithm],
                        cwd=script_dir.parent / "build",
                        stdout=subprocess.PIPE,
                    )
                    times.append(int(res.stdout.decode()))

                results.append({
                    "scale": scale,
                    "algorithm": algorithm,
                    "min_time": min(times),
                })
                progress.update(task, advance=1)
                progress.refresh()

    df = pd.DataFrame(results)
    df.to_csv(script_dir.parent / "results" / "benchmark.csv", index=False)

    # Results can be visualized via the VisualizeBenchmark notebook
