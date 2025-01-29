import subprocess
from pathlib import Path

from rich import print

from benchmark import build_code

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    build_code(script_dir)

    # Compare cache misses for the two parallel algorithms
    for algorithm in ["parallel", "parallel2"]:
        print(f"[bold magenta]Algorithm: {algorithm}[/bold magenta]")

        res = subprocess.run(
            ["valgrind", "--tool=cachegrind", "./integral_image", "4", algorithm],
            cwd=script_dir.parent / "build",
            stdout=subprocess.PIPE,
        )
        print(res.stdout.decode())
