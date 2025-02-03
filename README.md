# Integral Image

This project compares several algorithms to compute the integral image

$$
I_{\int}(x, y) = \sum_{i=0}^{i\leq x} \sum_{j=0}^{j\leq y} I(i, j).
$$

At every location $(x, y)$, the integral image $I_{\int}(x, y)$ contains the sum of values of the original image $I(i, j)$ spanned by the rectangle from the top left corner $(0, 0)$ to the point $(x, y)$. Focus of this project is on a comparison of serial and parallel algorithms and their efficiency. More specifically, the following algorithms are compared with each other:

-   A serial version which iterates once over the image.
-   A parallel version which computes the cumulative row and column sums in parallel (`parallel_cache`).
-   A second parallel version to highlight the importance of cache locality in the parallel version (`parallel_naive`).
-   The vectorized version [`cv::integral()`](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga97b87bec26908237e8ba0f6e96d23e28) of OpenCV.
-   A torch version simply doing `x.cumsum(0).cumsum(1)`.
-   A CUDA version similar to the parallel version with one thread operating over each row/column.

## Setup

The easiest option to run the benchmark is to use the [Visual Studio Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) extension. Simply open this repository folder in Visual Studio Code, hit `F1` and select `Dev Containers: Open Folder in Container...`. This should open a new window and build the Docker container. For the first time, it may take a while until all dependencies are ready.

<details closed>
<summary>Manual setup</summary>
Alternatively, you can also build and run the Docker container yourself:

```bash
# Build the container
docker build --file .devcontainer/Dockerfile --network host --tag integral_image .

# Run the container and mount the current working directory
docker run --rm --gpus all -it -v ${PWD}:/workspaces/integral_image integral_image bash
```

</details>

## Run the Benchmark

Inside the container, run the [`benchmark.py`](./src/benchmark.py) script to start the benchmark. This will run all algorithms with different image sizes. The results will be stored in `results/benchmark.csv` and can be visualized via the [`VisualizeBenchmark.ipynb`](./src/VisualizeBenchmark.ipynb) notebook.

## Cache Misses

The two parallel algorithms have a different cache behavior. In general, there are more cache misses with `parallel_naive` than with `parallel_cache`. This can be made explicit via the [cachegrind tool of valgrind](https://valgrind.org/docs/manual/cg-manual.html). Run the [`cache_misses.py`](./src/cache_misses.py) script to get the output of the tool for those two algorithms.

## Tests

The tests are defined in [`tests.cpp`](./src/tests.cpp) and can be run via the following commands:

```bash
cmake --build /workspaces/integral_image/build --config Release -j 34
ctest --test-dir /workspaces/integral_image/build
```
