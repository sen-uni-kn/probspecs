# VCAS Experiment

## ProbSpecs VCAS Experiment

Steps to reproduce the ``probspecs`` VCAS experiment:
1.  Place the directory containing this file in the `experiments` directory of the `probspecs` repository.
2. Open a terminal in the `probspecs` repository.
3. Download the VCAS model using the following commands:
    ```bash
    mkdir resources/vcas
    wget -O resources/vcas/VCAS1.onnx https://raw.githubusercontent.com/Zhang-Xiyue/PreimageApproxForNNs/2cc0dc47e447b83f1e18626272a75d5e16059f12/model_dir/VertCAS_1.onnx
    ```
4. Install additional python dependencies:
    ```bash
    pip install git+https://github.com/Verified-Intelligence/onnx2pytorch.git@8447c42c3192dad383e5598edc74dddac5706ee2
    ```
5. Run the experiment:
    ```bash
    ./experiments/vcas/run.sh --log
    ```

## PreimageApproxForNNs VCAS Experiment

Steps to reproduce the ``PreimageApproxForNNs`` VCAS experiment:
1. Install docker: https://docs.docker.com/engine/install/ubuntu/
2. Navigate to ``probspecs/experiments/vcas/PreimageApproxForNNs``
3. Build the docker image: `docker build -t preimage .`
4. Run the PreimageApproxForNNs VCAS experiment using the docker image: `docker run preimage`

Note that this only reports an estimated (unsound) lower bound. 
This bound has to be validated manually (the source code of `PreimageApproxForNNs`, for example, contains absolute paths containing an author name for this part of the code). 
The runtime reported by PreimageApproxForNNs does not include the time needed to validate the bound.

## Polytope Volume Computation Runtime

To get an idea, how much time it would take to validate the bounds by computing the exact volume of a union of polytopes, we provide a small script that uses `timeit` to measure how long it takes to compute the volume of a random polytope using `scipy.spacial`. 
This is the package `PreimageApproxForNNs` uses in the part that can be used to validate the unsound verification results.

To get an idea how this component scales with the input size, run:
```bash
n=2 ./time_polytope_volume.sh
n=3 ./time_polytope_volume.sh
n=4 ./time_polytope_volume.sh
n=5 ./time_polytope_volume.sh
n=6 ./time_polytope_volume.sh
n=7 ./time_polytope_volume.sh
n=8 ./time_polytope_volume.sh
n=9 ./time_polytope_volume.sh
```
Keep in mind that these runtimes are the cost of computing the volume of only *one* polytope.
