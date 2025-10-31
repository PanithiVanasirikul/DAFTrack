# DAFTrack: Distortion-Aware Multi-Object Tracking via Virtual Plane Projection in Overhead Fisheye Cameras

BMVC, 2025

## Requirements

- `Python 3.8` or `Conda`

## Setup

1. Create a virtual environment using:

    ```bash
    python -m venv env
    ```

    or using Conda:

    ```bash
    conda create -n daftrack python=3.8
    ```

2. Activate the virtual environment:
    - Venv:
        - On Windows:

        ```bash
        .\env\Scripts\activate
        ```

        - On Linux/Mac:

        ```bash
        source env/bin/activate
        ```
    
    - Conda:

        ```bash
        conda activate daftrack
        ```


3. Install the required packages using:

    ```bash
    pip install -r requirements.txt
    ```


4. Download [WEPDTOF](https://vip.bu.edu/projects/vsns/cossy/datasets/wepdtof/) and [CEPDOF](https://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/)
    
    After finishing downloading and extracting, place the files in the `datasets` folder and unzip them. The directory structure should look like this:

    ```
    .
    ├── datasets
    │   ├── WEPDTOF
    │   ├── CEPDOF
    ```

    And remove all `__MACOSX` folders in the `datasets` folder and `.DS_Store` in every folders in `dataset` folder.

    Or you can use the following command to preprocess the datasets:

    - Linux/Mac:
        ```bash
        sh cleanup_datasets.sh
        ```
    - Windows:
        ```bat
        cleanup_datasets.bat
        ```


5. Run benchmarking scripts:
    - For WEPDTOF dataset:
        ```bash
        python all_demo.py --dataset WEPDTOF
        ```
    
    - For CEPDOF dataset with ground truth camera height:
        ```
        python all_demo.py --dataset CEPDOF --camera_height 1.7
        ```

    - For CEPDOF dataset without ground truth camera height:
        ```
        python all_demo.py --dataset CEPDOF
        ```