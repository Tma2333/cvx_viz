# cvx_viz 

Visualize convex optimization with ManimCE. A EE364B Project. 

**Version** 0.1.0

![Alt Text](https://raw.githubusercontent.com/Tma2333/cvx_viz/main/docs/QuadraticPGD_ManimCE_v0.15.2.gif)

## Quickstart

For more detail please check out [documentation](https://github.com/Tma2333/cvx_viz/wiki)

### Windows

1. Download and install [MiKTeX](https://miktex.org/download) or other LaTeX distribution
2. Create Manim conda enviroment
```
conda env create -f environment.yml
conda activate cvx_viz
```

**Render Visualization**

You can either run example cell in `demo.ipynb` using `cvx_viz` kernel. 

or 

Run visualization script in `./visualization`

```
cd visualization
manim -pql quadratic_pgd.py QuadraticPGD
```

### Colab

No installation needed. Follow instruction in [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tma2333/cvx_viz/blob/main/demo.ipynb)

### Ohter OS
Have not tested with other OS, you can try to follow official [ManimCE](https://docs.manim.community/en/stable/installation.html) installation guide. Or use Colab. 