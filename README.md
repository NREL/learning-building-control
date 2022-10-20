#  From Model-Based to Model-Free: Learning Building Control for Demand Response

### Description

This repository encompasses the source code and data to reproduce and extend results for a manuscript that compares demand responsive control schemes for multi-zone buildings. It includes examples of model predictive control (MPC and MPC-C), value function approximation via CVXPYLAYERS (MPC-CL), differentiable predictive control (DPC) and reinforcement learning control (RLC). The goal of the study is to evaluate state-of-the-art controllers and establish the efficacy (if any) of learning-based approaches that leverage deep neural networks in one way or another.

Please refer to our [preprint on arXiv](https://arxiv.org/abs/2210.10203) for 
more details.

### Basic installation instructions

Env setup using platform-dependent yaml file:

```
conda env create -n <env-name> -f env-xxxx.yaml
pip install -e .
```

### Citation

If citing this work, please use the following:

```bibtex
@article{biagioni2022lbc,
  title={From Model-Based to Model-Free: Learning Building Control for Demand Response},
  author={Biagioni, David and Zhang, Xiangyu and Adcock, Christiane and Sinner, Michael and Graf, Peter and King, Jennifer},
  journal={arXiv preprint arXiv:2210.10203},
  year={2022}
}
```


