# Logic-LfD

## This is a python implementation of our paper "Logic-LfD: Logic Learning from Demonstrations for Multi-step Manipulation Tasks in Dynamic Environments", published in IEEE RA-L 2024 
[[pdf]](https://arxiv.org/pdf/2404.16138v2)

[[webpage]](https://sites.google.com/view/logic-lfd)

## Installation
1. Clone the repo along with the submodules. It may take a while.
	```
	git clone git@github.com:ollieyzhang/Logic-LfD.git --recursive
	```
2. Install dependencies:
	```
	conda env create -f environment.yml
	sudo activate logic_lfd
	```
3. Build FastDownward, used by PDDLStream planner
	```
	## sudo apt install cmake g++ git make python3
	cd pddlstream; ./downward/build.py
	```

## Scripts
1. Logic-LfD for block stacking
	```
	python -m experiments/logic_lfd_block_stacking_refine_init_generalize.py
	```
2. Reactive TAMP for block stacking
	```
	python -m experiments/reactive_tamp_block_stacking_refine.py
	```

## Notes
This repo aims to illustrate how LogicLfD accelerates TAMP solvers with one single demonstration. We simplified the planning domain defintion and streams to help the readers to understand our method. For full TAMP planning, one should modify the PDDL files correspondingly for their specific problems.
## Citation
If this project helps your work, please consider citing our paper with
```
@article{zhang2024logic,
  title={Logic Learning from Demonstrations for Multi-step Manipulation Tasks in Dynamic Environments},
  author={Zhang, Yan and Xue, Teng and Razmjoo, Amirreza and Calinon, Sylvain},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  volume={9},
  number={8},
  pages={7214-7221},
  doi={10.1109/LRA.2024.3418276}
}
```

## Reference
This package is developed based on the shared GitHub package [kitchen-worlds](https://github.com/Learning-and-Intelligent-Systems/kitchen-worlds.git) as well as its dependencies.