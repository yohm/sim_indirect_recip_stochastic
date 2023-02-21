# Indirect Reciprocity with Stochastic Reputation

The source code for the manuscript "Indirect Reciprocity with Stochastic Reputation" by Yohsuke Murase and Christian Hilbe.

## Build

Use `cmake` to build the project.
Clone the repository with submodules.

```bash
git clone --recursive git@github.com:yohm/sim_indirect_recip_stochastic.git
cd sim_indirect_recip_stochastic
mkdir build
cd build
cmake ..
cmake --build .
```

## Executables

- `main_public_rep`: main executable for the public reputation model. Conduct all the calculations for the paper.
- `test_PublicRepGame`: unit tests for PublicRepGame class
- `test_Norm`: unit test for Norm class
