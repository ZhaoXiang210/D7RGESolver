# D7RGESolver

D7RGESolver is a Python-based tool for solving the full Renormalization Group Equations (RGE) of dimension-5 and dimension-7 SMEFT interactions automatically. 
We also provide usage examples for solving RGEs using D7RGESolver and calculating neutrinoless double beta decay (0vbb) through external code interfacing.

# Contact

**Contact:**  
- **Name:** Xiang Zhao  
- **Email:** zhaox88@mail2.sysu.edu.cn 

# Features

- Solve the RGEs of full dimension-5 and dimension-7 SMEFT operators.
- Solve the RGEs at both up- and down-quark flavor basis.
- Generate the constraints on the single Wilson coefficient at any scale above electroweak scale from 0vbb automatically.
- Can be used to study the lepton-number violation and baryon-number-violation process.

# Citation

If you use `D7RGESolver` in your work, please cite the following:

> "RGE solver for the complete dim-7 SMEFT interactions and its application to 0νββ decay"
>
>  Yi Liao, Xiao-Dong Ma, Hao-Lin Wang and Xiang Zhao
>
>  [arXiv:1111.1111 [hep-ph]](https://arxiv.org/abs/1111.1111)


# Related work


- The SMEFT RGEs are based on [arXiv:1901.10302](https://arxiv.org/abs/1901.10302) and [arXiv:2310.11055](https://arxiv.org/abs/2310.11055).

For the external tool `nudobe` (developed by others) used for neutrinoless double-beta decay calculations, please cite: [arXiv:2304.05415 [hep-ph]](https://arxiv.org/abs/2304.05415)


# Dependencies
Required Python packages:
- numpy
- scipy 
- pandas
- matplotlib
-...

The project can interface with [nudobe] for calculations related to neutrinoless double-beta decay observables. Note that nudobe is a separate tool developed by others.

# License

D7RGESolver is licensed under the Creative Commons NonCommercial-ShareAlike 4.0 International License (cc-by-nc-sa-4.0).
