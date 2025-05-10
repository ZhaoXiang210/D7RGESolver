# D7RGESolver

D7RGESolver is a Python-based tool for solving the full Renormalization Group Equations (RGE) of dimension-5 and dimension-7 SMEFT interactions automatically. 
We provide examples for solving RGEs using D7RGESolver and calculating neutrinoless double beta decay (0vbb) through interfacing with external code.

# Contact

**Contact:**  
- **Name:** Xiang Zhao  
- **Email:** zhaox88@mail2.sysu.edu.cn 

# Features

- Solve the RGEs of full dimension-5 and dimension-7 SMEFT operators.
- Solve the RGEs at both up- and down-quark flavor bases.
- Generate the constraints on the single Wilson coefficient at any scale above electroweak scale from 0vbb automatically.
- Study other lepton-number-violating and baryon-number-violating processes with RGE effects.

# Citation

If you use `D7RGESolver` in your work, please cite the following:

> "RGE solver for the complete dim-7 SMEFT interactions and its application to 0νββ decay"
>
>  Yi Liao, Xiao-Dong Ma, Hao-Lin Wang and Xiang Zhao
>
>  [arXiv:1111.1111 [hep-ph]](https://arxiv.org/abs/1111.1111)


# Related work

- The dimension-5 and dimension-7 SMEFT operator bases and RGEs are based on [10.1103/PhysRevLett.43.1566], [hep-ph/0108005], [arXiv:1410.4193],[arXiv:1607.07309], [arXiv:1901.10302], [arXiv:2306.03008], and [arXiv:2310.11055].
                                                                             
                                                                            
# Dependencies
Required Python packages:
- numpy
- scipy 
- pandas
- matplotlib

The project can interface with [nudobe] for calculations related to neutrinoless double-beta decay observables. Note that nudobe is a separate tool developed by others.

# License

D7RGESolver is licensed under the Creative Commons NonCommercial-ShareAlike 4.0 International License (cc-by-nc-sa-4.0).
