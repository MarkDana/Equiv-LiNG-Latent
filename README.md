# Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models: Characterization and Learning

[Paper](https://openreview.net/forum?id=b8TlYh6PN6) by [Haoyue Dai](https://hyda.cc), [Immanuel Albrecht](https://de.linkedin.com/in/alb-i), [Peter Spirtes](https://www.cmu.edu/dietrich/philosophy/people/faculty/spirtes.html), [Kun Zhang](https://www.andrew.cmu.edu/user/kunz1/index.html). Appears at ICLR 2026 (oral).

*When are two latent-variable models with arbitrary structure and feedback cycles indistinguishable from data? How can the equivalence class be traversed, represented, and recovered? We address these questions in the linear non-Gaussian setting. This is, to our knowledge, the first structural-assumption-free method for latent-variable causal discovery.*

This repository contains two modules that address **characterization** and **learning**, respectively:




## 1. Characterization module

Given any latent-variable graph as input, this module can compute **the CPDAG-like representation** of the input graph's equivalence class, and can **traverse all member graphs** within that equivalence class. This is the code used in our online demo at https://equiv.cc.

To run the demo locally, run the following commands:

```bash
cd equivalence_class_searcher

# download prebuilt static files, mainly graph plots (56.9M)
curl -L https://raw.githubusercontent.com/MarkDana/Equiv-LiNG-Latent-static-storage/main/ling-iclr26-website-static.tar.gz | tar -xz

# start the Flask app
python app.py
```

Once the server starts, open your browser and navigate to:

```text
http://127.0.0.1:5000
```

You should then see the interactive demo interface running locally.

**Note:** For website response speed, we impose limits on the maximum class size and the maximum traversal time. While the returned CPDAG-like representation is always correct, the traversal of the equivalence class may be incomplete due to these early stopping criteria; users are explicitly notified when this occurs.

To obtain a complete traversal of the equivalence class on your local machine, you can disable these limits in `app.py`:

```python
# Line 107 in app.py
resdict = ling_equiv_checker.traverse_dist_equiv_class_from_an_irreducible_graph_config(
    ...,
    max_class_size_for_column_augmentation=None,     # <- set to None
    max_time_seconds_for_column_augmentation=None,   # <- set to None
    # set all other stopping-related parameters to None as well
)
```

This allows the traversal to run until completion, regardless of the size of the equivalence class or the time used. Be aware that for large equivalence classes, this process can consume a long time and a large memory (See Table 3 in Appendix D.1 in the paper).






## 2. Learning module

This module implements **glvLiNG** (**g**eneral **l**atent-**v**ariable **Li**near **N**on-**G**aussian causal discovery), a structural-assumption-free algorithm for learning causal structure with latent variables.

Given **observational data** generated from an arbitrary, unknown latent-variable graph under the LiNG model, glvLiNG recovers **one representative graph** from the equivalence class of the true underlying graph. Specifically, the recovered graph is a maximal member of the equivalence class, sharing the same set of edges as the corresponding CPDAG-like representation.

If you want to further know the edge types in this returned representation, or to traverse the entire equivalence class starting from it, please refer to the characterization module described above.

Our code takes as input an **over-complete ICA (OICA)** estimated rectangular mixing matrix, which can be obtained by applying any suitable OICA method to the data. In our experiments, we use the MATLAB implementation of the OICA approach proposed by Podosinnikova et al. (2019), available at https://github.com/gilgarmish/oica.

Once you have the OICA estimated mixing matrix, you can run glvLiNG as follows:

```bash
cd glvLiNG_algorithm
python main.py  # A synthetic example is provided; replace it with your own input as needed.
```

**Note:** The glvLiNG algorithm is primarily intended as a **proof of concept**, showing that a structural-assumption-free method is possible in principle. However, due to the known computational inefficiency of OICA, this implementation is far away from an off-the-shelf solution for practical use. The hyperparameters may also need careful tuning; please refer to the code and comments in `main.py`. Overall, the main contribution of this paper is the **Characterization module** above.






## Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{
  dai2026distributional,
  title={Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models: Characterization and Learning},
  author={Haoyue Dai and Immanuel Albrecht and Peter Spirtes and Kun Zhang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=b8TlYh6PN6}
}
```
