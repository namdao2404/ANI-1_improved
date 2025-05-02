# ANI-1_improved

---

Machine learning for interatomic potentials (MLIPs) can leverage quantum mechanical data generated
by ab initio quantum chemistry calculations to generate state-of-the art force field predictions of
atomic and material systems, allowing unprecedented efficiency and accuracy while avoiding the
computational burden suffered by more traditional methods like density functional theory (DFT),
coupled-cluster theory (CC). T o achieve the translational, permutational and rotational invariance
required to accurately train such MLIPs, one often used method comes from Behler and Parinello
(2007), who proposed a neural network for the learning and representation of high dimensional
potential energy surfaces (PES) via the introduction of a symmetry functions (SFs) that enforce the
desired equivariances. These SFs are then embedded within local atomic environments within a
message passing neural network (MPNN) or graph neural network (GNN) for training.

Since symmetry function-based atomic representations have low transferability outside of bulk systems
or water cases, Smith et. al (2017) have suggested that SFs are inadequately featurized to account for
specific local environment features or atomic number differentiation, and proposed ANAKIN
(Accurate NeurAl networK engINe for Molecular Energies) or ANI for short, a new construction of a
MLIP that utilizes single-atom atomic environment vectors (AEV s) to achieve the transferability that
the SFs of Behler and Parinello (2007) lack. After training on a GDB dataset, the ANI model
successfully achieves DFT-level chemical accuracy on systems much larger than the ones from the
training data, demonstrating good generalization.

In this work, I show how a more sophisticated machine learning architecture of the ANI-1 model and
trainer method could improve the performance of the ANI-1 model even further. I also compare the
performance of my new model compared to an improved ANI-1 model, ANI-1x.


---
References
1. J. Behler and M. Parrinello , Phys. Rev. Lett., 2007, 98, 146401
2. J. S. Smith, O.Isayev, and A. E. Roitberg, Chem. Sci., 2017,8, 3192-3203
3. J. S. Smith, B. Nebgen, N. Lubbers, O. Isayev, and A. E. Roitberg. J. Chem. Phys., 2018, 148,
241733
4. X. Gao, F. Ramezanghorbani, O. Isayev, J. S. Smith, and A. E. Roitberg. J. Chem. Information
and Modeling, 2020 60 (7), 3408-3415


