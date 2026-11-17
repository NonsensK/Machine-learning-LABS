# Problem definition

Implement and validate scalar reverse-mode automatic differentiation, then classify two synthetic interleaving moons with a 2 → 8 → 1 network. Success requires maximum gradient error below 1e-6 against both PyTorch autograd and central finite differences. Compare the median training duration over five identical runs, report held-out accuracy, and measure eager versus traced inference and contiguous versus strided tensor operation costs on CPU.
