"""Export helper -- copied verbatim from README-training-parameters.md
Appendix A. Not part of the assignment; imported by train_wakeword.py."""
import os

import torch


def export_to_executorch(model: torch.nn.Module, example_input: tuple, pte_path: str, print_ops: bool = True):
    """model.eval() -> torch.export.export -> exir.to_edge -> to_executorch,
    writes the .pte to pte_path. Returns (exported_program, size_bytes).

    Prints the full operator list used by the exported graph -- this is
    informational only (for setting EXECUTORCH_SELECT_OPS_LIST when
    rebuilding ExecuTorch itself)."""
    import executorch.exir as exir

    model.eval()  # mandatory -- folds BatchNorm into conv weights
    ep = torch.export.export(model, example_input)
    edge = exir.to_edge(ep)

    if print_ops:
        # Op names must be read from the EDGE-lowered graph (after
        # exir.to_edge), not the raw torch.export graph -- the raw graph's
        # ATen "default"-variant names don't correspond 1:1 to what's
        # registered in the portable kernel library. node.target.name()
        # gives the clean base name (e.g. "aten::convolution"); the kernel
        # registry entry EXECUTORCH_SELECT_OPS_LIST needs is normally that
        # name + ".out" (e.g. "aten::convolution.out") -- but this is a
        # best-effort suggestion, not guaranteed: verify by attempting the
        # stm32 build and watching for OperatorMissing (error code 0x14 at
        # load_method), adjusting the list if any op doesn't match.
        ops = sorted({
            node.target.name() for node in edge.exported_program().graph.nodes
            if node.op == "call_function" and hasattr(node.target, "name")
        })
        suggested = [op + ".out" for op in ops]
        print("\n=== Operators used by this model ===")
        for op in ops:
            print(f"  {op}")
        print("=====================================")
        print("Best-effort EXECUTORCH_SELECT_OPS_LIST suggestion (verify against")
        print("OperatorMissing errors when building/running -- see README-testing.md):")
        print("  -DEXECUTORCH_SELECT_OPS_LIST=\"" + ",".join(suggested) + "\"\n")

    et_program = edge.to_executorch()

    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)

    size = os.path.getsize(pte_path)
    print(f"Written: {pte_path} ({size} bytes, {size / 1024:.1f} KB)")
    return ep, size
