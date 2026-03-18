import torch
from collections import defaultdict

def _take_first_tensor(x):
    # Pull a representative tensor out of nested outputs
    if torch.is_tensor(x): return x
    if isinstance(x, (list, tuple)):
        for t in x:
            y = _take_first_tensor(t)
            if y is not None: return y
    if isinstance(x, dict):
        for t in x.values():
            y = _take_first_tensor(t)
            if y is not None: return y
    return None

class GradMonitor:
    """
    Attach to a module; records per-module activation and dLoss/dOutput stats.
    Call .summary() after backward to print a compact report.
    """
    def __init__(self, modules_to_watch=None, print_empty=False):
        self.handles = []
        self.data = defaultdict(dict)
        self.modules_to_watch = set(modules_to_watch) if modules_to_watch else None
        self.print_empty = print_empty

    def _should_watch(self, name):
        if self.modules_to_watch is None: return True
        return any(name == k or name.endswith(f".{k}") for k in self.modules_to_watch)

    def install(self, root_module: torch.nn.Module):
        for name, m in root_module.named_modules():
            if not self._should_watch(name):
                continue

            # Forward: log activations + retain grad on outputs
            def fwd_hook(mod, inputs, output, *, _name=name):
                t = _take_first_tensor(output)
                if t is None: 
                    self.data[_name]["act"] = None
                    return
                with torch.no_grad():
                    self.data[_name]["shape"] = tuple(t.shape)
                    self.data[_name]["act"] = (
                        t.min().item(), t.mean().item(), t.max().item()
                    )
                # Make sure we can see dLoss/dOutput for this module
                if t.requires_grad:
                    t.retain_grad()
                    def _bwd_hook(grad, *, _n=_name):
                        with torch.no_grad():
                            self.data[_n]["grad"] = (
                                grad.min().item(),
                                grad.abs().mean().item(),
                                grad.max().item()
                            )
                        return grad
                    t.register_hook(_bwd_hook)
                else:
                    self.data[_name]["grad"] = None

            h1 = m.register_forward_hook(fwd_hook)
            self.handles.append(h1)
        return self

    def remove(self):
        for h in self.handles:
            try: 
                h.remove()
            except: 
                pass
        self.handles.clear()

    def summary(self, sort_by_name=True):
        lines = []
        keys = list(self.data.keys())
        if sort_by_name: keys.sort()
        for k in keys:
            ent = self.data[k]
            shape = ent.get("shape")
            act = ent.get("act")
            grd = ent.get("grad")
            if not self.print_empty and (act is None and grd is None):
                continue
            s_shape = f"{shape}" if shape else "–"
            s_act = "–" if act is None else f"act[min/mean/max]=({act[0]:.3e},{act[1]:.3e},{act[2]:.3e})"
            s_grd = "–" if grd is None else f"grad[min/|mean|/max]=({grd[0]:.3e},{grd[1]:.3e},{grd[2]:.3e})"
            lines.append(f"{k:40s}  {s_shape:>20s}  {s_act}  {s_grd}")
        report = "\n".join(lines) if lines else "(no data)"
        return report
