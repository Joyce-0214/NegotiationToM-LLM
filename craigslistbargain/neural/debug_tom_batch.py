import torch


def _fmt_tensor(x, max_rows=2):
    if x is None:
        return "None"
    if isinstance(x, list):
        return f"list(len={len(x)})"
    if not isinstance(x, torch.Tensor):
        return f"{type(x)}"

    rows = min(max_rows, x.shape[0]) if x.dim() > 0 else 1
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "sample": x[:rows].detach().cpu()
    }


def inspect_tom_batch(tom_batch, name="tom_batch", max_rows=2):
    print(f"\n===== {name} =====")

    print("identity_state:", _fmt_tensor(tom_batch.identity_state, max_rows))
    print("extra         :", _fmt_tensor(tom_batch.extra, max_rows))
    print("state         :", _fmt_tensor(tom_batch.state, max_rows))
    print("last_price    :", _fmt_tensor(getattr(tom_batch, "last_price", None), max_rows))
    print("act_intent    :", _fmt_tensor(tom_batch.act_intent, max_rows))
    print("act_price     :", _fmt_tensor(tom_batch.act_price, max_rows))
    print("act_price_mask:", _fmt_tensor(tom_batch.act_price_mask, max_rows))

    uttr = tom_batch.uttr
    if isinstance(uttr, list):
        print(f"uttr: list(len={len(uttr)})")
        if len(uttr) > 0 and isinstance(uttr[0], torch.Tensor):
            print("uttr[0] shape:", tuple(uttr[0].shape), "dtype:", uttr[0].dtype)
            print("uttr[0] sample:", uttr[0][:min(10, uttr[0].shape[0])].detach().cpu())
    else:
        print("uttr:", _fmt_tensor(uttr, max_rows))