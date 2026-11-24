"""One-shot classifiers on top of frozen, L2-normalised embeddings.

Every method has two parts:
  prepare(ctx, cfg) -> state   builds / trains whatever is learned from the 16 shots and the prompts
  predict(state, hp, feats)    class scores for a batch of query embeddings
`hp` holds the few inference-time hyperparameters chosen on the validation split.
The only labelled images a method ever sees in `prepare` are the 16 shots (one per class).
"""
import torch
import torch.nn.functional as F


def text_proto(ctx):
    """Prompt ensemble: per prompt set, the renormalised mean embedding of each class; the sets
    (class-name templates, short descriptions) are then averaged with equal weight."""
    protos = [F.normalize(t.mean(1), dim=-1) for t in ctx["text"]]
    return F.normalize(torch.stack(protos).mean(0), dim=-1)


def aug_proto(ctx):
    """Mean embedding of the shot and its augmented views, renormalised."""
    views = torch.cat([ctx["shot"][:, None], ctx["shot_aug"]], dim=1)
    return F.normalize(views.mean(1), dim=-1)


# --- 1. zero-shot, 2. one-shot image prototype, 5a. augmented prototype -------------------------

def prep_zeroshot(ctx, cfg):
    return {"w": text_proto(ctx)}


def prep_proto(ctx, cfg):
    return {"w": aug_proto(ctx) if cfg.get("augmented") else ctx["shot"].clone()}


def predict_linear(state, hp, feats):
    return feats @ state["w"].T


# --- 3. text + image prototype fusion (5b with the augmented prototype) -------------------------

def prep_fusion(ctx, cfg):
    return {"img": aug_proto(ctx) if cfg.get("augmented") else ctx["shot"].clone(), "txt": text_proto(ctx)}


def predict_fusion(state, hp, feats):
    a = hp["alpha"]
    return a * (feats @ state["img"].T) + (1 - a) * (feats @ state["txt"].T)


# --- 4. Tip-Adapter and Tip-Adapter-F (Zhang et al., ECCV 2022) ---------------------------------

def prep_tip(ctx, cfg):
    return {"keys": ctx["shot"].clone(), "txt": text_proto(ctx), "scale": ctx["scale"]}


def predict_tip(state, hp, feats):
    zero_shot = state["scale"] * (feats @ state["txt"].T)
    affinity = feats @ state["keys"].T  # one key per class, so the cache values are the identity
    return zero_shot + hp["alpha"] * torch.exp(-hp["beta"] * (1 - affinity))


def prep_tip_f(ctx, cfg):
    """Fine-tune the cache keys on augmented views of the 16 shots; everything else stays frozen."""
    state = prep_tip(ctx, cfg)
    hp = cfg["default"]
    x = ctx["shot_aug"].flatten(0, 1)
    y = torch.arange(ctx["shot_aug"].shape[0]).repeat_interleave(ctx["shot_aug"].shape[1])
    g = torch.Generator().manual_seed(ctx["seed"])
    keys = torch.nn.Parameter(state["keys"].clone())
    opt = torch.optim.AdamW([keys], lr=cfg["lr"], eps=1e-4)
    steps = cfg["epochs"] * ((len(x) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            logits = predict_tip({**state, "keys": keys}, hp, x[b])
            loss = F.cross_entropy(logits, y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    state["keys"] = keys.detach()
    return state


# --- 5c. linear head initialised from the text prototypes ---------------------------------------

def prep_linear_head(ctx, cfg):
    """Linear classifier that starts at the prompt weights and is trained on the 16 shots and their
    augmented views, with an L2 pull back toward the prompt weights (strength `lam`).
    One head is trained per candidate `lam`; the choice between them is an inference hyperparameter.
    """
    w0 = text_proto(ctx)
    x = torch.cat([ctx["shot"], ctx["shot_aug"].flatten(0, 1)])
    n_cls, n_aug = ctx["shot_aug"].shape[:2]
    y = torch.cat([torch.arange(n_cls), torch.arange(n_cls).repeat_interleave(n_aug)])
    heads = {}
    for lam in cfg["lams"]:
        w = torch.nn.Parameter(w0.clone())
        opt = torch.optim.Adam([w], lr=cfg["lr"])
        for _ in range(cfg["steps"]):
            loss = F.cross_entropy(ctx["scale"] * (x @ w.T), y) + lam * (w - w0).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        heads[float(lam)] = w.detach()
    return {"heads": heads}


def predict_linear_head(state, hp, feats):
    return feats @ state["heads"][float(hp["lam"])].T


# --- 6. transductive prototype refinement (test-time adaptation on unlabelled queries) ----------

def sinkhorn(logits, iters=30):
    """Soft assignment with equal class marginals (each class gets N/C of the mass)."""
    logp = logits.log_softmax(1)
    n, c = logp.shape
    target = torch.log(torch.tensor(n / c))
    for _ in range(iters):
        logp = logp - logp.logsumexp(0, keepdim=True) + target
        logp = logp - logp.logsumexp(1, keepdim=True)
    return logp.exp()


def prep_transductive(ctx, cfg):
    return {"img": aug_proto(ctx), "txt": text_proto(ctx)}


def predict_transductive(state, hp, feats):
    """Soft k-means in image space, seeded by the shot prototype and guided by the text scores.

    The whole query batch is used, without labels: prototypes are re-estimated from the queries'
    own soft assignments, with the shot acting as `kappa` pseudo-observations of its class.
    With no shot (state["img"] is None) the first assignment comes from the text scores alone.
    """
    text_sim = feats @ state["txt"].T
    text_scores = hp["gamma"] * text_sim
    shot = state["img"]
    proto = shot
    for _ in range(hp.get("iters", 10)):
        scores = text_sim if proto is None else feats @ proto.T + text_scores
        p = sinkhorn(hp["tau"] * scores) if hp["balanced"] else (hp["tau"] * scores).softmax(1)
        proto = p.T @ feats
        if shot is not None:
            proto = proto + hp["kappa"] * shot
        proto = F.normalize(proto, dim=-1)
    return feats @ proto.T + text_scores


METHODS = {
    "zeroshot": (prep_zeroshot, predict_linear),
    "proto": (prep_proto, predict_linear),
    "fusion": (prep_fusion, predict_fusion),
    "tip_adapter": (prep_tip, predict_tip),
    "tip_adapter_f": (prep_tip_f, predict_tip),
    "linear_head": (prep_linear_head, predict_linear_head),
    "transductive": (prep_transductive, predict_transductive),
}


def accuracy(scores, labels):
    return (scores.argmax(1) == labels).float().mean().item()
