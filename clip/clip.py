import torch

def cow_clip(w, g, ratio=1, ids=None, cnts=None, min_w=0.03, const=False):
    if g.is_sparse:
        # FIXME: This part is not tested
        values = g.to_dense()
        clipnorm = torch.norm(w.index_select(0, g._indices().squeeze()), dim=-1)
    else:
        values = g
        if const:
            clipnorm = torch.full_like(g, min_w)
        else:
            clipnorm = torch.norm(w, dim=-1)
            # bound weight norm by min_w
            clipnorm = torch.max(clipnorm, torch.tensor(min_w).to(clipnorm.device))
        # scale by cnting
        if ids is not None and cnts is not None:
            cnts = torch.ones_like(clipnorm, dtype=torch.int32).scatter_(0, ids, cnts)
            clipnorm = clipnorm * cnts.float()

    clip_t = ratio * clipnorm
    l2sum_row = torch.sum(values * values, dim=-1)
    pred = l2sum_row > 0
    l2sum_row_safe = torch.where(pred, l2sum_row, torch.ones_like(l2sum_row))
    l2norm_row = torch.sqrt(l2sum_row_safe)

    intermediate = values * clip_t.unsqueeze(-1)
    g_clip = intermediate / torch.max(l2norm_row, clip_t).unsqueeze(-1)

    if g.is_sparse:
        return torch.sparse.FloatTensor(g._indices(), g_clip, g.shape)
    else:
        return g_clip