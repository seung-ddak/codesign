def cond_masks(n, mode, M):
    """mode: 0 joint (s0), 1 dynamics (E·s0·A), 2 given_E (E·s0), 3 free (없음, s0 도 생성). known 이면 1."""
    kx = torch.zeros(n, args.horizon, X_DIM, device=dev)
    kx[:, 0, :S_DIM] = (mode != 3).float()[:, None]
    kx[:, :, S_DIM:] += (mode == 1).float()[:, None, None]
    ke = (mode >= 1).float()[:, None].expand(n, E_DIM)
    return (kx * M[..., None], ke)