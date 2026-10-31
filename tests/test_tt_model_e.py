import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parents[1] / "code" / "tt_dynamics"))
from rollout_data import E_DIM, X_DIM
from tt_model import EGuidedNet, Net


def args(n=2, H=4):
    return (torch.randn(n, H, X_DIM), torch.randn(n, E_DIM), torch.zeros(n, H, X_DIM), torch.ones(n, E_DIM),
            torch.zeros(n), torch.zeros(n, H, dtype=torch.bool))


def test_film_starts_as_plain_model_and_learns_from_e():
    torch.manual_seed(0)
    plain, film = Net(4, 16, 2).eval(), Net(4, 16, 2, film=True).eval()
    film.load_state_dict({**film.state_dict(), **plain.state_dict()})
    a = args()
    assert torch.allclose(plain(*a)[0], film(*a)[0], atol=1e-6)
    film.film[0][-1].bias.data.fill_(.1)
    assert not torch.allclose(plain(*a)[0], film(*a)[0], atol=1e-3)


def test_e_guidance_extrapolates_known_rows_only():
    net = Net(4, 16, 2).eval()
    x, e, kx, ke, k, pad = args()
    ke[1] = 0
    base = net(x, e, kx, ke, k, pad)[0]
    hidden = net(x, torch.zeros_like(e), kx, torch.zeros_like(ke), k, pad)[0]
    out = EGuidedNet(net, 3.)(x, e, kx, ke, k, pad)[0]
    assert torch.allclose(out[0], hidden[0] + 3 * (base[0] - hidden[0]), atol=1e-5)
    assert torch.allclose(out[1], base[1])
    assert EGuidedNet(net, 1.)(x, e, kx, ke, k, pad)[0].equal(base)
