import functools

import torch
from lloca.utils.polar_decomposition import restframe_boost
from lloca.utils.utils import get_batch_from_ptr

from experiments.hep import (
    EPPP_to_PtPhiEtaM2,
    PtPhiEtaM2_to_EPPP,
    get_eta,
    get_phi,
    get_pt,
    get_rapidity,
)

EPS = 1e-5

# weaver defaults for tagging features standardization (mean, std)
AUXILIARY_SCALARS_PREPROCESSING = [
    [1.7, 0.7],  # log_pt
    [2.0, 0.7],  # log_energy
    [-4.7, 0.7],  # log_pt_rel
    [-4.7, 0.7],  # log_energy_rel
    [0, 1],  # dphi
    [0, 1],  # deta
    [0.2, 4],  # dr
]


def embed_tagging_data(fourmomenta, scalars, cfg_data):
    """
    Embed tagging data
    We use torch_geometric sparse representations to be more memory efficient
    Note that we do not embed the label, because it is handled elsewhere

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (batchsize, n_particles, 4)
        Four-momenta in the format (E, px, py, pz)
    scalars: torch.tensor of shape (batchsize, n_particles, n_features)
        Optional scalar features, n_features=0 is possible
    cfg_data: settings for embedding

    Returns
    -------
    vectors: torch.Tensor
        Lorentz vectors with spurions included, shape (batchsize, n_particles + n_spurions, C, 4) with
        channel 0 the four-momentum. C=1 normally.
    scalars: torch.Tensor
        Scalar features with spurions included, shape (batchsize, n_particles + n_spurions, n_features)
    auxiliary_scalars: torch.Tensor
        Precomputed tagging features, shape (batchsize, n_particles + n_spurions, n_auxiliary_scalars)
    is_spurion: torch.BoolTensor
        Boolean mask with 'True' in spurion positions, shape (batchsize, n_particles + n_spurions)
    mask: torch.BoolTensor
        Boolean mask with 'True' in valid particle positions, shape (batchsize, n_particles + n_spurions)
    """
    # crop jets to max_particles
    if cfg_data.max_particles is not None:
        fourmomenta = fourmomenta[:, : cfg_data.max_particles]
        scalars = scalars[:, : cfg_data.max_particles]

    # include spurions if specified
    spurions = get_spurion(
        cfg_data.beam_reference,
        cfg_data.add_time_reference,
        cfg_data.two_beams,
        fourmomenta.device,
        fourmomenta.dtype,
    )
    spurions = spurions * cfg_data.spurion_scale
    n_spurions = spurions.shape[0]

    spurions = spurions.unsqueeze(0).repeat(fourmomenta.shape[0], 1, 1)
    fourmomenta = torch.cat([spurions, fourmomenta], dim=1)
    spurion_scalars = torch.zeros(
        *spurions.shape[:-1], scalars.shape[-1], device=fourmomenta.device, dtype=scalars.dtype
    )
    scalars = torch.cat([spurion_scalars, scalars], dim=1)
    is_spurion = torch.zeros(
        fourmomenta.shape[0],
        fourmomenta.shape[1],
        dtype=torch.bool,
        device=fourmomenta.device,
    )
    is_spurion[:, :n_spurions] = True

    mask = (fourmomenta.abs() > EPS).any(dim=-1)
    max_size = int(mask.sum(dim=-1).max())
    fourmomenta = fourmomenta[:, :max_size]
    scalars = scalars[:, :max_size]
    is_spurion = is_spurion[:, :max_size]
    mask = mask[:, :max_size]

    vectors = fourmomenta.unsqueeze(2)

    # construct canonicalization based on momenta, then apply to all Lorentz vectors
    if cfg_data.canonicalize in ["beam_eta", "beam_y"]:
        # apply boost in z direction and rotation around z direction to set eta_jet=phi_jet=0
        # can use either rapidity ('y') or pseudo rapidity ('eta')
        # transformation also applied to spurions, therefore it does not violate Lorentz equivariance
        jet = vectors[:, n_spurions:, 0].sum(dim=1, keepdim=True)
        phi_jet = get_phi(jet).unsqueeze(-1)
        eta_jet = (
            get_eta(jet) if cfg_data.canonicalize == "beam_eta" else get_rapidity(jet)
        ).unsqueeze(-1)
        ptphietam2 = EPPP_to_PtPhiEtaM2(vectors)
        if cfg_data.canonicalize_spurions:
            ptphietam2[..., 1] -= phi_jet
            ptphietam2[..., 2] -= eta_jet
            vectors = PtPhiEtaM2_to_EPPP(ptphietam2)
        else:
            # canonicalize only the constituents: the EPPP<->PtPhiEtaM2 round-trip is lossy for
            # light-like beam spurions (pt=0 -> eta clamps to CUTOFF)
            ptphietam2[:, n_spurions:, :, 1] -= phi_jet
            ptphietam2[:, n_spurions:, :, 2] -= eta_jet
            vectors[:, n_spurions:] = PtPhiEtaM2_to_EPPP(ptphietam2[:, n_spurions:])
    elif cfg_data.canonicalize == "rest":
        # boost to the jet rest frame to avoid large boosts
        jet = vectors[:, n_spurions:, 0].sum(dim=1, keepdim=True)
        jet_boost = restframe_boost(jet).unsqueeze(2)
        if cfg_data.canonicalize_spurions:
            vectors = torch.einsum("...jk,...k->...j", jet_boost, vectors)
        else:
            vectors[:, n_spurions:] = torch.einsum(
                "...jk,...k->...j", jet_boost, vectors[:, n_spurions:]
            )
    elif cfg_data.canonicalize is None:
        pass
    else:
        raise ValueError(f"canonicalize option {cfg_data.canonicalize} not implemented")
    vectors[~mask] = 0.0
    scalars[~mask] = 0.0

    # precompute tagging features
    momentum = vectors[..., 0, :]
    jet = momentum[:, n_spurions:].sum(dim=1, keepdim=True)
    auxiliary_scalars = get_auxiliary_scalars(
        momentum,
        jet,
        auxiliary_scalars=cfg_data.auxiliary_scalars,
    )
    auxiliary_scalars[:, :n_spurions] = 0.0
    auxiliary_scalars[~mask] = 0.0
    auxiliary_scalars = auxiliary_scalars.to(scalars.dtype)

    return [vectors, scalars, auxiliary_scalars, is_spurion, mask]


def dense_to_sparse(dense_tensors, mask):
    assert len(dense_tensors) > 0
    device = dense_tensors[0].device

    num_particles = mask.sum(dim=-1)
    ptr = torch.zeros(len(num_particles) + 1, device=device, dtype=torch.long)
    ptr[1:] = torch.cumsum(num_particles, dim=0)
    idxs = mask.flatten().nonzero().squeeze(-1)
    batch = get_batch_from_ptr(ptr, num_items=idxs.shape[0])

    sparse_tensors = []
    for dense_tensor in dense_tensors:
        if dense_tensor.numel() > 0:
            sparse_tensor = dense_tensor.flatten(0, 1).index_select(0, idxs)
        else:
            sparse_tensor = torch.zeros(
                idxs.shape[0],
                *dense_tensor.shape[2:],
                device=dense_tensor.device,
                dtype=dense_tensor.dtype,
            )
        sparse_tensors.append(sparse_tensor)
    return sparse_tensors, batch, ptr


@functools.cache
def get_spurion(
    beam_reference,
    add_time_reference,
    two_beams,
    device,
    dtype,
):
    """
    Construct spurion. Cached, so callers must not mutate the returned tensor.

    Parameters
    ----------
    beam_reference: str
        Different options for adding a beam_reference
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    device
    dtype

    Returns
    -------
    spurion: torch.tensor with shape (n_spurions, 4)
        spurion embedded as fourmomenta object
    """

    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, device=device, dtype=dtype).reshape(1, 4)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 3] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)
    elif beam_reference == "all":
        beam = torch.tensor(
            [
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
            ],
            device=device,
            dtype=dtype,
        )

    elif beam_reference is None:
        beam = torch.empty(0, 4, device=device, dtype=dtype)

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, device=device, dtype=dtype).reshape(1, 4)
    else:
        time = torch.empty(0, 4, device=device, dtype=dtype)

    spurion = torch.cat((beam, time), dim=-2)
    return spurion


def get_auxiliary_scalars(fourmomenta, jet, auxiliary_scalars="all", eps=1e-10):
    """
    Compute features typically used in jet tagging

    Parameters
    ----------
    fourmomenta: torch.tensor of shape (n_particles, 4)
        Fourmomenta in the format (E, px, py, pz)
    jet: torch.tensor of shape (n_particles, 4)
        Jet momenta in the shape (E, px, py, pz)
    auxiliary_scalars: str
        Type of tagging features to include. Options are None, 'all', 'zinvariant', 'so3invariant'.
        Note that all features are SO(2)-invariant.
    eps: float

    Returns
    -------
    features: torch.tensor of shape (n_particles, 7)
        Features: log_pt, log_energy, log_pt_rel, log_energy_rel, dphi, deta, dr
    """
    log_pt = get_pt(fourmomenta).unsqueeze(-1).log()
    log_energy = fourmomenta[..., 0].unsqueeze(-1).clamp(min=eps).log()

    log_pt_rel = (get_pt(fourmomenta).log() - get_pt(jet).log()).unsqueeze(-1)
    log_energy_rel = (
        fourmomenta[..., 0].clamp(min=eps).log() - jet[..., 0].clamp(min=eps).log()
    ).unsqueeze(-1)
    phi_4, phi_jet = get_phi(fourmomenta), get_phi(jet)
    dphi = ((phi_4 - phi_jet + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    eta_4, eta_jet = get_eta(fourmomenta), get_eta(jet)
    deta = -(eta_4 - eta_jet).unsqueeze(-1)
    dr = torch.sqrt((dphi**2 + deta**2).clamp(min=eps))
    features = [
        log_pt,
        log_energy,
        log_pt_rel,
        log_energy_rel,
        dphi,
        deta,
        dr,
    ]
    for i, feature in enumerate(features):
        mean, factor = AUXILIARY_SCALARS_PREPROCESSING[i]
        features[i] = (feature - mean) * factor
    if auxiliary_scalars == "zinvariant":
        # exclude energy, because it is not invariant under z-boosts
        idx = [0, 2, 4, 5, 6]
    elif auxiliary_scalars == "so3invariant":
        # exclude everything except energy, because it is not invariant under SO(3) rotations
        idx = [1, 3]
    elif auxiliary_scalars is None:
        return torch.zeros(
            *features[0].shape[:-1], 0, device=fourmomenta.device, dtype=fourmomenta.dtype
        )
    elif auxiliary_scalars == "all":
        idx = list(range(len(features)))
    else:
        raise ValueError(f"auxiliary_scalars={auxiliary_scalars} not implemented")
    features = [features[i] for i in idx]
    features = torch.cat(features, dim=-1)
    return features


def get_num_auxiliary_scalars(auxiliary_scalars="all"):
    if auxiliary_scalars == "all":
        return 7
    elif auxiliary_scalars == "zinvariant":
        return 5
    elif auxiliary_scalars == "so3invariant":
        return 2
    elif auxiliary_scalars is None:
        return 0
    else:
        raise ValueError(f"auxiliary_scalars={auxiliary_scalars} not implemented")
