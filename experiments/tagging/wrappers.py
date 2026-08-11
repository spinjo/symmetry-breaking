import torch
import torch.distributed as dist
from lgatr import embed_vector, extract_scalar
from lloca.framesnet.frames import Frames
from lloca.framesnet.nonequi_frames import IdentityFrames
from lloca.reps.tensorreps import TensorReps
from lloca.reps.tensorreps_transform import TensorRepsTransform
from lloca.utils.utils import (
    get_batch_from_ptr,
    get_edge_attr,
    get_edge_index_from_ptr,
    get_ptr_from_batch,
)
from torch import nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import scatter, to_dense_batch

from experiments.misc import get_attention_mask
from experiments.tagging.embedding import (
    AUXILIARY_SCALARS_PREPROCESSING,
    dense_to_sparse,
    get_auxiliary_scalars,
)


class LLoCaWrapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        framesnet,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.framesnet = framesnet
        self.trafo_fourmomenta = TensorRepsTransform(TensorReps("1x1n"))

    def init_standardization(self, vectors_dense, mask, is_spurion=None):
        # framesnet equivectors edge_attr standardization (if applicable)
        if hasattr(self.framesnet, "equivectors") and hasattr(
            self.framesnet.equivectors, "init_standardization"
        ):
            fourmomenta_dense = vectors_dense[..., 0, :]
            [fourmomenta_sparse], _, ptr = dense_to_sparse([fourmomenta_dense], mask)
            self.framesnet.equivectors.init_standardization(fourmomenta_sparse, ptr)

    def forward(
        self,
        fourmomenta_spurions,
        scalars_spurions,
        auxiliary_scalars_spurions,
        is_spurion,
        batch_spurions,
        ptr_spurions,
        num_graphs,
    ):
        # FramesNet forward pass; uses sparse tensor representation

        # remove spurions and recompute attributes
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)
        fourmomenta_nospurions = fourmomenta_spurions.index_select(0, nospurion_idxs)
        scalars_nospurions = scalars_spurions.index_select(0, nospurion_idxs)
        batch_nospurions = batch_spurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions, num_graphs=num_graphs)
        B = ptr_nospurions.numel() - 1

        scalars_spurions = torch.cat([auxiliary_scalars_spurions, scalars_spurions], dim=-1)
        frames_spurions, tracker = self.framesnet(
            fourmomenta_spurions,
            scalars_spurions,
            ptr=ptr_spurions,
            return_tracker=True,
            num_graphs=num_graphs,
        )
        matrices = frames_spurions.matrices.index_select(0, nospurion_idxs)
        frames_nospurions = Frames(
            matrices,
            is_global=frames_spurions.is_global,
            det=frames_spurions.det.index_select(0, nospurion_idxs),
            inv=frames_spurions.inv.index_select(0, nospurion_idxs),
            is_identity=frames_spurions.is_identity,
            device=frames_spurions.device,
            dtype=frames_spurions.dtype,
            shape=matrices.shape,
        )

        # transform features into local frames
        fourmomenta_local_nospurions = self.trafo_fourmomenta(
            fourmomenta_nospurions, frames_nospurions
        )
        jet_nospurions = scatter(
            fourmomenta_nospurions,
            index=batch_nospurions,
            dim=0,
            reduce="sum",
            dim_size=B,
        )
        jet_local_nospurions = self.trafo_fourmomenta(
            jet_nospurions.index_select(0, batch_nospurions), frames_nospurions
        )
        local_auxiliary_scalars_nospurions = get_auxiliary_scalars(
            fourmomenta_local_nospurions,
            jet_local_nospurions,
            auxiliary_scalars="all",
        )

        features_local_nospurions = torch.cat(
            [local_auxiliary_scalars_nospurions, scalars_nospurions], dim=-1
        )

        # change dtype (see embedding.py fourmomenta_float64 option)
        features_local_nospurions = features_local_nospurions.to(scalars_nospurions.dtype)
        jet_nospurions = jet_nospurions.to(scalars_nospurions.dtype)
        frames_nospurions.to(scalars_nospurions.dtype)

        return (
            features_local_nospurions,
            fourmomenta_local_nospurions,
            frames_nospurions,
            ptr_nospurions,
            batch_nospurions,
            tracker,
            jet_nospurions,
        )


class TransformerWrapper(LLoCaWrapper):
    def __init__(
        self,
        net,
        *args,
        use_amp: bool = False,
        attention_backend: str = "xformers",
        mean_aggregation: bool = False,
        zeropad: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_amp = use_amp
        self.attention_backend = attention_backend
        self.mean_aggregation = mean_aggregation
        self.zeropad = zeropad
        self.net = net(in_channels=self.in_channels, out_channels=self.out_channels)
        if mean_aggregation and not zeropad:
            self.aggregator = MeanAggregation()

        if attention_backend == "flex":
            compile_flex_attention(package_name="lloca")

    def _forward_sparse(
        self, fourmomenta, scalars, auxiliary_scalars, is_spurion, batch, ptr, num_graphs, maxlen
    ):
        # precompute attention mask to avoid cudaStreamSynchronize
        # from .tolist() in get_xformers_attention_mask
        batch_spurions = batch
        ptr_spurions = ptr
        nospurion_idxs = (~is_spurion).nonzero(as_tuple=False).squeeze(-1)
        batch_nospurions = batch_spurions.index_select(0, nospurion_idxs)
        ptr_nospurions = get_ptr_from_batch(batch_nospurions, num_graphs=num_graphs)
        ptr, batch = ptr_nospurions, batch_nospurions
        if not self.mean_aggregation:
            batchsize = len(ptr) - 1
            ptr = ptr.clone()
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr, num_items=nospurion_idxs.shape[0] + batchsize)
            maxlen = maxlen + 1
        mask_kwarg = get_attention_mask(
            batch,
            dtype=scalars.dtype,
            attention_backend=self.attention_backend,
            ptr=ptr,
            maxlen=maxlen,
        )

        (features_local, _, frames, ptr, batch, tracker, jet) = super().forward(
            fourmomenta,
            scalars,
            auxiliary_scalars,
            is_spurion,
            batch_spurions,
            ptr_spurions,
            num_graphs,
        )

        # handle global token
        if not self.mean_aggregation:
            # append global tokens to batch, ptr, features_local and frames
            batchsize = len(ptr) - 1
            num_total = features_local.shape[0] + batchsize
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            nonglobal_idxs = torch.arange(features_local.shape[0], device=batch.device) + batch + 1
            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr, num_items=num_total)

            new_features = torch.zeros(
                num_total,
                features_local.shape[-1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            new_features[nonglobal_idxs, :-1] = features_local
            new_features[:, -1].index_fill_(0, global_idxs, 1.0)
            features_local = new_features

            # global token frames are identity
            matrices_new = torch.eye(4, device=frames.device, dtype=frames.dtype)
            matrices_new = matrices_new.unsqueeze(0).expand(num_total, -1, -1).clone()
            matrices_new[nonglobal_idxs] = frames.matrices
            det_new = torch.ones(num_total, device=frames.device, dtype=frames.dtype)
            det_new[nonglobal_idxs] = frames.det
            inv_new = torch.eye(4, device=frames.device, dtype=frames.dtype)
            inv_new = inv_new.unsqueeze(0).expand(num_total, -1, -1).clone()
            inv_new[nonglobal_idxs] = frames.inv
            frames = Frames(
                matrices_new,
                is_global=frames.is_global,
                det=det_new,
                inv=inv_new,
                is_identity=frames.is_identity,
                device=frames.device,
                dtype=frames.dtype,
                shape=matrices_new.shape,
            )

        features_local = features_local.unsqueeze(0)
        frames = frames.reshape(1, *frames.shape)
        with torch.autocast(features_local.device.type, enabled=self.use_amp):
            outputs = self.net(
                inputs=features_local,
                frames=frames,
                p_ref=jet,
                ptr=ptr,
                **mask_kwarg,
            )
        outputs = outputs.squeeze(0)

        # aggregation
        if self.mean_aggregation:
            B = ptr.numel() - 1
            score = self.aggregator(outputs, index=batch, dim_size=B)
        else:
            score = outputs.index_select(0, global_idxs)
        return score, tracker, frames

    def _forward_dense(
        self, fourmomenta, scalars, auxiliary_scalars, is_spurion, batch, ptr, num_graphs
    ):
        (features_local, _, frames, ptr, batch, tracker, jet) = super().forward(
            fourmomenta,
            scalars,
            auxiliary_scalars,
            is_spurion,
            batch,
            ptr,
            num_graphs,
        )

        features_local, mask = to_dense_batch(features_local, batch)
        frames_matrices, _ = to_dense_batch(frames.matrices, batch)
        frames_inv, _ = to_dense_batch(frames.inv, batch)
        frames_det, _ = to_dense_batch(frames.det, batch)
        eye = torch.eye(4, device=frames.device, dtype=frames.dtype)
        frames_matrices[~mask] = eye
        frames_inv[~mask] = eye
        frames_det[~mask] = 1.0
        frames = Frames(
            matrices=frames_matrices,
            inv=frames_inv,
            det=frames_det,
            is_global=frames.is_global,
            is_identity=frames.is_identity,
            device=frames.device,
            dtype=frames.dtype,
            shape=frames_matrices.shape,
        )

        if not self.mean_aggregation:
            new_features = torch.zeros(
                features_local.shape[0],
                features_local.shape[1] + 1,
                features_local.shape[2] + 1,
                device=features_local.device,
                dtype=features_local.dtype,
            )
            new_features[:, 1:, :-1] = features_local
            new_features[:, 0, -1] = 1.0
            features_local = new_features

            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
            matrices_global = (
                torch.eye(4, device=frames.device, dtype=frames.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(features_local.shape[0], 1, 1, 1)
            )
            det_global = torch.ones(
                (features_local.shape[0], 1), device=frames.device, dtype=frames.dtype
            )
            frames = Frames(
                torch.cat([matrices_global, frames.matrices], dim=1),
                is_global=frames.is_global,
                det=torch.cat([det_global, frames.det], dim=1),
                inv=torch.cat([matrices_global, frames.inv], dim=1),
            )

        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        with torch.autocast(features_local.device.type, enabled=self.use_amp):
            outputs = self.net(
                inputs=features_local,
                frames=frames,
                p_ref=jet,
                attn_mask=attn_mask,
            )
        outputs[~mask] = 0.0

        if self.mean_aggregation:
            score = outputs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
        else:
            score = outputs[:, 0]
        return score, tracker, frames

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        fourmomenta = vectors[..., 0, :]
        if isinstance(self.framesnet, IdentityFrames):
            # shortcut for non-LLoCa transformer
            features = torch.cat([auxiliary_scalars, scalars], dim=-1)

            if self.zeropad:
                if not self.mean_aggregation:
                    new_features = torch.zeros(
                        features.shape[0],
                        features.shape[1] + 1,
                        features.shape[2] + 1,
                        device=features.device,
                        dtype=features.dtype,
                    )
                    new_features[:, 1:, :-1] = features
                    new_features[:, 0, -1] = 1.0
                    features = new_features
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)

                frames = Frames(
                    is_identity=True,
                    device=features.device,
                    dtype=features.dtype,
                    shape=features.shape[:-1],
                )

                attn_mask = mask.unsqueeze(1).unsqueeze(2)
                with torch.autocast(features.device.type, enabled=self.use_amp):
                    outputs = self.net(inputs=features, frames=frames, attn_mask=attn_mask)
                outputs[~mask] = 0.0

                if self.mean_aggregation:
                    score = outputs.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
                else:
                    score = outputs[:, 0]
                return score, {}, frames

            else:
                [features], batch, ptr = dense_to_sparse([features], mask)
                if not self.mean_aggregation:
                    batchsize = len(ptr) - 1
                    num_total = features.shape[0] + batchsize
                    global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
                    nonglobal_idxs = (
                        torch.arange(features.shape[0], device=batch.device) + batch + 1
                    )
                    ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
                    batch = get_batch_from_ptr(ptr, num_items=num_total)

                    new_features = torch.zeros(
                        num_total,
                        features.shape[-1] + 1,
                        dtype=scalars.dtype,
                        device=scalars.device,
                    )
                    new_features[nonglobal_idxs, :-1] = features
                    new_features[:, -1].index_fill_(0, global_idxs, 1.0)
                    features = new_features

                frames = Frames(
                    is_identity=True,
                    device=features.device,
                    dtype=features.dtype,
                    shape=features.shape[:-1],
                )
                mask_kwargs = get_attention_mask(
                    batch,
                    dtype=scalars.dtype,
                    attention_backend=self.attention_backend,
                    ptr=ptr,
                    maxlen=mask.shape[1] + (0 if self.mean_aggregation else 1),
                )
                features = features.unsqueeze(0)
                frames = frames.reshape(1, *frames.shape)
                with torch.autocast(features.device.type, enabled=self.use_amp):
                    outputs = self.net(inputs=features, frames=frames, **mask_kwargs)
                outputs = outputs.squeeze(0)

                # aggregation
                if self.mean_aggregation:
                    B = ptr.numel() - 1
                    score = self.aggregator(outputs, index=batch, dim_size=B)
                else:
                    score = outputs.index_select(0, global_idxs)
                return score, {}, frames

        else:
            # full LLoCa experience
            num_graphs = fourmomenta.shape[0]
            features_dense = [fourmomenta, scalars, auxiliary_scalars, is_spurion]
            features_sparse, batch, ptr = dense_to_sparse(features_dense, mask)
            [fourmomenta, scalars, auxiliary_scalars, is_spurion] = features_sparse

            if self.zeropad:
                return self._forward_dense(
                    fourmomenta, scalars, auxiliary_scalars, is_spurion, batch, ptr, num_graphs
                )
            else:
                return self._forward_sparse(
                    fourmomenta,
                    scalars,
                    auxiliary_scalars,
                    is_spurion,
                    batch,
                    ptr,
                    num_graphs,
                    maxlen=mask.shape[1],
                )


class ParticleNetWrapper(LLoCaWrapper):
    def __init__(
        self,
        net: callable,
        *args,
        zeropad: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert zeropad, "ParticleNet only supports zero-padding"
        self.zeropad = zeropad
        self.net = net(input_dims=self.in_channels, num_classes=self.out_channels)

    def forward(self, *embedding_list):
        embedding_list = (embedding_list[0][..., 0, :], *embedding_list[1:])
        if isinstance(self.framesnet, IdentityFrames):
            # shortcut for non-LLoCa ParticleNet
            _, scalars_local, auxiliary_scalars_local, _, mask = embedding_list
            features_local = torch.cat([auxiliary_scalars_local, scalars_local], dim=-1)
            frames = Frames(
                is_identity=True,
                device=features_local.device,
                dtype=features_local.dtype,
                shape=(features_local.shape[0] * features_local.shape[1],),
            )
            tracker = {}
        else:
            num_graphs = embedding_list[0].shape[0]
            mask = embedding_list[-1]
            embedding_list_sparse, batch, ptr = dense_to_sparse(embedding_list[:-1], mask)

            features_local, _, frames, _, batch, tracker, _ = super().forward(
                *embedding_list_sparse, batch, ptr, num_graphs
            )

            features_local, mask = to_dense_batch(features_local, batch)
            dense_matrices, _ = to_dense_batch(frames.matrices, batch)
            dense_det, _ = to_dense_batch(frames.det, batch)
            dense_inv, _ = to_dense_batch(frames.inv, batch)
            eye = torch.eye(4, device=frames.device, dtype=frames.dtype)
            dense_matrices[~mask] = eye
            dense_inv[~mask] = eye
            dense_det[~mask] = 1.0
            frames = Frames(
                dense_matrices.view(-1, 4, 4),
                is_global=frames.is_global,
                det=dense_det.view(-1),
                inv=dense_inv.view(-1, 4, 4),
                is_identity=frames.is_identity,
            )

        phieta_local = features_local[..., [4, 5]]  # ParticleNet uses L2 norm in (phi, eta) for kNN
        phieta_local = phieta_local.transpose(1, 2)
        features_local = features_local.transpose(1, 2)
        mask = mask.unsqueeze(1)

        score = self.net(
            points=phieta_local,
            features=features_local,
            frames=frames,
            mask=mask,
        )
        return score, tracker, frames


class ParTWrapper(LLoCaWrapper):
    def __init__(
        self,
        net: callable,
        *args,
        use_amp: bool = False,
        zeropad: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert zeropad, "ParT only supports zero-padding"
        self.zeropad = zeropad
        self.use_amp = use_amp
        self.net = net(input_dim=self.in_channels, num_classes=self.out_channels)

    def forward(self, *embedding_list):
        embedding_list = (embedding_list[0][..., 0, :], *embedding_list[1:])
        if isinstance(self.framesnet, IdentityFrames):
            # shortcut for non-LLoCa ParT
            fourmomenta_local, scalars_local, auxiliary_scalars_local, _, mask = embedding_list
            features_local = torch.cat([auxiliary_scalars_local, scalars_local], dim=-1)
            # reference (jet) momentum; identity frames => local == global
            jet = (fourmomenta_local * mask.unsqueeze(-1)).sum(dim=1)
            frames = Frames(
                is_identity=True,
                device=features_local.device,
                dtype=features_local.dtype,
                shape=features_local.shape[:-1],
            )
            tracker = {}
        else:
            num_graphs = embedding_list[0].shape[0]
            mask = embedding_list[-1]
            embedding_list_sparse, batch, ptr = dense_to_sparse(embedding_list[:-1], mask)

            features_local, fourmomenta_local, frames, _, batch, tracker, jet = super().forward(
                *embedding_list_sparse, batch, ptr, num_graphs
            )

            features_local, mask = to_dense_batch(features_local, batch)
            fourmomenta_local, _ = to_dense_batch(fourmomenta_local, batch)
            frames_matrices, _ = to_dense_batch(frames.matrices, batch)
            det, _ = to_dense_batch(frames.det, batch)
            inv, _ = to_dense_batch(frames.inv, batch)
            eye = torch.eye(4, device=frames.device, dtype=frames.dtype)
            frames_matrices[~mask] = eye
            inv[~mask] = eye
            det[~mask] = 1.0
            frames = Frames(
                matrices=frames_matrices,
                is_global=frames.is_global,
                det=det,
                inv=inv,
                is_identity=frames.is_identity,
            )

        fourmomenta_local = fourmomenta_local.to(features_local.dtype)
        fourmomenta_local = fourmomenta_local[..., [1, 2, 3, 0]]  # ParT expects (px, py, pz, E)
        features_local = features_local.transpose(1, 2)
        fourmomenta_local = fourmomenta_local.transpose(1, 2)
        mask = mask.unsqueeze(1).float()

        with torch.autocast(features_local.device.type, enabled=self.use_amp):
            score = self.net(
                x=features_local,
                frames=frames,
                v=fourmomenta_local,
                mask=mask,
                p_ref=jet,
            )
        return score, tracker, frames


class LGATrWrapper(nn.Module):
    def __init__(
        self,
        net: callable,
        framesnet: nn.Module,
        out_channels: int,
        mean_aggregation: bool = False,
        use_amp: bool = False,
        attention_backend: str = "xformers",
        units: int = 1,
        zeropad: bool = False,
    ):
        super().__init__()
        self.use_amp = use_amp
        self.units = units
        self.attention_backend = attention_backend
        self.zeropad = zeropad
        self._init_net(net, out_channels)
        self.mean_aggregation = mean_aggregation
        if mean_aggregation and not zeropad:
            self.aggregator = MeanAggregation()

        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

        if attention_backend == "flex":
            compile_flex_attention(package_name="lgatr")

    def _init_net(self, net, out_channels):
        self.net = net(out_mv_channels=out_channels)

    def _forward_sparse(self, vectors, scalars, batch, ptr, maxlen):
        # handle global token
        if not self.mean_aggregation:
            maxlen = maxlen + 1
            batchsize = len(ptr) - 1
            num_total = vectors.shape[0] + batchsize
            global_idxs = ptr[:-1] + torch.arange(batchsize, device=batch.device)
            nonglobal_idxs = torch.arange(vectors.shape[0], device=batch.device) + batch + 1

            new_fm = torch.zeros(
                num_total,
                *vectors.shape[1:],
                dtype=vectors.dtype,
                device=vectors.device,
            )
            new_fm[nonglobal_idxs] = vectors
            vectors = new_fm

            new_s = torch.zeros(
                num_total,
                scalars.shape[1] + 1,
                dtype=scalars.dtype,
                device=scalars.device,
            )
            new_s[nonglobal_idxs, : scalars.shape[1]] = scalars
            new_s[:, -1].index_fill_(0, global_idxs, 1.0)
            scalars = new_s

            ptr[1:] = ptr[1:] + (torch.arange(batchsize, device=ptr.device) + 1)
            batch = get_batch_from_ptr(ptr, num_items=num_total)

        vectors = vectors.unsqueeze(0)
        scalars = scalars.unsqueeze(0)

        mask_kwarg = get_attention_mask(
            batch,
            dtype=scalars.dtype,
            attention_backend=self.attention_backend,
            ptr=ptr,
            maxlen=maxlen,
        )

        with torch.autocast(vectors.device.type, enabled=self.use_amp):
            out = self._call_network(vectors, scalars, **mask_kwarg)
        out = out.squeeze(0)

        if self.mean_aggregation:
            B = ptr.numel() - 1
            logits = self.aggregator(out, index=batch, dim_size=B)
        else:
            logits = out.index_select(0, global_idxs)
        return logits, {}, None

    def _forward_dense(self, vectors, scalars, mask):
        if not self.mean_aggregation:
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)
            vectors = torch.cat([torch.zeros_like(vectors[:, :1]), vectors], dim=1)
            new_s = torch.zeros(
                scalars.shape[0],
                scalars.shape[1] + 1,
                scalars.shape[2] + 1,
                device=scalars.device,
                dtype=scalars.dtype,
            )
            new_s[:, 1:, :-1] = scalars
            new_s[:, 0, -1] = 1.0
            scalars = new_s

        attn_mask = mask.unsqueeze(1).unsqueeze(2)
        with torch.autocast(vectors.device.type, enabled=self.use_amp):
            out = self._call_network(vectors, scalars, attn_mask=attn_mask)
        out[~mask] = 0.0

        if self.mean_aggregation:
            logits = out.sum(dim=-2) / mask.sum(dim=-1, keepdim=True)
        else:
            logits = out[:, 0]
        return logits, {}, None

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        momentum, extra = vectors[..., 0, :], vectors[..., 1:, :]

        # units are four-momentum-specific -> applied to channel 0 only
        momentum = momentum * torch.where(is_spurion.unsqueeze(-1), 1.0, 1.0 / self.units)
        vectors = torch.cat([momentum.unsqueeze(2), extra], dim=2).to(scalars.dtype)
        scalars = torch.cat([auxiliary_scalars, scalars], dim=-1)

        if not self.zeropad:
            [vectors, scalars], batch, ptr = dense_to_sparse([vectors, scalars], mask)
            return self._forward_sparse(vectors, scalars, batch, ptr, maxlen=mask.shape[1])
        else:
            return self._forward_dense(vectors, scalars, mask)

    def _call_network(self, vectors, scalars, **mask_kwarg):
        mv = embed_vector(vectors)
        s = scalars if scalars.shape[-1] > 0 else None
        mv_outputs, _ = self.net(mv, s, **mask_kwarg)
        out = extract_scalar(mv_outputs)[..., 0]
        return out


class LGATrSlimWrapper(LGATrWrapper):
    def _init_net(self, net: callable, out_channels: int):
        self.net = net(out_s_channels=out_channels)

    def _call_network(self, vectors, scalars, **mask_kwarg):
        _, out = self.net(vectors, scalars, **mask_kwarg)
        return out


class MIParTWrapper(nn.Module):
    def __init__(
        self,
        net: callable,
        framesnet: nn.Module,
        in_channels: int,
        out_channels: int,
        use_amp: bool = False,
        zeropad: bool = True,
    ):
        super().__init__()
        assert zeropad, "MI-ParT only supports zero-padding"
        self.zeropad = zeropad
        self.net = net(input_dim=in_channels, num_classes=out_channels, use_amp=use_amp)

        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        assert is_spurion.sum() == 0
        features = torch.cat([auxiliary_scalars, scalars], dim=-1)
        fourmomenta = vectors[..., 0, :]
        fourmomenta = fourmomenta.to(auxiliary_scalars.dtype)
        fourmomenta = fourmomenta[..., [1, 2, 3, 0]]  # ParT expects (px, py, pz, E)

        features = features.transpose(1, 2)
        fourmomenta = fourmomenta.transpose(1, 2)
        mask = mask.unsqueeze(1).float()

        # network
        score = self.net(
            x=features,
            v=fourmomenta,
            mask=mask,
        )
        return score, {}, None


class LorentzNetWrapper(nn.Module):
    def __init__(
        self,
        net: callable,
        framesnet: nn.Module,
        out_channels: int,
        units: int = 1,
        zeropad: bool = False,
    ):
        super().__init__()
        self.net = net(n_class=out_channels)
        self.units = units
        assert not zeropad, "LorentzNet does not support zero-padding"

        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        fourmomenta = vectors[..., 0, :]
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / self.units
        scalars = torch.cat([auxiliary_scalars, scalars], dim=-1)

        [fourmomenta, scalars], batch, ptr = dense_to_sparse([fourmomenta, scalars], mask)

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=True)
        fourmomenta = fourmomenta.to(scalars.dtype)
        output = self.net(scalars, fourmomenta, edges=edge_index, batch=batch)
        return output, {}, None


class PELICANLiteWrapper(nn.Module):
    def __init__(
        self,
        net: callable,
        framesnet: nn.Module,
        out_channels: int,
        units: int = 1,
        zeropad: bool = False,
    ):
        super().__init__()
        assert not zeropad, "PELICAN-lite does not support zero-padding"
        self.net = net(out_channels=out_channels)
        self.units = units

        self.register_buffer("edge_inited", torch.tensor(False))
        self.register_buffer("edge_mean", torch.tensor(0.0))
        self.register_buffer("edge_std", torch.tensor(1.0))

        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        fourmomenta = vectors[..., 0, :]
        fourmomenta[~is_spurion] = fourmomenta[~is_spurion] / self.units
        scalars = torch.cat([auxiliary_scalars, scalars], dim=-1)
        num_graphs = scalars.shape[0]

        [fourmomenta, scalars], batch, ptr = dense_to_sparse([fourmomenta, scalars], mask)

        edge_index = get_edge_index_from_ptr(ptr, fourmomenta.shape, remove_self_loops=False)
        fourmomenta = fourmomenta.to(scalars.dtype)
        edge_attr = self.get_edge_attr(fourmomenta, edge_index).to(scalars.dtype)
        output = self.net(
            in_rank2=edge_attr,
            edge_index=edge_index,
            batch=batch,
            in_rank1=scalars,
            num_graphs=num_graphs,
        )
        return output, {}, None

    def get_edge_attr(self, fourmomenta, edge_index):
        edge_attr = get_edge_attr(fourmomenta, edge_index)
        if not self.edge_inited:
            self.edge_mean = edge_attr.mean().detach()
            self.edge_std = edge_attr.std().clamp(min=1e-5).detach()
            if dist.is_available() and dist.is_initialized():
                # broadcast rank-0 stats so every rank normalizes identically
                dist.broadcast(self.edge_mean, src=0)
                dist.broadcast(self.edge_std, src=0)
            self.edge_inited = torch.tensor(True, device=edge_attr.device)
        edge_attr = (edge_attr - self.edge_mean) / self.edge_std
        return edge_attr.unsqueeze(-1)


class SaltWrapper(nn.Module):
    """Wrapper class for the Salt model v0.12 (https://gitlab.cern.ch/aft/algorithms/salt)"""

    def __init__(
        self,
        net: callable,
        in_channels: int,
        out_channels: int,
        framesnet: nn.Module,
        global_object: str = "jets",
        zeropad: bool = False,
        use_amp: bool = False,
        compile: bool = False,
        compile_mode: str = "default",
        compile_dynamic: bool = True,
    ):
        super().__init__()
        self.net = net
        self.zeropad = zeropad
        self.use_amp = use_amp

        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

        assert self.net.init_nets[0].net.input_size == in_channels
        assert self.net.tasks[0].net.output_size == out_channels

        # propagate metadata to tasks
        self.global_object = global_object
        self.net.global_object = self.global_object
        for task in self.net.tasks:
            task.global_object = self.net.global_object
            task.model_name = "salt"

        if compile:
            self.net.forward = torch.compile(
                self.net.forward, dynamic=compile_dynamic, mode=compile_mode, fullgraph=zeropad
            )

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        features = torch.cat([auxiliary_scalars, scalars], dim=-1)
        features = {"tracks": features, self.global_object: None}
        pad_mask = {"pad_mask": ~mask}  # True where padded
        amp = self.use_amp or not self.zeropad  # flash-varlen requires fp16/bf16
        with torch.autocast(scalars.device.type, enabled=amp):
            preds, _ = self.net(features, pad_masks=pad_mask)
        out = preds[self.global_object]["jets_classification"]
        return out, {}, None


class PET2Wrapper(nn.Module):
    def __init__(
        self,
        net: callable,
        framesnet: nn.Module,
        in_channels: int,
        out_channels: int,
        use_amp: bool = False,
        zeropad: bool = True,
    ):
        super().__init__()
        assert zeropad, "PET2 only supports zero-padding"
        self.zeropad = zeropad
        self.use_amp = use_amp
        self.net = net(input_dim=in_channels, num_classes=out_channels)

        self.framesnet = framesnet
        assert isinstance(framesnet, IdentityFrames)

    def forward(self, vectors, scalars, auxiliary_scalars, is_spurion, mask):
        assert is_spurion.sum() == 0
        mean_logpt, std_logpt = AUXILIARY_SCALARS_PREPROCESSING[0]
        auxiliary_scalars[..., 0] = auxiliary_scalars[..., 0] / std_logpt + mean_logpt
        auxiliary_scalars[..., :7] = auxiliary_scalars[
            ..., [5, 4, 0, 1, 2, 3, 6]
        ]  # need (eta, phi, logpt) first for local feature evaluation
        features = torch.cat([auxiliary_scalars, scalars], dim=-1)

        with torch.autocast(features.device.type, enabled=self.use_amp):
            results = self.net(
                x=features,
                y=None,
            )
        score = results["y_pred"]
        return score, {}, None


def compile_flex_attention(package_name="lgatr"):
    """Run torch.compile on the flex_attention function.

    However, as of today (Dec 2025, pytorch 2.9.0), torch.compile + flex_attention
    for variable-length sequences only works in a few cases:
    - CPU: Forward pass is supported, but backward pass not (https://github.com/pytorch/pytorch/issues/169224)
      To still let the code run through for tests, we skip torch.compile on CPU.
      This way the code runs through, but is super slow because it materializes the attention matrix.
      Note that we use essentially the same approach for xformers, where we fall back to default torch attention on CPU.
      On the plus side, flex_attention supports arbitrary head_dim if torch.compile is not used.
    - GPU: The docs say that only head dimensions being powers of 2 are supported.
      However, on my system only head_dim=2**n with n>=4 works, i.e. head_dim=16,32,...
      Setting head_dim=2,4,8 gives cryptic errors.
      Moreover, transformers with flex_attention are still significantly slower than
      transformers with xformers attention in our implementation.
    """
    if package_name == "lgatr":
        import lgatr.primitives.attention_backends.flex as flex
    elif package_name == "lloca":
        import lloca.backbone.attention_backends.flex as flex
    else:
        raise ValueError(f"Unknown package {package_name}")

    if torch.cuda.is_available():
        # max-autotune strongly recommended for flex-attention with variable-length sequences,
        # see https://pytorch.org/blog/flexattention-for-inference/
        flex.attention = torch.compile(
            flex.attention,
            dynamic=True,
        )
