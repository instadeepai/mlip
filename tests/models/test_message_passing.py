import e3j
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import pytest
from e3j.utils.options import Layout

from mlip.models.blocks import SO3Convolution


class _TestSO3Convolution:
    """Check that MessagePassing output is equivariant under SO(3) rotations.

    Tests the identity g(f(x)) == f(g(x)) where g is a rotation:
    - Equivariant inputs (node_feats, spherical_embedding) are rotated.
    - Invariant inputs (mix, senders, receivers) are left unchanged.
    """

    node_irreps: e3nn.Irreps
    l_max: int

    key = jax.random.key(42)
    n_node: int = 10
    n_edge: int = 25
    avg_num_neighbors: float = 2.5

    @pytest.fixture(scope="class")
    def inputs(self):
        y_lm_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)

        node_feats = e3nn.IrrepsArray(
            self.node_irreps,
            jax.random.normal(self.key, (self.n_node, self.node_irreps.dim)),
        )
        harmonics = e3nn.IrrepsArray(
            y_lm_irreps,
            jax.random.normal(jax.random.key(1), (self.n_edge, y_lm_irreps.dim)),
        )

        # Determine output irreps via e3j to get correct mix shape
        target_irreps = e3nn.Irreps(sorted({ir for _, ir in self.node_irreps}))
        tp = e3j.core.TensorProduct(
            source=(str(self.node_irreps), str(y_lm_irreps)),
            target=target_irreps,
        )
        output_irreps = e3nn.Irreps(str(tp.target))

        mix = jax.random.normal(
            jax.random.key(2), (self.n_edge, output_irreps.num_irreps)
        )
        senders = jax.random.randint(jax.random.key(3), (self.n_edge,), 0, self.n_node)
        receivers = jax.random.randint(
            jax.random.key(4), (self.n_edge,), 0, self.n_node
        )

        return dict(
            node_feats=node_feats,
            harmonics=harmonics,
            output_irreps=output_irreps,
            mix=mix,
            senders=senders,
            receivers=receivers,
            target_irreps=target_irreps,
        )

    def test_equivariance(self, inputs):
        mp = SO3Convolution(
            source_irreps=(
                inputs["node_feats"].irreps,
                inputs["harmonics"].irreps,
            ),
            target_irreps=inputs["target_irreps"],
            avg_num_neighbors=self.avg_num_neighbors,
            layout=Layout.E3NN,
        )

        rotation = e3nn.rand_matrix(self.key)
        output_irreps = inputs["output_irreps"]

        # g(f(x)): apply module then rotate output
        output = mp(
            inputs["node_feats"].array,
            inputs["harmonics"].array,
            inputs["mix"],
            inputs["senders"],
            inputs["receivers"],
        )
        D_out = output_irreps.D_from_matrix(rotation)
        gfx = output @ D_out

        # f(g(x)): rotate equivariant inputs then apply module
        D_node = inputs["node_feats"].irreps.D_from_matrix(rotation)
        D_ylm = inputs["harmonics"].irreps.D_from_matrix(rotation)

        fgx = mp(
            inputs["node_feats"].array @ D_node,
            inputs["harmonics"].array @ D_ylm,
            inputs["mix"],
            inputs["senders"],
            inputs["receivers"],
        )

        norm = float(jnp.sqrt(jnp.sum((gfx - fgx) ** 2)))
        norm /= gfx.size
        assert norm < 5e-5, f"Equivariance error: {norm}"

    def test_layout_consistent(self, inputs):
        """SO3Convolution commutes with LEADING <-> TRAILING cast."""
        n_channels = 4
        y_lm_irreps = e3nn.Irreps.spherical_harmonics(self.l_max)

        conv_leading = SO3Convolution(
            source_irreps=(self.node_irreps, y_lm_irreps),
            target_irreps=inputs["target_irreps"],
            avg_num_neighbors=self.avg_num_neighbors,
            layout=Layout.LEADING_CHANNELS,
        )
        conv_trailing = SO3Convolution(
            source_irreps=(self.node_irreps, y_lm_irreps),
            target_irreps=inputs["target_irreps"],
            avg_num_neighbors=self.avg_num_neighbors,
            layout=Layout.TRAILING_CHANNELS,
        )

        k1, k2, k3 = jax.random.split(jax.random.key(99), 3)
        node_feats_lc = jax.random.normal(
            k1, (self.n_node, n_channels, self.node_irreps.dim)
        )
        y_lm = jax.random.normal(k2, (self.n_edge, y_lm_irreps.dim))

        tp = e3j.core.TensorProduct(
            source=(str(self.node_irreps), str(y_lm_irreps)),
            target=inputs["target_irreps"],
        )
        output_irreps = e3nn.Irreps(str(tp.target))
        n_scalars = n_channels * output_irreps.num_irreps
        mix = jax.random.normal(k3, (self.n_edge, n_scalars))

        y_lc = conv_leading(
            node_feats_lc,
            y_lm,
            mix,
            inputs["senders"],
            inputs["receivers"],
        )

        node_feats_tc = jnp.swapaxes(node_feats_lc, -1, -2)
        mix_tc = jnp.swapaxes(
            mix.reshape(self.n_edge, n_channels, -1),
            -1,
            -2,
        ).reshape(self.n_edge, -1)
        y_tc = conv_trailing(
            node_feats_tc,
            y_lm,
            mix_tc,
            inputs["senders"],
            inputs["receivers"],
        )

        norm = float(jnp.sqrt(jnp.sum((jnp.swapaxes(y_lc, -1, -2) - y_tc) ** 2)))
        norm /= y_tc.size
        assert norm < 5e-5, f"Layout consistency error: {norm}"


class TestMessagePassingL2(_TestSO3Convolution):
    node_irreps = e3nn.Irreps("0e + 1o + 2e")
    l_max = 2


class TestConvolutionL3(_TestSO3Convolution):
    node_irreps = e3nn.Irreps("0e + 1o + 2e + 3o")
    l_max = 3


class TestConvolutionScalarsOnly(_TestSO3Convolution):
    node_irreps = e3nn.Irreps("0e")
    l_max = 1
