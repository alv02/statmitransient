from typing import Sequence

import drjit as dr
import mitsuba as mi
import numpy as np
from mitsuba import Float, Int32, ScalarInt32, TensorXf

from mitransient.render.transient_image_block import TransientImageBlock

from ..utils import BOX_COX_TRANSFORMATION, YEO_JOHNSON_TRANSFORMATION


class TransientHDRFilm(mi.Film):
    r"""

    .. film-transient_hdr_film:

    Transient HDR Film (:monosp:`transient_hdr_film`)
    -------------------------------------------------

    mitransient's equivalent to Mitsuba 3's HDRFilm

    Stores two image blocks simultaneously:

    * Steady block: Accumulates all samples (sum over all the time dimension)
    * Transient block: Accumulates samples separating them in time bins (histogram)

    The results can be retrieved using the ``develop(raw=True)`` method, which returns a (steady, transient) tuple.

    .. pluginparameters::

     * - temporal_bins
       - |int|
       - number of bins in the time dimension (histogram representation)

     * - bin_width_opl
       - |float|
       - width of each bin in the time dimension (histogram representation)

     * - start_opl
       - |float|
       - start of the time dimension (histogram representation)

    See also, from `mi.Film <https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_films.html>`_:

    * `width` (integer)
    * `height` (integer)
    * `crop_width` (integer)
    * `crop_height` (integer)
    * `crop_offset_x` (integer)
    * `crop_offset_y` (integer)
    * `sample_border` (bool)
    * `rfilter` (rfilter)
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        self.temporal_bins = props.get("temporal_bins", mi.UInt32(2048))
        self.bin_width_opl = props.get("bin_width_opl", mi.Float(0.003))
        self.start_opl = props.get("start_opl", mi.Float(0))
        self.transformation = props.get(
            "transformation", int(YEO_JOHNSON_TRANSFORMATION)
        )
        self.lambda_ = props.get("lambda", float(0.5))

    def end_opl(self):
        return self.start_opl + self.bin_width_opl * self.temporal_bins

    def base_channels_count(self):
        return self.steady.base_channels_count()

    def prepare(self, aovs: Sequence[str], spp: int):
        # Prepare steady film
        steady_hdrfilm_dict = {
            "type": "hdrfilm",
            "width": self.size().x,
            "height": self.size().y,
            "pixel_format": "luminance" if mi.is_monochromatic else "rgb",
            "crop_offset_x": self.crop_offset().x,
            "crop_offset_y": self.crop_offset().y,
            "crop_width": self.crop_size().x,
            "crop_height": self.crop_size().y,
            "sample_border": self.sample_border(),
            "rfilter": self.rfilter(),
        }
        self.steady: mi.Film = mi.load_dict(steady_hdrfilm_dict)
        self.steady.prepare(aovs)

        # Prepare transient image block
        channels = self.prepare_transient_(aovs, spp)
        return channels

    def prepare_transient_(self, aovs: Sequence[str], spp: int):
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)

        if mi.is_monochromatic:
            base_channels = "LAW" if alpha else "LW"
        else:
            # RGB
            base_channels = "RGBAW" if alpha else "RGBW"

        channels = []
        for i in range(len(base_channels)):
            channels.append(base_channels[i])

        for i in range(len(aovs)):
            channels.append(aovs[i])

        crop_offset_xyt = mi.ScalarPoint3i(
            self.crop_offset().x, self.crop_offset().y, 0
        )
        crop_size_xyt = mi.ScalarVector3u(
            self.size().x, self.size().y, self.temporal_bins
        )

        self.transient_storage = TransientImageBlock(
            size_xyt=crop_size_xyt,
            offset_xyt=crop_offset_xyt,
            channel_count=len(channels),
            rfilter=self.rfilter(),
            spp=spp,
            lambda_=self.lambda_,
            transformation=self.transformation,
        )
        self.channels = channels

        if len(set(channels)) != len(channels):
            mi.Log(
                mi.LogLevel.Error, "Film::prepare_transient_(): duplicate channel name."
            )

        return len(self.channels)

    def clear(self):
        self.steady.clear()

        if self.transient_storage:
            self.transient_storage.clear()

    def develop(self, raw: bool = False, total_spp=0):
        steady_image = self.steady.develop(raw=raw)
        transient_image = self.develop_transient_(raw=raw)
        stats = self.develop_stats(total_spp)

        return steady_image, transient_image, stats

    def gather_tensor(self, data):
        pixel_count = dr.prod(data.shape[0:-1])
        source_ch = data.shape[-1]
        # Remove alpha and weight channels
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        target_ch = source_ch - (ScalarInt32(2) if alpha else ScalarInt32(1))

        idx = dr.arange(Int32, pixel_count * target_ch)
        pixel_idx = idx // target_ch
        channel_idx = dr.fma(pixel_idx, -target_ch, idx)

        values_idx = dr.fma(pixel_idx, source_ch, channel_idx)

        values_ = dr.gather(Float, data.array, values_idx)

        return TensorXf(values_, tuple(list(data.shape[0:-1]) + [target_ch]))

    def develop_stats(self, total_spp):
        count = self.gather_tensor(self.transient_storage.count_tensor)
        media = self.gather_tensor(self.transient_storage.tensor)
        sum1 = self.gather_tensor(self.transient_storage.sum1_tensor)
        sum2 = self.gather_tensor(self.transient_storage.sum2_tensor)
        sum3 = self.gather_tensor(self.transient_storage.sum3_tensor)

        samples_bounce = total_spp * 17
        missing = samples_bounce - count
        # Box-Cox of zero (0 transformed)
        if self.transformation == YEO_JOHNSON_TRANSFORMATION:
            zero_transformed = self.transient_storage.yeo_johnson(0.0)
        elif self.transformation == BOX_COX_TRANSFORMATION:
            zero_transformed = self.transient_storage.box_cox(0.0)
        else:
            zero_transformed = 0.0

        # Fill accumulators with zeros
        sum1 += missing * zero_transformed
        sum2 += missing * zero_transformed**2
        sum3 += missing * zero_transformed**3

        # Calculate stats with accumulated non central moments
        mu = sum1 / samples_bounce

        var = (
            sum2 / samples_bounce
        ) - mu**2  # (sum2 - (samples_bounce * mu**2)) / (samples_bounce - 1)
        var = dr.select(var < 0.0, 0.0, var)

        m3 = (sum3 / samples_bounce) - (3 * mu * var) - (mu**3)

        # When the variance is 0 there is no need for skewness correction
        estimands = dr.select(var == 0.0, mu, mu + m3 / (6 * var * samples_bounce))
        estimands_variance = var / samples_bounce

        # mu = dr.select(count > 0, sum1 / count, zero_transformed)
        #
        # var = dr.select(count > 1, (sum2 - (count * mu**2)) / (count - 1), 0)
        # var = dr.select(var < 0.0, 0.0, var)
        #
        # m3 = dr.select(count > 0, (sum3 / count) - (3 * mu * var) - (mu**3), 0)
        #
        # estimands = dr.select(
        #     (var > 0.0) & (count > 0), mu + m3 / (6 * var * count), mu
        # )
        # estimands_variance = var / samples_bounce
        print(
            "Min Mean Max estimands: ",
            dr.min(estimands),
            " ",
            dr.mean(estimands),
            " ",
            dr.max(estimands),
        )
        print(
            "Min Mean Max estimands_variance: ",
            dr.min(estimands_variance),
            " ",
            dr.mean(estimands_variance),
            " ",
            dr.max(estimands_variance),
        )
        print(
            "Min Mean Max m3: ",
            dr.min(m3),
            " ",
            dr.mean(m3),
            " ",
            dr.max(m3),
        )
        # Crear tensor total_spp con la misma forma que estimands

        return (estimands, estimands_variance)

    def develop_transient_(self, raw: bool = False):
        if not self.transient_storage:
            mi.Log(
                mi.LogLevel.Error,
                "No transient storage allocated, was prepare_transient_() called first?",
            )

        if raw:
            return self.transient_storage.tensor

        data = self.transient_storage.tensor

        pixel_count = dr.prod(data.shape[0:-1])
        source_ch = data.shape[-1]
        # Remove alpha and weight channels
        alpha = mi.has_flag(self.flags(), mi.FilmFlags.Alpha)
        target_ch = source_ch - (ScalarInt32(2) if alpha else ScalarInt32(1))

        idx = dr.arange(Int32, pixel_count * target_ch)
        pixel_idx = idx // target_ch
        channel_idx = dr.fma(pixel_idx, -target_ch, idx)

        values_idx = dr.fma(pixel_idx, source_ch, channel_idx)
        weight_idx = dr.fma(pixel_idx, source_ch, source_ch - 1)

        weight = dr.gather(Float, data.array, weight_idx)
        values_ = dr.gather(Float, data.array, values_idx)

        values = values_ / dr.select((weight == 0.0), 1.0, weight)

        return TensorXf(values, tuple(list(data.shape[0:-1]) + [target_ch]))

    def add_transient_data(
        self,
        pos: mi.Vector2f,
        distance: mi.Float,
        wavelengths: mi.UnpolarizedSpectrum,
        spec: mi.Spectrum,
        spec_: mi.Spectrum,
        ray_weight: mi.Float,
        active: mi.Bool,
    ):
        """
        Add a path's contribution to the film:
        * pos: pixel position
        * distance: distance traveled by the path (opl)
        * wavelengths: for spectral rendering, wavelengths sampled
        * spec: Spectrum / contribution of the path
        * ray_weight: weight of the ray given by the sensor
        * active: mask
        """
        pos_distance = (distance - self.start_opl) / self.bin_width_opl

        coords = mi.Vector3f(pos.x, pos.y, pos_distance)
        mask = (pos_distance >= 0) & (pos_distance < self.temporal_bins)

        self.transient_storage.put(
            pos=coords,
            wavelengths=wavelengths,
            value=spec * ray_weight,
            value_=spec_ * ray_weight,
            alpha=mi.Float(0.0),
            # value should have the sample scale already multiplied
            weight=mi.Float(0.0),
            active=active & mask,
        )

    def update_stats(
        self,
        value: mi.Spectrum,
        transient_pos: mi.UInt32,
        pos: mi.Vector2f,
        ray_weight,
        active,
    ):
        mask = (transient_pos >= 0) & (transient_pos < self.temporal_bins)
        self.transient_storage.update_stats(
            value * ray_weight, transient_pos, pos, active & mask
        )

    def to_string(self):
        string = "TransientHDRFilm[\n"
        string += f"  size = {self.size()},\n"
        string += f"  crop_size = {self.crop_size()},\n"
        string += f"  crop_offset = {self.crop_offset()},\n"
        string += f"  sample_border = {self.sample_border()},\n"
        string += f"  filter = {self.rfilter()},\n"
        string += f"  temporal_bins = {self.temporal_bins},\n"
        string += f"  bin_width_opl = {self.bin_width_opl},\n"
        string += f"  start_opl = {self.start_opl},\n"
        string += f"]"
        return string

    def traverse(self, callback):
        super().traverse(callback)
        callback.put_parameter(
            "temporal_bins", self.temporal_bins, mi.ParamFlags.NonDifferentiable
        )
        callback.put_parameter(
            "bin_width_opl", self.bin_width_opl, mi.ParamFlags.NonDifferentiable
        )
        callback.put_parameter(
            "start_opl", self.start_opl, mi.ParamFlags.NonDifferentiable
        )

    def parameters_changed(self, keys):
        super().parameters_changed(keys)


mi.register_film("transient_hdr_film", lambda props: TransientHDRFilm(props))
