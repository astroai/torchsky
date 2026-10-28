"""
Tests for spectral and data cube functionality.
"""

import os

# Add src to path for testing
import tempfile

import numpy as np
import pytest
import unittest.mock
import torch
import torchfits

from torchsky.spectral import (
    DataCube,
    SpectralAxis,
    SpectralReader,
    Spectrum1D,
    flux_conserving_rebin,
    relativistic_doppler_shift,
    spectral_axis_from_header,
    spectral_bin_edges_from_header,
)


class TestSpectralAxis:
    """Test SpectralAxis functionality."""

    def test_wavelength_axis(self):
        """Test wavelength spectral axis."""
        wavelengths = torch.linspace(4000, 7000, 1000)  # Angstroms
        axis = SpectralAxis(wavelengths, "Angstrom", "WAVE")

        assert torch.allclose(axis.to_wavelength(), wavelengths)

        # Test frequency conversion
        freq = axis.to_frequency()
        assert freq.shape == wavelengths.shape
        assert torch.all(freq > 0)

    def test_frequency_axis(self):
        """Test frequency spectral axis."""
        frequencies = torch.linspace(1e14, 2e14, 1000)  # Hz
        axis = SpectralAxis(frequencies, "Hz", "FREQ")

        assert torch.allclose(axis.to_frequency(), frequencies)

        # Test wavelength conversion
        wave = axis.to_wavelength()
        assert wave.shape == frequencies.shape
        assert torch.all(wave > 0)

    def test_unit_conversions(self):
        """Test unit conversions."""
        # Test nm to Angstrom
        wave_nm = torch.tensor([400.0, 500.0, 600.0])  # nm
        axis = SpectralAxis(wave_nm, "nm", "WAVE")
        wave_angstrom = axis.to_wavelength()
        expected = wave_nm * 10.0
        assert torch.allclose(wave_angstrom, expected)

        freq_ghz = torch.tensor([100.0, 200.0], dtype=torch.float64)
        axis = SpectralAxis(freq_ghz, "GHz", "FREQ")
        expected_wave = 299_792_458.0e10 / (freq_ghz * 1.0e9)
        torch.testing.assert_close(axis.to_wavelength(), expected_wave)

        freq_mhz = torch.tensor([100.0, 200.0], dtype=torch.float64)
        axis = SpectralAxis(freq_mhz, "MHz", "FREQ")
        expected_freq = freq_mhz * 1.0e6
        torch.testing.assert_close(axis.to_frequency(), expected_freq)

        axis = SpectralAxis(freq_ghz, "ghz", "freq")
        torch.testing.assert_close(axis.to_wavelength(), expected_wave)

        wave_nm = torch.tensor([400.0, 500.0], dtype=torch.float64)
        torch.testing.assert_close(
            SpectralAxis(wave_nm, "nanometer", "WAVE").to_wavelength(),
            wave_nm * 10.0,
        )
        torch.testing.assert_close(
            SpectralAxis(torch.tensor([0.4], dtype=torch.float64), "micrometer", "WAVE").to_wavelength(),
            torch.tensor([4000.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            SpectralAxis(torch.tensor([4.0e-5], dtype=torch.float64), "cm", "WAVE").to_wavelength(),
            torch.tensor([4000.0], dtype=torch.float64),
        )
        joule = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(
            SpectralAxis(joule, "joule", "ENER").to_frequency(),
            joule / 6.626_070_15e-34,
        )
        with pytest.raises(ValueError, match="unsupported spectral unit"):
            SpectralAxis(torch.tensor([1.0]), "furlong", "WAVE").to_wavelength()

    def test_energy_wavenumber_and_differentiability(self):
        energy = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        axis = SpectralAxis(energy, "keV", "ENER")
        frequency = axis.to_frequency()
        torch.testing.assert_close(axis.to_energy("keV"), energy)
        torch.testing.assert_close(
            axis.to_wavenumber("1/m"),
            frequency / 299_792_458.0,
        )
        axis.to_wavelength().sum().backward()
        assert energy.grad is not None
        assert bool(torch.isfinite(energy.grad).all())

        wavenumber = SpectralAxis(
            torch.tensor([1.0, 2.0], dtype=torch.float64), "cm^-1", "WAVN"
        )
        torch.testing.assert_close(
            wavenumber.to_frequency(),
            wavenumber.values * 100.0 * 299_792_458.0,
        )
        header_axis, tensor_axis = spectral_axis_from_header(
            {
                "NAXIS": 1,
                "NAXIS1": 2,
                "CTYPE1": "WAVN",
                "CUNIT1": "cm^-1",
                "CRVAL1": 1.0,
                "CDELT1": 1.0,
                "CRPIX1": 1.0,
            }
        )
        assert tensor_axis == 0
        assert header_axis.quantity == "wavenumber"
        torch.testing.assert_close(
            header_axis.to_frequency(),
            torch.tensor([100.0, 200.0], dtype=torch.float64) * 299_792_458.0,
        )

    def test_awav_uses_air_refraction(self):
        air = torch.tensor([5000.0, 6563.0], dtype=torch.float64)
        axis = SpectralAxis(air, "Angstrom", "AWAV")
        from torchsky.wcs.spectral import air_wavelength_to_vacuum, spectral_to_frequency

        vac = air_wavelength_to_vacuum(air, unit="Angstrom")
        torch.testing.assert_close(axis.to_wavelength(), vac)
        torch.testing.assert_close(
            axis.to_frequency(),
            spectral_to_frequency(air, "AWAV", unit="Angstrom"),
        )

    def test_irregular_descending_axis_contract(self):
        values = torch.tensor([8.0, 5.0, 3.0], dtype=torch.float64)
        edges = torch.tensor([9.5, 6.5, 4.0, 2.0], dtype=torch.float64)
        axis = SpectralAxis(values, "GHz", "FREQ", bin_edges=edges)

        assert axis.quantity == "frequency"
        assert axis.direction == -1
        assert not axis.is_regular()
        torch.testing.assert_close(axis.edges(), edges)
        torch.testing.assert_close(
            axis.channel_widths(), torch.tensor([3.0, 2.5, 2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            axis.quadrature_weights(),
            torch.tensor([1.5, 2.5, 1.0], dtype=torch.float64),
        )

        selected, mask = axis.select_interval(3.0, 5.0)
        torch.testing.assert_close(
            selected.values, torch.tensor([5.0, 3.0], dtype=torch.float64)
        )
        torch.testing.assert_close(selected.bin_edges, edges[1:])
        torch.testing.assert_close(mask, torch.tensor([False, True, True]))
        transferred = selected.to(dtype=torch.float32)
        assert transferred.values.dtype == torch.float32
        assert transferred.bin_edges is not None
        assert transferred.bin_edges.dtype == torch.float32

    def test_axis_validation_and_unsupported_conversion(self):
        with pytest.raises(ValueError, match="duplicates"):
            _ = SpectralAxis(torch.tensor([1.0, 1.0]), "Hz", "FREQ").direction
        with pytest.raises(ValueError, match="strictly monotonic"):
            _ = SpectralAxis(torch.tensor([1.0, 3.0, 2.0]), "Hz", "FREQ").direction

        axis = SpectralAxis(torch.tensor([1.0, 2.0]), "m/s", "VELO")
        with pytest.raises(ValueError, match="rest_frequency or rest_wavelength"):
            _ = axis.to_wavelength()

    def test_linear_spectral_axis_from_header(self):
        header = {
            "NAXIS": 3,
            "NAXIS1": 16,
            "NAXIS2": 8,
            "NAXIS3": 4,
            "CTYPE3": "WAVE",
            "CUNIT3": "Angstrom",
            "CRVAL3": 5000.0,
            "CDELT3": 2.0,
            "CRPIX3": 1.0,
        }

        axis, tensor_axis_index = spectral_axis_from_header(header)
        edges, edge_axis_index = spectral_bin_edges_from_header(header)

        assert tensor_axis_index == 0
        assert edge_axis_index == 0
        torch.testing.assert_close(
            axis.values,
            torch.tensor([5000.0, 5002.0, 5004.0, 5006.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            edges,
            torch.tensor([4999.0, 5001.0, 5003.0, 5005.0, 5007.0], dtype=torch.float64),
        )


class TestSpectrum1D:
    """Test 1D spectrum functionality."""

    def create_test_spectrum(self, n_wave=1000, with_ivar=True, with_mask=True):
        """Create test spectrum."""
        wavelengths = torch.linspace(4000, 7000, n_wave)
        flux = torch.sin(wavelengths / 100) + torch.randn(n_wave) * 0.1

        spectral_axis = SpectralAxis(wavelengths, "Angstrom", "WAVE")

        ivar = None
        if with_ivar:
            ivar = torch.ones_like(flux) * 100  # High S/N

        mask = None
        if with_mask:
            mask = torch.ones_like(flux, dtype=torch.bool)
            # Mask out some bad pixels
            mask[100:110] = False
            mask[500:505] = False

        return Spectrum1D(flux, spectral_axis, ivar, mask)

    def test_spectrum_creation(self):
        """Test spectrum creation and validation."""
        spectrum = self.create_test_spectrum()

        assert spectrum.flux.ndim == 1
        assert len(spectrum.spectral_axis.values) == len(spectrum.flux)
        assert spectrum.ivar.shape == spectrum.flux.shape
        assert spectrum.mask.shape == spectrum.flux.shape

    def test_wavelength_frequency_access(self):
        """Test wavelength and frequency access."""
        spectrum = self.create_test_spectrum()

        wavelength = spectrum.wavelength
        frequency = spectrum.frequency

        assert wavelength.shape == spectrum.flux.shape
        assert frequency.shape == spectrum.flux.shape
        assert torch.all(wavelength > 0)
        assert torch.all(frequency > 0)

    def test_error_calculation(self):
        """Test error calculation from inverse variance."""
        spectrum = self.create_test_spectrum()

        error = spectrum.error
        assert error is not None
        assert error.shape == spectrum.flux.shape

        # Check that error = 1/sqrt(ivar)
        expected_error = torch.sqrt(1.0 / spectrum.ivar)
        assert torch.allclose(error, expected_error)

    def test_mask_application(self):
        """Test mask application."""
        spectrum = self.create_test_spectrum()
        original_length = len(spectrum.flux)

        masked_spectrum = spectrum.apply_mask()

        # Should have fewer pixels after masking
        assert len(masked_spectrum.flux) < original_length
        assert len(masked_spectrum.spectral_axis.values) == len(masked_spectrum.flux)

        # All remaining pixels should be good
        assert torch.all(masked_spectrum.mask)

    def test_resampling(self):
        """Test spectrum resampling."""
        spectrum = self.create_test_spectrum()

        # Create new wavelength grid
        new_wavelength = torch.linspace(4500, 6500, 500)
        resampled = spectrum.resample(new_wavelength)

        assert len(resampled.flux) == len(new_wavelength)
        assert torch.allclose(resampled.wavelength, new_wavelength)

        # Check that flux values are reasonable
        assert torch.all(torch.isfinite(resampled.flux))

    def test_resampling_uses_linear_interpolation(self):
        """Spectrum resampling should interpolate instead of taking nearest samples."""
        spectrum = Spectrum1D(
            flux=torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64),
            spectral_axis=SpectralAxis(
                torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
                "Angstrom",
                "WAVE",
            ),
            ivar=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        )

        resampled = spectrum.resample(torch.tensor([0.5, 1.5], dtype=torch.float64))
        torch.testing.assert_close(
            resampled.flux,
            torch.tensor([5.0, 15.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            resampled.ivar,
            torch.tensor([2.0, 2.0], dtype=torch.float64),
        )


class TestSpectralPrimitives:
    def test_irregular_rebin_conserves_integrated_flux(self):
        input_edges = torch.tensor([1.0, 2.0, 4.0, 7.0], dtype=torch.float64)
        output_edges = torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64)
        flux = torch.full((3,), 2.0, dtype=torch.float64)

        result = flux_conserving_rebin(flux, input_edges, output_edges)

        torch.testing.assert_close(
            result.flux, torch.tensor([2.0, 2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            (result.flux * output_edges.diff()).sum(),
            (flux * input_edges.diff()).sum(),
        )
        assert result.mask is not None and bool(result.mask.all())

    def test_rebin_propagates_independent_inverse_variance(self):
        result = flux_conserving_rebin(
            torch.tensor([1.0, 3.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.0, 2.0], dtype=torch.float64),
            ivar=torch.tensor([4.0, 4.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            result.flux, torch.tensor([2.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.ivar, torch.tensor([8.0], dtype=torch.float64)
        )
        assert result.mask is not None and bool(result.mask.all())

    def test_rebin_marks_masked_or_outside_bins_invalid_without_renormalizing(self):
        result = flux_conserving_rebin(
            torch.tensor([2.0, 10.0, 4.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
            torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64),
            ivar=torch.ones(3, dtype=torch.float64),
            mask=torch.tensor([True, False, True]),
        )

        torch.testing.assert_close(
            result.flux,
            torch.tensor([0.0, 2.0, 0.0, 4.0], dtype=torch.float64),
        )
        assert result.mask is not None
        torch.testing.assert_close(
            result.mask, torch.tensor([False, True, False, True])
        )
        torch.testing.assert_close(
            result.ivar,
            torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float64),
        )

    def test_relativistic_shift_scales_edges_and_flux_density(self):
        velocity = 0.6 * 299_792_458.0  # Doppler factor = 2.
        result = relativistic_doppler_shift(
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            velocity,
            ivar=torch.tensor([4.0, 9.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            result.spectral_edges,
            torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            result.flux, torch.tensor([0.5, 0.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            result.ivar, torch.tensor([16.0, 36.0], dtype=torch.float64)
        )

    def test_relativistic_shift_rebins_batches_on_a_common_grid(self):
        flux = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
        velocity = torch.tensor([0.0, 0.6 * 299_792_458.0], dtype=torch.float64)
        result = relativistic_doppler_shift(
            flux,
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            velocity,
            output_edges=torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            result.flux,
            torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64),
        )
        assert result.mask is not None
        torch.testing.assert_close(
            result.mask,
            torch.tensor([[False, False], [True, True]]),
        )

    def test_rebin_is_differentiable_with_respect_to_flux(self):
        flux = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        result = flux_conserving_rebin(
            flux,
            torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.0, 0.5, 1.5, 2.0], dtype=torch.float64),
        )
        result.flux.sum().backward()
        assert flux.grad is not None and bool(torch.isfinite(flux.grad).all())

    def test_rebin_matches_overlap_matrix_reference(self):
        generator = torch.Generator().manual_seed(7)
        input_edges = torch.cat(
            [
                torch.zeros(1, dtype=torch.float64),
                torch.rand(17, generator=generator, dtype=torch.float64).cumsum(0),
            ]
        )
        output_edges = torch.linspace(
            input_edges[0], input_edges[-1], 12, dtype=torch.float64
        )
        flux = torch.randn(3, 17, generator=generator, dtype=torch.float64)

        result = flux_conserving_rebin(flux, input_edges, output_edges)
        overlap = (
            torch.minimum(output_edges[1:, None], input_edges[None, 1:])
            - torch.maximum(output_edges[:-1, None], input_edges[None, :-1])
        ).clamp_min(0)
        reference = torch.einsum("...n,mn->...m", flux, overlap) / output_edges.diff()

        torch.testing.assert_close(result.flux, reference, atol=1e-12, rtol=1e-12)

    def test_relativistic_shift_rejects_superluminal_velocity(self):
        with pytest.raises(ValueError, match="below c"):
            relativistic_doppler_shift(
                torch.ones(2),
                torch.tensor([1.0, 2.0, 3.0]),
                299_792_458.0,
            )


class TestDataCube:
    """Test 3D data cube functionality."""

    def create_test_cube(self, shape=(100, 64, 64), spectral_axis_index=0):
        """Create test data cube."""
        data = torch.randn(shape)

        # Create spectral axis
        n_spectral = shape[spectral_axis_index]
        wavelengths = torch.linspace(4000, 7000, n_spectral)
        spectral_axis = SpectralAxis(wavelengths, "Angstrom", "WAVE")

        return DataCube(data, spectral_axis, spectral_axis_index=spectral_axis_index)

    def test_cube_creation(self):
        """Test cube creation and validation."""
        cube = self.create_test_cube()

        assert cube.data.ndim == 3
        assert cube.n_spectral == cube.data.shape[cube.spectral_axis_index]
        assert len(cube.spectral_axis.values) == cube.n_spectral

    def test_different_spectral_axes(self):
        """Test cubes with spectral axis in different positions."""
        for axis_idx in [0, 1, 2]:
            shape = [100, 64, 64]
            cube = self.create_test_cube(tuple(shape), axis_idx)

            assert cube.spectral_axis_index == axis_idx
            assert cube.n_spectral == shape[axis_idx]

    def test_spatial_shape(self):
        """Test spatial shape with each supported spectral axis position."""
        cube0 = self.create_test_cube((100, 64, 32), spectral_axis_index=0)
        cube1 = self.create_test_cube((64, 100, 32), spectral_axis_index=1)
        cube2 = self.create_test_cube((64, 32, 100), spectral_axis_index=2)

        assert cube0.spatial_shape == (64, 32)
        assert cube1.spatial_shape == (64, 32)
        assert cube2.spatial_shape == (64, 32)

    def test_spectrum_extraction(self):
        """Test extracting 1D spectra from cube."""
        cube = self.create_test_cube()

        # Extract spectrum at center
        y, x = 32, 32
        spectrum = cube.extract_spectrum(y, x)

        assert isinstance(spectrum, Spectrum1D)
        assert len(spectrum.flux) == cube.n_spectral
        assert torch.allclose(spectrum.spectral_axis.values, cube.spectral_axis.values)

    def test_spectrum_extraction_preserves_yx_for_middle_spectral_axis(self):
        data = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        axis = SpectralAxis(torch.tensor([4000.0, 5000.0, 6000.0]), "Angstrom", "WAVE")
        cube = DataCube(data, axis, spectral_axis_index=1)

        spectrum = cube.extract_spectrum(y=1, x=2)

        torch.testing.assert_close(spectrum.flux, data[1, :, 2])

    def test_spectrum_extraction_preserves_yx_for_last_spectral_axis(self):
        data = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        axis = SpectralAxis(torch.tensor([4000.0, 5000.0, 6000.0]), "Angstrom", "WAVE")
        cube = DataCube(data, axis, spectral_axis_index=2)

        spectrum = cube.extract_spectrum(y=1, x=2)

        torch.testing.assert_close(spectrum.flux, data[1, 2, :])

    def test_spectral_collapse(self):
        """Test collapsing cube along spectral axis."""
        cube = self.create_test_cube()

        # Test different collapse methods
        for method in ["mean", "sum", "median", "max"]:
            collapsed = cube.collapse_spectral(method)

            expected_shape = list(cube.shape)
            expected_shape.pop(cube.spectral_axis_index)
            assert collapsed.shape == tuple(expected_shape)

    def test_wavelength_slice(self):
        """Test extracting 2D slice at specific wavelength."""
        cube = self.create_test_cube()

        # Extract slice at middle wavelength
        mid_wavelength = cube.spectral_axis.values[cube.n_spectral // 2]
        slice_2d = cube.extract_slice(mid_wavelength.item())

        expected_shape = list(cube.shape)
        expected_shape.pop(cube.spectral_axis_index)
        assert slice_2d.shape == tuple(expected_shape)

    def test_device_transfer(self):
        """Test moving cube to different device."""
        cube = self.create_test_cube()

        # Test CPU (should be no-op)
        cpu_cube = cube.to(torch.device("cpu"))
        assert cpu_cube.data.device == torch.device("cpu")

        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_cube = cube.to(torch.device("cuda"))
            assert cuda_cube.data.device.type == "cuda"


class TestSpectralReader:
    """Test reading spectral data from FITS files."""

    def create_test_spectrum_file(self):
        """Create test spectrum FITS file."""
        n_wave = 1000
        wavelength = np.linspace(4000, 7000, n_wave)
        flux = np.sin(wavelength / 100) + np.random.normal(0, 0.1, n_wave)
        ivar = np.ones_like(flux) * 100
        mask = np.ones_like(flux, dtype=bool)
        mask[100:110] = False

        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        torchfits.table.write(
            path,
            {
                "WAVELENGTH": wavelength,
                "FLUX": flux,
                "IVAR": ivar,
                "MASK": mask,
            },
            overwrite=True,
        )
        return path

    def create_test_cube_file(self):
        """Create test data cube FITS file."""
        shape = (100, 64, 64)
        data = np.random.randn(*shape).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        torchfits.write(
            path,
            torch.from_numpy(data),
            header={
                "CTYPE3": "WAVE",
                "CRVAL3": 4000.0,
                "CDELT3": 30.0,
                "CRPIX3": 1.0,
                "CUNIT3": "Angstrom",
            },
            overwrite=True,
        )
        return path

    def test_read_spectrum_1d(self):
        """Test reading 1D spectrum from FITS table."""
        filepath = self.create_test_spectrum_file()

        try:
            spectrum = SpectralReader.read_spectrum_1d(filepath, hdu=1)

            assert isinstance(spectrum, Spectrum1D)
            assert spectrum.flux.ndim == 1
            assert spectrum.ivar is not None
            assert spectrum.mask is not None

            # Check wavelength range
            wavelength = spectrum.wavelength
            assert wavelength[0] >= 4000
            assert wavelength[-1] <= 7000

        finally:
            os.unlink(filepath)

    @unittest.mock.patch("torchsky.spectral.WCS")
    @unittest.mock.patch("torchsky.spectral.spectral_axis_from_header")
    @unittest.mock.patch("torchfits.open")
    def test_read_data_cube_wcs_error(
        self, mock_open, mock_spectral_axis, mock_wcs_class
    ):
        """Test read_data_cube handles WCS creation failure gracefully."""
        mock_hdul = unittest.mock.MagicMock()
        mock_open.return_value.__enter__.return_value = mock_hdul

        mock_hdu = unittest.mock.MagicMock()
        mock_hdul.__getitem__.return_value = mock_hdu

        # Setup dummy data and header
        mock_hdu.to_tensor.return_value = torch.ones((10, 10, 10))
        mock_hdu.header = {"TEST": 1}

        # Setup dummy spectral axis to pass post-init checks
        mock_axis = unittest.mock.MagicMock()
        mock_axis.values = torch.ones(10)
        mock_spectral_axis.return_value = (mock_axis, 0)

        # Make WCS.from_header raise an exception
        mock_wcs_class.from_header.side_effect = Exception("WCS Error")

        # Should not raise exception
        cube = SpectralReader.read_data_cube("fake_path.fits")

        assert isinstance(cube, DataCube)
        assert cube.spatial_wcs is None

    def test_read_data_cube(self):
        """Test reading 3D data cube from FITS image."""
        filepath = self.create_test_cube_file()

        try:
            cube = SpectralReader.read_data_cube(filepath)

            assert isinstance(cube, DataCube)
            assert cube.data.ndim == 3
            assert cube.spectral_axis_index == 0  # Should detect wavelength axis

            # Check spectral axis
            wavelength = cube.spectral_axis.to_wavelength()
            assert wavelength[0] >= 4000
            assert wavelength[-1] <= 7000

        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
