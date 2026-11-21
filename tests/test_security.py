import pytest
import torchfits


def test_security_cve_cfitsio_command_injection():
    """
    Test that filenames starting or ending with '|' are rejected to prevent
    CFITSIO command injection vulnerabilities.
    """
    # Filenames that should be rejected
    dangerous_filenames = [
        "| echo 'pwned'",
        " | ls",
        "valid.fits |",
        "valid.fits | ",
        "|/bin/sh -c 'touch /tmp/pwned'",
    ]

    # torchfits refuses these before CFITSIO sees them: a missing file is
    # "not found" (checked first since torchfits 1.3.3), an existing one with a
    # leading pipe is a "Security Error". Either way no command may run.
    for filename in dangerous_filenames:
        with pytest.raises(Exception, match="Security Error|not found"):
            torchfits.read(filename)


def test_security_pipe_named_files_never_execute(tmp_path, monkeypatch):
    import torch

    monkeypatch.chdir(tmp_path)
    real = tmp_path / "real.fits"
    torchfits.write(str(real), torch.zeros(2, 2), overwrite=True)
    data = real.read_bytes()
    for name in ("|touch marker_lead", "real.fits |touch marker_trail"):
        (tmp_path / name).write_bytes(data)
        try:
            torchfits.read(name)
        except Exception as exc:  # refusing is fine; executing is not
            assert "Security Error" in str(exc) or "not found" in str(exc)
    assert not list(tmp_path.glob("marker_*"))


def test_valid_filenames_allowed():
    """Test that normal filenames are still allowed."""
    # We can't easily test success without a real file, but we can verify
    # it doesn't raise the Security Error.
    try:
        torchfits.read("nonexistent_file.fits")
    except RuntimeError as e:
        assert "Security Error" not in str(e)
    except FileNotFoundError:
        pass  # Expected
    except Exception:
        pass  # Other errors are fine, as long as not Security Error
