"""
Tests for the dependencies module.

Tests DependencyInfo, OptionalDependencies, and utility functions.
"""

import warnings

import pytest

from ml4t.engineer.utils.dependencies import (
    DEPS,
    DependencyInfo,
    OptionalDependencies,
    check_dependency,
    get_dependency_summary,
    require_dependency,
    warn_if_missing,
)


class TestDependencyInfo:
    """Tests for DependencyInfo dataclass."""

    def test_basic_creation(self):
        """Test basic DependencyInfo creation."""
        dep = DependencyInfo(
            name="TestPackage",
            import_name="test_package",
            install_cmd="pip install test-package",
            purpose="Testing",
            features=["feature1", "feature2"],
        )

        assert dep.name == "TestPackage"
        assert dep.import_name == "test_package"
        assert dep.install_cmd == "pip install test-package"
        assert dep.purpose == "Testing"
        assert dep.features == ["feature1", "feature2"]
        assert dep.alternatives == []

    def test_with_alternatives(self):
        """Test DependencyInfo with alternatives."""
        dep = DependencyInfo(
            name="TestPackage",
            import_name="test_package",
            install_cmd="pip install test-package",
            purpose="Testing",
            features=["feature1"],
            alternatives=["alt1", "alt2"],
        )

        assert dep.alternatives == ["alt1", "alt2"]

    def test_post_init_none_alternatives(self):
        """Test __post_init__ handles None alternatives."""
        dep = DependencyInfo(
            name="Test",
            import_name="test",
            install_cmd="pip install test",
            purpose="Testing",
            features=[],
            alternatives=None,
        )

        # __post_init__ should convert None to []
        assert dep.alternatives == []

    def test_is_available_installed_package(self):
        """Test is_available for installed package."""
        # numpy is always installed
        dep = DependencyInfo(
            name="NumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Numerical computing",
            features=[],
        )

        assert dep.is_available is True

    def test_is_available_missing_package(self):
        """Test is_available for missing package."""
        dep = DependencyInfo(
            name="NonExistent",
            import_name="nonexistent_package_xyz_123",
            install_cmd="pip install nonexistent",
            purpose="Does not exist",
            features=[],
        )

        assert dep.is_available is False

    def test_require_installed_package(self):
        """Test require for installed package (should not raise)."""
        dep = DependencyInfo(
            name="NumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Numerical computing",
            features=[],
        )

        # Should not raise
        dep.require("test feature")

    def test_require_missing_package(self):
        """Test require for missing package raises ImportError."""
        dep = DependencyInfo(
            name="NonExistent",
            import_name="nonexistent_package_xyz_123",
            install_cmd="pip install nonexistent",
            purpose="Does not exist",
            features=[],
        )

        with pytest.raises(ImportError, match="NonExistent is required"):
            dep.require()

    def test_require_missing_with_feature(self):
        """Test require error message includes feature name."""
        dep = DependencyInfo(
            name="NonExistent",
            import_name="nonexistent_package_xyz_123",
            install_cmd="pip install nonexistent",
            purpose="Does not exist",
            features=[],
        )

        with pytest.raises(ImportError, match="for test feature"):
            dep.require("test feature")

    def test_require_missing_with_alternatives(self):
        """Test require error message includes alternatives."""
        dep = DependencyInfo(
            name="NonExistent",
            import_name="nonexistent_package_xyz_123",
            install_cmd="pip install nonexistent",
            purpose="Does not exist",
            features=[],
            alternatives=["alt1", "alt2"],
        )

        with pytest.raises(ImportError, match="Alternatives"):
            dep.require()

    def test_warn_if_missing_installed(self):
        """Test warn_if_missing returns True for installed package."""
        dep = DependencyInfo(
            name="NumPy",
            import_name="numpy",
            install_cmd="pip install numpy",
            purpose="Numerical computing",
            features=[],
        )

        result = dep.warn_if_missing("test feature")
        assert result is True

    def test_warn_if_missing_missing(self):
        """Test warn_if_missing returns False and warns for missing package."""
        dep = DependencyInfo(
            name="NonExistent",
            import_name="nonexistent_package_xyz_123",
            install_cmd="pip install nonexistent",
            purpose="Does not exist",
            features=[],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = dep.warn_if_missing("test feature", "using fallback")

            assert result is False
            assert len(w) == 1
            assert "NonExistent not available" in str(w[0].message)
            assert "using fallback" in str(w[0].message)

    def test_warn_if_missing_with_alternatives(self):
        """Test warn_if_missing includes alternatives in warning."""
        dep = DependencyInfo(
            name="NonExistent",
            import_name="nonexistent_package_xyz_123",
            install_cmd="pip install nonexistent",
            purpose="Does not exist",
            features=[],
            alternatives=["alt1", "alt2"],
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dep.warn_if_missing()

            assert len(w) == 1
            assert "or use:" in str(w[0].message)


class TestOptionalDependencies:
    """Tests for OptionalDependencies class."""

    def test_init_registers_dependencies(self):
        """Test that init registers known dependencies."""
        deps = OptionalDependencies()

        # Check known dependencies are registered
        assert "lightgbm" in deps._deps
        assert "xgboost" in deps._deps
        assert "shap" in deps._deps
        assert "plotly" in deps._deps

    def test_getattr_valid(self):
        """Test __getattr__ for valid dependency."""
        deps = OptionalDependencies()

        info = deps.lightgbm
        assert info.name == "LightGBM"
        assert info.import_name == "lightgbm"

    def test_getattr_invalid(self):
        """Test __getattr__ for invalid dependency raises AttributeError."""
        deps = OptionalDependencies()

        with pytest.raises(AttributeError, match="Unknown dependency"):
            _ = deps.nonexistent_dep

    def test_getitem_valid(self):
        """Test __getitem__ for valid dependency."""
        deps = OptionalDependencies()

        info = deps["lightgbm"]
        assert info.name == "LightGBM"

    def test_getitem_invalid(self):
        """Test __getitem__ for invalid dependency raises KeyError."""
        deps = OptionalDependencies()

        with pytest.raises(KeyError):
            _ = deps["nonexistent_dep"]

    def test_get_valid(self):
        """Test get method for valid dependency."""
        deps = OptionalDependencies()

        info = deps.get("lightgbm")
        assert info is not None
        assert info.name == "LightGBM"

    def test_get_invalid(self):
        """Test get method returns default for invalid dependency."""
        deps = OptionalDependencies()

        info = deps.get("nonexistent", "default_value")
        assert info == "default_value"

    def test_get_invalid_none(self):
        """Test get method returns None by default for invalid dependency."""
        deps = OptionalDependencies()

        info = deps.get("nonexistent")
        assert info is None

    def test_check_valid(self):
        """Test check for known dependency."""
        deps = OptionalDependencies()

        # This depends on whether lightgbm is installed
        result = deps.check("lightgbm")
        assert isinstance(result, bool)

    def test_check_unknown(self):
        """Test check for unknown dependency returns False."""
        deps = OptionalDependencies()

        result = deps.check("nonexistent_dep")
        assert result is False

    def test_check_multiple(self):
        """Test check_multiple method."""
        deps = OptionalDependencies()

        results = deps.check_multiple(["lightgbm", "xgboost", "numpy_fake"])
        assert isinstance(results, dict)
        assert "lightgbm" in results
        assert "xgboost" in results
        assert "numpy_fake" in results
        assert results["numpy_fake"] is False

    def test_get_missing(self):
        """Test get_missing method."""
        deps = OptionalDependencies()

        # numpy_fake should always be missing
        missing = deps.get_missing(["lightgbm", "numpy_fake"])
        assert "numpy_fake" in missing

    def test_warn_missing(self):
        """Test warn_missing method."""
        deps = OptionalDependencies()

        # Create a new deps instance with a fake dependency we know is missing
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # This will warn about any missing deps
            missing = deps.warn_missing(["lightgbm"], "test feature")

            # Check that missing list is returned
            assert isinstance(missing, list)

    def test_summary(self):
        """Test summary method."""
        deps = OptionalDependencies()

        summary = deps.summary()
        assert "Optional Dependencies Status:" in summary
        assert "LightGBM" in summary
        assert "XGBoost" in summary


class TestGlobalDEPS:
    """Tests for the global DEPS instance."""

    def test_deps_is_optional_dependencies(self):
        """Test DEPS is an OptionalDependencies instance."""
        assert isinstance(DEPS, OptionalDependencies)

    def test_deps_has_registered_dependencies(self):
        """Test DEPS has registered dependencies."""
        assert "lightgbm" in DEPS._deps
        assert "xgboost" in DEPS._deps


class TestCheckDependency:
    """Tests for check_dependency function."""

    def test_check_unknown(self):
        """Test check_dependency for unknown dependency."""
        result = check_dependency("completely_fake_package")
        assert result is False

    def test_check_known(self):
        """Test check_dependency for known dependency."""
        result = check_dependency("lightgbm")
        assert isinstance(result, bool)


class TestRequireDependency:
    """Tests for require_dependency function."""

    def test_require_unknown_raises(self):
        """Test require_dependency for unknown dependency raises ImportError."""
        with pytest.raises(ImportError, match="Unknown dependency"):
            require_dependency("completely_fake_package")

    def test_require_known_missing(self):
        """Test require_dependency for known but missing dependency."""
        # This will either pass (if installed) or raise (if not installed)
        # We just test that it doesn't raise "Unknown dependency"
        try:
            require_dependency("lightgbm", "test feature")
        except ImportError as e:
            # If it raises, it should mention LightGBM, not "Unknown"
            assert "Unknown" not in str(e)


class TestWarnIfMissing:
    """Tests for warn_if_missing function."""

    def test_warn_unknown(self):
        """Test warn_if_missing for unknown dependency."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = warn_if_missing("completely_fake_package")

            assert result is False
            assert len(w) == 1
            assert "Unknown dependency" in str(w[0].message)

    def test_warn_known_installed(self):
        """Test warn_if_missing for installed package."""
        # polars should be installed (it's a core dependency)
        # Since polars isn't in DEPS, let's check a known registered one
        result = warn_if_missing("lightgbm", "test feature", "skipping")
        # Result depends on whether lightgbm is installed
        assert isinstance(result, bool)


class TestGetDependencySummary:
    """Tests for get_dependency_summary function."""

    def test_returns_string(self):
        """Test get_dependency_summary returns a string."""
        summary = get_dependency_summary()
        assert isinstance(summary, str)

    def test_contains_header(self):
        """Test get_dependency_summary contains header."""
        summary = get_dependency_summary()
        assert "Optional Dependencies Status:" in summary

    def test_contains_dependencies(self):
        """Test get_dependency_summary lists dependencies."""
        summary = get_dependency_summary()
        assert "LightGBM" in summary
        assert "XGBoost" in summary
        assert "SHAP" in summary
        assert "Plotly" in summary
