"""
Column Mapper Utility - Intelligent Solar Energy Column Detection & Standardization

Consolidates 150+ lines of duplicate column detection code across notebooks into a
robust, configurable utility with comprehensive error handling and validation.

Key Features:
- Intelligent pattern-based column detection
- Automatic standardization to consistent naming
- Confidence scoring and validation
- Multi-language support and custom patterns
- Comprehensive logging and error recovery
- Performance optimization with caching

Usage:
    # Basic usage - auto-detect and standardize
    mapper = ColumnMapper()
    standardized_df = mapper.standardize_columns(df)

    # Get column mappings for inspection
    mappings = mapper.detect_columns(df)

    # Validate data has required columns
    validation = mapper.validate_energy_data(df)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, FrozenSet
from dataclasses import dataclass, field
from functools import lru_cache
import re
from difflib import SequenceMatcher


@dataclass
class DetectionResult:
    """Results of column detection with comprehensive metadata."""
    mapping: Dict[str, str]
    confidence_scores: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    ambiguous_matches: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def overall_confidence(self) -> float:
        """Calculate overall detection confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection is high confidence (>0.8)."""
        return self.overall_confidence > 0.8


class ColumnMapperError(Exception):
    """Base exception for ColumnMapper errors."""
    pass


class AmbiguousColumnsError(ColumnMapperError):
    """Raised when multiple columns match the same energy pattern."""

    def __init__(self, energy_type: str, candidates: List[str]):
        self.energy_type = energy_type
        self.candidates = candidates
        super().__init__(
            f"Multiple {energy_type} columns found: {candidates}. "
            f"Use strict_mode=False for automatic resolution or specify custom patterns."
        )


class MissingColumnsError(ColumnMapperError):
    """Raised when required columns are missing in strict mode."""

    def __init__(self, missing_columns: List[str]):
        self.missing_columns = missing_columns
        super().__init__(
            f"Required energy columns not found: {missing_columns}. "
            f"Use strict_mode=False to continue with available columns."
        )


class ColumnMapperConfig:
    """Configuration for ColumnMapper behavior and detection patterns."""

    def __init__(self):
        # Detection patterns for each energy type
        self.detection_patterns = {
            'production': ['production', 'produced', 'generation', 'generated', 'pv', 'solar'],
            'consumption': ['consumption', 'consumed', 'usage', 'used', 'load', 'demand'],
            'export': ['export', 'exported', 'sold', 'grid_out', 'feed_in', 'injection'],
            'import': ['import', 'imported', 'purchased', 'grid_in', 'bought', 'consumption_grid'],
            'datetime': ['date', 'time', 'timestamp', 'datetime', 'period']
        }

        # International pattern support
        self.international_patterns = {
            'production': {
                'es': ['producción', 'generación', 'solar'],
                'fr': ['production', 'génération', 'solaire'],
                'de': ['produktion', 'erzeugung', 'solar'],
                'it': ['produzione', 'generazione', 'solare']
            },
            'consumption': {
                'es': ['consumo', 'uso'],
                'fr': ['consommation', 'usage'],
                'de': ['verbrauch', 'nutzung'],
                'it': ['consumo', 'uso']
            }
        }

        # Standard target column names
        self.standard_columns = {
            'production': 'Production (kWh)',
            'consumption': 'Consumption (kWh)',
            'export': 'Export (kWh)',
            'import': 'Import (kWh)',
            'datetime': 'Date/Time'
        }

        # Detection configuration
        self.priority_order = ['production', 'consumption', 'export', 'import', 'datetime']
        self.confidence_threshold = 0.7
        self.fuzzy_match_threshold = 0.8
        self.case_sensitive = False

        # Custom patterns (user-extensible)
        self.custom_patterns = {}

    def add_custom_pattern(self, energy_type: str, patterns: List[str], language: str = 'en'):
        """Add custom detection patterns for specific energy types."""
        if energy_type not in self.custom_patterns:
            self.custom_patterns[energy_type] = {}
        if language not in self.custom_patterns[energy_type]:
            self.custom_patterns[energy_type][language] = []
        self.custom_patterns[energy_type][language].extend(patterns)

    def get_all_patterns(self, energy_type: str) -> List[str]:
        """Get all patterns for an energy type including international and custom."""
        patterns = self.detection_patterns.get(energy_type, []).copy()

        # Add international patterns
        if energy_type in self.international_patterns:
            for lang_patterns in self.international_patterns[energy_type].values():
                patterns.extend(lang_patterns)

        # Add custom patterns
        if energy_type in self.custom_patterns:
            for lang_patterns in self.custom_patterns[energy_type].values():
                patterns.extend(lang_patterns)

        return list(set(patterns))  # Remove duplicates


class ColumnMapper:
    """
    Intelligent column detection and standardization for solar energy data.

    Automatically detects and standardizes column names across different solar energy
    data formats, eliminating the need for manual column detection in notebooks.
    """

    def __init__(self,
                 strict_mode: bool = False,
                 log_level: str = 'INFO',
                 config: Optional[ColumnMapperConfig] = None,
                 compatibility_mode: str = 'standard'):
        """
        Initialize ColumnMapper with configuration.

        Args:
            strict_mode: If True, raises errors for missing/ambiguous columns
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            config: Custom configuration (uses default if None)
            compatibility_mode: 'strict', 'standard', 'permissive', or 'legacy'
        """
        self.strict_mode = strict_mode
        self.compatibility_mode = compatibility_mode
        self.config = config or ColumnMapperConfig()

        # Setup logging with notebook-friendly defaults
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Only add handler if none exists and we're not in a notebook environment
        if not self.logger.handlers and log_level.upper() != 'INFO':
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Cache for performance optimization
        self._detection_cache = {}
        self._pattern_cache = {}

    @lru_cache(maxsize=128)
    def _get_cached_patterns(self, energy_type: str) -> Tuple[str, ...]:
        """Get cached patterns for energy type (frozen for hashing)."""
        return tuple(self.config.get_all_patterns(energy_type))

    def _calculate_match_confidence(self, column_name: str, patterns: List[str]) -> float:
        """Calculate confidence score for pattern matching."""
        column_lower = column_name.lower()

        # Exact pattern match
        for pattern in patterns:
            if pattern.lower() in column_lower:
                # Higher confidence for exact word matches
                if f" {pattern.lower()} " in f" {column_lower} " or \
                   column_lower.startswith(pattern.lower()) or \
                   column_lower.endswith(pattern.lower()):
                    return 1.0
                return 0.9

        # Fuzzy matching for partial matches
        best_ratio = 0.0
        for pattern in patterns:
            ratio = SequenceMatcher(None, column_lower, pattern.lower()).ratio()
            best_ratio = max(best_ratio, ratio)

        return best_ratio if best_ratio >= self.config.fuzzy_match_threshold else 0.0

    def _detect_energy_type(self, column_name: str) -> Tuple[Optional[str], float]:
        """Detect energy type for a single column with confidence score."""
        best_match = None
        best_confidence = 0.0

        for energy_type in self.config.priority_order:
            patterns = self._get_cached_patterns(energy_type)
            confidence = self._calculate_match_confidence(column_name, list(patterns))

            if confidence > best_confidence and confidence >= self.config.confidence_threshold:
                best_match = energy_type
                best_confidence = confidence

        return best_match, best_confidence

    def _resolve_ambiguous_matches(self,
                                 ambiguous: Dict[str, List[str]]) -> Dict[str, str]:
        """Resolve ambiguous column matches using heuristics."""
        resolved = {}

        for energy_type, candidates in ambiguous.items():
            if len(candidates) == 1:
                resolved[energy_type] = candidates[0]
                continue

            # Heuristic 1: Prefer columns with units in parentheses
            unit_columns = [col for col in candidates if '(' in col and ')' in col]
            if unit_columns:
                resolved[energy_type] = unit_columns[0]
                continue

            # Heuristic 2: Prefer shorter column names (less likely to be composite)
            shortest = min(candidates, key=len)
            resolved[energy_type] = shortest

            self.logger.warning(
                f"Ambiguous {energy_type} columns resolved by choosing shortest: "
                f"{shortest} from {candidates}"
            )

        return resolved

    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Detect energy columns in DataFrame.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict mapping energy type to detected column name

        Raises:
            AmbiguousColumnsError: In strict mode with ambiguous matches
            MissingColumnsError: In strict mode with missing required columns
        """
        # Cache key for performance
        cache_key = frozenset(df.columns)
        if cache_key in self._detection_cache:
            self.logger.debug("Using cached column detection results")
            return self._detection_cache[cache_key].copy()

        self.logger.info(f"Detecting columns in DataFrame with {len(df.columns)} columns")
        self.logger.debug(f"Available columns: {list(df.columns)}")

        # Track matches for each energy type
        matches = {energy_type: [] for energy_type in self.config.priority_order}
        confidence_scores = {}

        # Detect each column
        for column in df.columns:
            energy_type, confidence = self._detect_energy_type(str(column))
            if energy_type:
                matches[energy_type].append(column)
                confidence_scores[column] = confidence
                self.logger.debug(
                    f"Column '{column}' matched {energy_type} with confidence {confidence:.2f}"
                )

        # Resolve matches
        result = {}
        ambiguous = {}

        for energy_type, candidates in matches.items():
            if not candidates:
                continue
            elif len(candidates) == 1:
                result[energy_type] = candidates[0]
            else:
                # Multiple matches - handle based on mode
                ambiguous[energy_type] = candidates
                if self.strict_mode:
                    raise AmbiguousColumnsError(energy_type, candidates)

        # Resolve ambiguous matches in non-strict mode
        if ambiguous and not self.strict_mode:
            resolved = self._resolve_ambiguous_matches(ambiguous)
            result.update(resolved)

        # Check for required columns in strict mode
        if self.strict_mode:
            required = ['production']  # Minimum required
            missing = [req for req in required if req not in result]
            if missing:
                raise MissingColumnsError(missing)

        # Cache results
        self._detection_cache[cache_key] = result.copy()

        self.logger.info(f"Detection complete: {len(result)} energy columns found")
        return result

    def detect_columns_with_confidence(self, df: pd.DataFrame) -> DetectionResult:
        """
        Enhanced detection with comprehensive results and confidence metrics.

        Args:
            df: DataFrame to analyze

        Returns:
            DetectionResult with mapping, confidence scores, and metadata
        """
        try:
            mapping = self.detect_columns(df)

            # Calculate confidence scores
            confidence_scores = {}
            for energy_type, column in mapping.items():
                patterns = self._get_cached_patterns(energy_type)
                confidence_scores[energy_type] = self._calculate_match_confidence(
                    column, list(patterns)
                )

            # Generate suggestions and warnings
            warnings = []
            suggestions = []

            # Check for low confidence matches
            low_confidence = {k: v for k, v in confidence_scores.items() if v < 0.8}
            if low_confidence:
                warnings.append(f"Low confidence matches: {low_confidence}")
                suggestions.append(
                    "Consider adding custom patterns for better detection accuracy"
                )

            # Check for missing common columns
            missing_common = set(['production', 'consumption']) - set(mapping.keys())
            if missing_common:
                warnings.append(f"Common energy columns not detected: {missing_common}")
                suggestions.append(
                    "Verify column names match expected patterns or add custom patterns"
                )

            return DetectionResult(
                mapping=mapping,
                confidence_scores=confidence_scores,
                warnings=warnings,
                suggestions=suggestions
            )

        except (AmbiguousColumnsError, MissingColumnsError) as e:
            # Return partial results with error information
            return DetectionResult(
                mapping={},
                confidence_scores={},
                warnings=[str(e)],
                suggestions=[
                    "Use strict_mode=False for automatic resolution",
                    "Add custom patterns for better detection"
                ]
            )

    def standardize_columns(self, df: pd.DataFrame,
                          preserve_non_energy: bool = True) -> pd.DataFrame:
        """
        Rename columns to standard naming convention.

        Args:
            df: DataFrame to standardize
            preserve_non_energy: Keep non-energy columns unchanged

        Returns:
            DataFrame with standardized column names
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for standardization")
            return df.copy()

        # Detect energy columns
        detected = self.detect_columns(df)

        if not detected:
            self.logger.warning("No energy columns detected for standardization")
            return df.copy()

        # Create rename mapping
        rename_mapping = {}
        for energy_type, current_name in detected.items():
            standard_name = self.config.standard_columns.get(energy_type)
            if standard_name:
                rename_mapping[current_name] = standard_name

        # Apply renaming
        result_df = df.rename(columns=rename_mapping)

        self.logger.info(
            f"Standardized {len(rename_mapping)} columns: "
            f"{list(rename_mapping.keys())} -> {list(rename_mapping.values())}"
        )

        return result_df

    def get_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get mapping from current column names to standard names.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict mapping current column names to standard names
        """
        detected = self.detect_columns(df)

        mapping = {}
        for energy_type, current_name in detected.items():
            standard_name = self.config.standard_columns.get(energy_type)
            if standard_name:
                mapping[current_name] = standard_name

        return mapping

    def validate_energy_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that DataFrame contains expected energy columns.

        Args:
            df: DataFrame to validate

        Returns:
            Validation results with status, missing columns, and recommendations
        """
        detected = self.detect_columns(df)

        # Define expected columns by importance
        critical = ['production']
        important = ['consumption', 'export', 'import']
        optional = ['datetime']

        # Check what's missing
        missing_critical = [col for col in critical if col not in detected]
        missing_important = [col for col in important if col not in detected]
        missing_optional = [col for col in optional if col not in detected]

        # Determine overall status
        if missing_critical:
            status = 'INVALID'
        elif missing_important:
            status = 'WARNING'
        else:
            status = 'VALID'

        # Generate recommendations
        recommendations = []
        if missing_critical:
            recommendations.append(
                f"Critical columns missing: {missing_critical}. "
                "Cannot perform energy analysis without production data."
            )
        if missing_important:
            recommendations.append(
                f"Important columns missing: {missing_important}. "
                "Some analysis features may be limited."
            )
        if missing_optional:
            recommendations.append(
                f"Optional columns missing: {missing_optional}. "
                "Time-based analysis may be affected."
            )

        if not recommendations:
            recommendations.append("All expected energy columns detected successfully.")

        return {
            'status': status,
            'detected_columns': detected,
            'missing_critical': missing_critical,
            'missing_important': missing_important,
            'missing_optional': missing_optional,
            'recommendations': recommendations,
            'column_count': len(detected),
            'total_columns': len(df.columns)
        }

    def get_energy_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get all detected energy columns with their types.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict mapping energy type to column name
        """
        return self.detect_columns(df)

    @staticmethod
    def suggest_column_names(columns: List[str]) -> Dict[str, List[str]]:
        """
        Suggest likely energy column matches for manual review.

        Args:
            columns: List of column names to analyze

        Returns:
            Dict with suggestions for each energy type
        """
        # Create temporary mapper for pattern matching
        temp_mapper = ColumnMapper(strict_mode=False)

        suggestions = {
            energy_type: [] for energy_type in temp_mapper.config.priority_order
        }

        for column in columns:
            energy_type, confidence = temp_mapper._detect_energy_type(column)
            if energy_type and confidence >= 0.5:  # Lower threshold for suggestions
                suggestions[energy_type].append((column, confidence))

        # Sort by confidence and return top suggestions
        result = {}
        for energy_type, candidates in suggestions.items():
            if candidates:
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                result[energy_type] = [col for col, conf in sorted_candidates[:3]]

        return result

    def print_detection_summary(self, df: pd.DataFrame) -> None:
        """Print a formatted summary of column detection results."""
        detection_result = self.detect_columns_with_confidence(df)

        print("\n📊 Column Detection Summary")
        print("=" * 50)

        if detection_result.mapping:
            print(f"\n✅ Detected Energy Columns ({len(detection_result.mapping)}):")
            for energy_type, column in detection_result.mapping.items():
                confidence = detection_result.confidence_scores.get(energy_type, 0.0)
                standard_name = self.config.standard_columns.get(energy_type, "Unknown")
                print(f"  • {energy_type.title()}: '{column}' → '{standard_name}' "
                      f"(confidence: {confidence:.2f})")
        else:
            print("\n❌ No energy columns detected")

        if detection_result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in detection_result.warnings:
                print(f"  • {warning}")

        if detection_result.suggestions:
            print(f"\n💡 Suggestions:")
            for suggestion in detection_result.suggestions:
                print(f"  • {suggestion}")

        print(f"\n📈 Overall Confidence: {detection_result.overall_confidence:.2f}")
        print(f"🎯 Status: {'High Confidence' if detection_result.is_high_confidence else 'Needs Review'}")


# Convenience functions for backward compatibility and easy integration
def detect_energy_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Quick column detection function for simple use cases."""
    mapper = ColumnMapper()
    return mapper.detect_columns(df)


def standardize_energy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Quick standardization function for simple use cases."""
    mapper = ColumnMapper()
    return mapper.standardize_columns(df)


def validate_solar_data_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick validation function for simple use cases."""
    mapper = ColumnMapper()
    return mapper.validate_energy_data(df)