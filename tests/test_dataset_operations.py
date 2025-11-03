import pytest
import json
import os
import tempfile
from pathlib import Path
from src.dataset_operations import get_data, save_summaries, create_metadata_file


class TestGetData:
    """Test suite for get_data function."""

    def test_get_data_success(self, tmp_path):
        """Test successful loading of JSON files."""
        # Create test JSON files
        test_data_1 = {"summary": {"text": ["Test data 1"]}}
        test_data_2 = {"summary": {"text": ["Test data 2"]}}

        file1 = tmp_path / "1e.json"
        file2 = tmp_path / "2e.json"
        file1.write_text(json.dumps(test_data_1))
        file2.write_text(json.dumps(test_data_2))

        # Call function
        result = get_data(str(tmp_path), "*e.json")

        # Assertions
        assert len(result) == 2
        assert result[0]["data"] == test_data_1
        assert result[1]["data"] == test_data_2
        assert result[0]["file_path"] == str(file1)
        assert result[1]["file_path"] == str(file2)

    def test_get_data_pattern_filtering(self, tmp_path):
        """Test that only files matching pattern are loaded."""
        # Create files with different extensions
        (tmp_path / "1e.json").write_text(json.dumps({"data": "match"}))
        (tmp_path / "2.json").write_text(json.dumps({"data": "no match"}))
        (tmp_path / "3e.json").write_text(json.dumps({"data": "match"}))

        result = get_data(str(tmp_path), "*e.json")

        assert len(result) == 2
        assert all("e.json" in item["file_path"] for item in result)

    def test_get_data_empty_directory(self, tmp_path):
        """Test behavior with no matching files."""
        result = get_data(str(tmp_path), "*e.json")

        assert result == []

    def test_get_data_nonexistent_directory(self):
        """Test error handling for nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            get_data("/nonexistent/path", "*.json")

    def test_get_data_invalid_json(self, tmp_path, caplog):
        """Test handling of invalid JSON files."""
        # Create valid and invalid JSON files
        valid_file = tmp_path / "1e.json"
        invalid_file = tmp_path / "2e.json"

        valid_file.write_text(json.dumps({"valid": "data"}))
        invalid_file.write_text("{ invalid json }")

        result = get_data(str(tmp_path), "*e.json")

        # Should only load valid file
        assert len(result) == 1
        assert result[0]["data"] == {"valid": "data"}

        # Should log error for invalid file
        assert "Error decoding" in caplog.text

    def test_get_data_sorted_order(self, tmp_path):
        """Test that files are returned in sorted order."""
        # Create files in non-alphabetical order
        (tmp_path / "3e.json").write_text(json.dumps({"order": 3}))
        (tmp_path / "1e.json").write_text(json.dumps({"order": 1}))
        (tmp_path / "2e.json").write_text(json.dumps({"order": 2}))

        result = get_data(str(tmp_path), "*e.json")

        # Verify sorted order
        assert result[0]["data"]["order"] == 1
        assert result[1]["data"]["order"] == 2
        assert result[2]["data"]["order"] == 3


class TestSaveSummaries:
    """Test suite for save_summaries function."""

    def test_save_summaries_basic(self, tmp_path):
        """Test basic summary saving functionality."""
        summaries = [
            {"summary": {"text": ["Test 1"]}},
            {"summary": {"text": ["Test 2"]}}
        ]

        save_summaries(summaries, str(tmp_path), "e.json")

        # Verify files were created
        assert (tmp_path / "1e.json").exists()
        assert (tmp_path / "2e.json").exists()

        # Verify content
        with open(tmp_path / "1e.json") as f:
            data = json.load(f)
            assert "call_id" in data
            assert data["summary"]["text"] == ["Test 1"]

    def test_save_summaries_auto_increment(self, tmp_path):
        """Test that numbering continues from existing files."""
        # Create existing files
        (tmp_path / "1e.json").write_text(json.dumps({"existing": 1}))
        (tmp_path / "2e.json").write_text(json.dumps({"existing": 2}))

        summaries = [{"summary": {"text": ["New"]}}]
        save_summaries(summaries, str(tmp_path), "e.json")

        # Should create file starting at 3
        assert (tmp_path / "3e.json").exists()
        assert not (tmp_path / "4e.json").exists()

    def test_save_summaries_call_id_format(self, tmp_path):
        """Test that call_id is generated correctly."""
        summaries = [{"summary": {"text": ["Test"]}}]
        save_summaries(summaries, str(tmp_path), "e.json")

        with open(tmp_path / "1e.json") as f:
            data = json.load(f)
            assert "call_id" in data
            assert data["call_id"].startswith("1-record-")
            assert data["call_id"].endswith("_ms")

    def test_save_summaries_call_id_first_key(self, tmp_path):
        """Test that call_id is the first key in saved JSON."""
        summaries = [{"summary": {"text": ["Test"]}, "other": "data"}]
        save_summaries(summaries, str(tmp_path), "e.json")

        # Read raw JSON to check key order
        with open(tmp_path / "1e.json") as f:
            content = f.read()
            data = json.loads(content)
            first_key = list(data.keys())[0]
            assert first_key == "call_id"

    def test_save_summaries_creates_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_dir" / "nested"
        summaries = [{"summary": {"text": ["Test"]}}]

        save_summaries(summaries, str(output_dir), "e.json")

        assert output_dir.exists()
        assert (output_dir / "1e.json").exists()

    def test_save_summaries_custom_suffix(self, tmp_path):
        """Test saving with custom suffix."""
        summaries = [{"data": "test"}]
        save_summaries(summaries, str(tmp_path), "custom.json")

        assert (tmp_path / "1custom.json").exists()

    def test_save_summaries_preserves_existing_call_id(self, tmp_path):
        """Test that existing call_id in summary is handled correctly."""
        summaries = [{"call_id": "old-id", "summary": {"text": ["Test"]}}]
        save_summaries(summaries, str(tmp_path), "e.json")

        with open(tmp_path / "1e.json") as f:
            data = json.load(f)
            # call_id should be overwritten with new format
            assert data["call_id"] != "old-id"
            assert data["call_id"].startswith("1-record-")

    def test_save_summaries_encoding(self, tmp_path):
        """Test that non-ASCII characters are handled correctly."""
        summaries = [{"summary": {"text": ["Тест с кириллицей и émojis 🎉"]}}]
        save_summaries(summaries, str(tmp_path), "e.json")

        with open(tmp_path / "1e.json", encoding="utf-8") as f:
            data = json.load(f)
            assert "кириллицей" in data["summary"]["text"][0]
            assert "🎉" in data["summary"]["text"][0]


class TestCreateMetadataFile:
    """Test suite for create_metadata_file function."""

    def test_create_metadata_file_basic(self, tmp_path):
        """Test basic metadata file creation."""
        # Create a simple config class instead of Mock to avoid recursion
        class MockConfig:
            SETTING_1 = "value1"
            SETTING_2 = 42
            SETTING_3 = ["list", "of", "values"]

        filepath = tmp_path / "metadata.json"
        create_metadata_file(MockConfig, str(filepath))

        # Verify file was created
        assert filepath.exists()

        # Verify content
        with open(filepath) as f:
            data = json.load(f)
            assert data["SETTING_1"] == "value1"
            assert data["SETTING_2"] == 42
            assert data["SETTING_3"] == ["list", "of", "values"]

    def test_create_metadata_file_excludes_private_attrs(self, tmp_path):
        """Test that private attributes are excluded."""
        class MockConfig:
            PUBLIC_VAR = "visible"
            __private_var = "hidden"
            __dunder__ = "hidden"

        filepath = tmp_path / "metadata.json"
        create_metadata_file(MockConfig, str(filepath))

        with open(filepath) as f:
            data = json.load(f)
            assert "PUBLIC_VAR" in data
            assert "__private_var" not in data
            assert "__dunder__" not in data

    def test_create_metadata_file_excludes_callables(self, tmp_path):
        """Test that callable attributes (functions/methods) are excluded."""
        class MockConfig:
            VARIABLE = "value"
            @staticmethod
            def function():
                return "result"

        filepath = tmp_path / "metadata.json"
        create_metadata_file(MockConfig, str(filepath))

        with open(filepath) as f:
            data = json.load(f)
            assert "VARIABLE" in data
            assert "function" not in data

    def test_create_metadata_file_creates_directory(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        class MockConfig:
            VAR = "value"

        filepath = tmp_path / "nested" / "dirs" / "metadata.json"
        create_metadata_file(MockConfig, str(filepath))

        assert filepath.exists()
        assert filepath.parent.exists()

    def test_create_metadata_file_encoding(self, tmp_path):
        """Test that non-ASCII characters in config are handled correctly."""
        class MockConfig:
            TEXT = "Тест с кириллицей"

        filepath = tmp_path / "metadata.json"
        create_metadata_file(MockConfig, str(filepath))

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            assert data["TEXT"] == "Тест с кириллицей"
