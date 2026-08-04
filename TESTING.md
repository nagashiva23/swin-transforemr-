# Testing Guide

This document describes the testing infrastructure for the Swin Transformer captioning model.

## Setup

Install test dependencies:
```bash
pip install -r requirements.txt
```

This includes:
- **pytest**: Testing framework
- **pytest-cov**: Code coverage reporting
- **pytest-xdist**: Parallel test execution

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`.

### Run specific test file
```bash
pytest tests/test_data.py -v
```

### Run specific test
```bash
pytest tests/test_data.py::TestVocabulary::test_vocabulary_encode -v
```

### Run tests in parallel (faster)
```bash
pytest -n auto
```

### Run with markers
```bash
pytest -m unit              # Run only unit tests
pytest -m integration       # Run only integration tests
pytest -m "not slow"        # Skip slow tests
```

## Test Structure

### `tests/conftest.py`
Shared pytest fixtures and configuration:
- `temp_dir`: Temporary directory for test artifacts
- `sample_vocabulary`: A small test vocabulary
- `test_dataset_dir`: A minimal dataset with images and captions
- `swin_captioner`: A small model instance
- `sample_batch`: Sample batch of images and captions
- `device`: Appropriate device (MPS/CUDA/CPU)

### `tests/test_data.py`
Tests for data loading and preprocessing:
- **TestTokenize**: Tokenization function tests
- **TestVocabulary**: Vocabulary building, encoding, and persistence
- **TestReadCaptions**: CSV caption reading
- **TestRoCoCaptionDataset**: Dataset loading, normalization, and padding

### `tests/test_swin_transformer.py`
Tests for model architecture components:
- **TestConvNormAct**: Convolutional blocks
- **TestPatchEmbedding**: Patch embedding layer
- **TestSwinBlock**: Transformer blocks
- **TestSwinStem**: Encoder stems
- **TestMultiScaleGatedSwinEncoder**: Multi-scale encoder
- **TestVisualEvidenceDecoder**: Decoder with attention
- **TestSwinCaptioner**: Full model forward pass, gradients, and different batch sizes

### `tests/test_losses.py`
Tests for loss functions:
- **TestBiomedicalTermAwareFocalLoss**: Focal loss computation, gradients, and special cases

### `tests/test_train.py`
Integration tests for the training pipeline:
- Single training and validation steps
- Gradient accumulation
- Checkpoint save/load
- Mixed precision compatibility (CUDA)

## Writing New Tests

### Test template
```python
import pytest
import torch
from src.module import Component

class TestComponent:
    """Test the Component class."""
    
    def test_basic_functionality(self):
        """Test that component works as expected."""
        component = Component()
        result = component.process(input_data)
        assert result is not None
    
    def test_with_fixture(self, sample_vocabulary, device):
        """Test using fixtures."""
        # Use fixtures provided by conftest.py
        pass
```

### Fixture usage
```python
def test_with_dataset(self, test_dataset_dir, sample_vocabulary):
    """Test using dataset fixture."""
    dataset = RoCoCaptionDataset(
        test_dataset_dir,
        "train",
        sample_vocabulary,
        image_size=224,
    )
    assert len(dataset) > 0
```

### Testing with device
```python
def test_on_device(self, swin_captioner, device):
    """Test model on appropriate device."""
    model = swin_captioner.to(device)
    images = torch.randn(2, 3, 224, 224, device=device)
    output = model(images)
```

## Common Issues

### Slow tests
Some tests may be slow due to model inference. Use:
```bash
pytest -m "not slow"
```

Or mark slow tests:
```python
@pytest.mark.slow
def test_large_model():
    pass
```

### GPU memory issues
Tests use small models by default to avoid memory issues. If needed:
```python
@pytest.mark.gpu
def test_large_batch():
    pass
```

Run only on GPU:
```bash
pytest -m gpu
```

### Random seed
Tests use random data by default. For reproducibility:
```python
def test_reproducible(self):
    torch.manual_seed(42)
    # Test code
```

## Coverage

Generate coverage report:
```bash
pytest --cov=src --cov-report=html
```

View coverage:
```bash
open htmlcov/index.html
```

Target: >80% code coverage for production code.

## Continuous Integration

Add to CI/CD pipeline:
```bash
pytest --cov=src --cov-report=xml
pytest --junitxml=test-results.xml
```

These generate reports for CI platforms like GitHub Actions or GitLab CI.
