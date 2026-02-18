# Contributing to Gravity-Lang

Thank you for your interest in contributing to Gravity-Lang! 🚀

## How to Contribute

### Reporting Bugs 🐛

If you find a bug, please [open an issue](https://github.com/dill-lk/Gravity-Lang/issues) with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your Python version and operating system

### Suggesting Features ✨

Feature requests are welcome! Please:
- Check if the feature was already requested
- Explain the use case and why it would be valuable
- Provide examples if possible

### Contributing Code 💻

1. **Fork the repository**
   ```bash
   git clone https://github.com/dill-lk/Gravity-Lang.git
   cd Gravity-Lang
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new features
   - Update documentation

4. **Run tests**
   ```bash
   python -m unittest discover -s tests -v
   ```

5. **Submit a pull request**
   - Describe what your PR does
   - Reference any related issues
   - Wait for review

## Development Setup

```bash
# Install dependencies
pip install numpy pyinstaller

# Run tests
python -m unittest discover -s tests -v

# Build executable
python gravity_lang_interpreter.py build-exe --name gravity-lang --outdir dist
```

## Code Style

- Follow PEP 8 Python style guide
- Use type hints for function signatures
- Write docstrings for public functions
- Keep functions focused and small

## Adding Tests

Tests live in `tests/test_interpreter.py`. Example:

```python
def test_new_feature(self):
    """Test description"""
    interp = GravityInterpreter()
    script = """
    # Your test script
    """
    output = interp.execute(script)
    self.assertEqual(len(output), expected_count)
```

## Documentation

- Update README.md for user-facing changes
- Update ADVANCED_FEATURES.md for technical details
- Add examples in `examples/` directory

## Areas Needing Help

### High Priority
- [ ] Better error messages with line numbers
- [ ] Variable support (`let x = 100`)
- [ ] More print options (speed, distance, energy)
- [ ] Performance benchmarks

### Medium Priority
- [ ] 3D visualization/plotting
- [ ] More example scripts
- [ ] Video tutorials
- [ ] API reference documentation

### Future
- [ ] C++ physics kernel
- [ ] GPU acceleration
- [ ] Web interface
- [ ] Cloud execution

## Questions?

- Open an [issue](https://github.com/dill-lk/Gravity-Lang/issues)
- Start a [discussion](https://github.com/dill-lk/Gravity-Lang/discussions)

## Code of Conduct

Be respectful, constructive, and professional. We're all here to make Gravity-Lang better!

---

**Thank you for contributing!** Every contribution, no matter how small, helps make Gravity-Lang better. 🙏
