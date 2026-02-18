# Building Gravity-Lang on Windows

## 🪟 Quick Build Guide

### Option 1: Using Batch Script (Easiest)

1. **Download Python 3.8+** from [python.org](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during installation

2. **Open Command Prompt** in the Gravity-Lang directory

3. **Run the build script:**
   ```cmd
   build-windows.bat
   ```

4. **Result:** `dist\gravity-lang-windows.exe` (25-30 MB)

---

### Option 2: Using PowerShell Script

1. **Download Python 3.8+** from [python.org](https://www.python.org/downloads/)

2. **Open PowerShell** in the Gravity-Lang directory

3. **If needed, enable script execution (run once):**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Run the build script:**
   ```powershell
   .\build-windows.ps1
   ```

5. **Result:** `dist\gravity-lang-windows.exe` (25-30 MB)

---

### Option 3: Manual Build

```powershell
# 1. Install dependencies
pip install pyinstaller numpy

# 2. Build executable
python gravity_lang_interpreter.py build-exe --name gravity-lang-windows --outdir dist

# 3. Test it
.\dist\gravity-lang-windows.exe --version
.\dist\gravity-lang-windows.exe run examples\moon_orbit.gravity
```

---

## 🧪 Testing the Executable

```powershell
# Check version
.\dist\gravity-lang-windows.exe --version

# Run an example
.\dist\gravity-lang-windows.exe run examples\moon_orbit.gravity

# Validate syntax
.\dist\gravity-lang-windows.exe check examples\solar_system.gravity
```

---

## 📊 Expected Results

### Build Output
```
Building executable...
✅ Executable created: dist\gravity-lang-windows.exe
✅ Size: ~25-30 MB
✅ Type: Windows PE32+ executable
```

### File Details
- **Name**: `gravity-lang-windows.exe`
- **Size**: 25-30 MB (includes Python + NumPy)
- **Type**: Standalone Windows executable
- **Requirements**: None (self-contained)

---

## ❓ Troubleshooting

### "Python not found"
**Solution**: Add Python to PATH
```
1. Search "Environment Variables" in Windows
2. Edit PATH variable
3. Add Python installation directory
4. Restart Command Prompt
```

### "pip not found"
**Solution**: Reinstall Python with pip checked, or:
```cmd
python -m ensurepip --upgrade
```

### "PyInstaller failed"
**Solution**: Check disk space (need ~500 MB) and run as administrator

### "Module not found"
**Solution**: Install dependencies:
```cmd
pip install --upgrade pyinstaller numpy
```

---

## 🚀 Distribution

Once built, you can:

1. **Distribute the .exe** - Copy `dist\gravity-lang-windows.exe` to any Windows PC
2. **No Python required** - Executable is self-contained
3. **No installation** - Just run it directly

---

## 📝 Build Artifacts

After building, you'll see:
```
Gravity-Lang/
├── dist/
│   └── gravity-lang-windows.exe    ← Your executable
├── build/                           ← Temporary build files (can delete)
└── gravity-lang-windows.spec        ← PyInstaller spec (can delete)
```

To clean up:
```cmd
rmdir /s /q build
del gravity-lang-windows.spec
```

---

## 🔄 Automated Builds (GitHub Actions)

For automatic multi-platform builds, see `.github/workflows/build.yml` (if available).

This can automatically build for:
- Windows (x64)
- Linux (x64)
- macOS (x64, ARM64)

---

## ✅ Verification Checklist

After building, verify:

- [ ] Executable exists: `dist\gravity-lang-windows.exe`
- [ ] Size is reasonable: 25-30 MB
- [ ] Version works: `.\dist\gravity-lang-windows.exe --version`
- [ ] Examples work: `.\dist\gravity-lang-windows.exe run examples\moon_orbit.gravity`
- [ ] No errors in output

---

## 📞 Need Help?

- 📖 [Read WINDOWS.md](WINDOWS.md)
- 🐛 [Report issues](https://github.com/dill-lk/Gravity-Lang/issues)
- 💬 [Ask in Discussions](https://github.com/dill-lk/Gravity-Lang/discussions)

---

**Ready to build? Run `build-windows.bat` and you're done!** 🎉
