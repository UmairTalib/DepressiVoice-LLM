#!/usr/bin/env python3
"""
validate_imports.py
Validates Python environment on compute node
Checks package imports, torch/CUDA, ffmpeg availability
Generates JSON reports for orchestration system
"""

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import os


def log(msg):
    """Print timestamped log message"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {msg}", flush=True)


def check_python_basics():
    """Check Python and pip versions"""
    log("Checking Python basics...")
    
    result = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_executable": sys.executable,
        "pip_version": None
    }
    
    try:
        import pip
        result["pip_version"] = pip.__version__
    except ImportError:
        log("⚠️  pip module not available via import")
        try:
            pip_output = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, text=True, timeout=10
            )
            if pip_output.returncode == 0:
                result["pip_version"] = pip_output.stdout.strip()
        except Exception as e:
            result["pip_version"] = f"error: {str(e)}"
    
    log(f"  Python: {result['python_version']}")
    log(f"  Pip: {result['pip_version']}")
    
    return result


def check_package_imports():
    """Attempt to import all required packages"""
    log("Checking package imports...")
    
    packages = [
        "pandas",
        "numpy",
        "soundfile",
        "librosa",
        "tqdm",
        "scipy",
        "sklearn",  # scikit-learn imports as sklearn
        "matplotlib",
        "seaborn",
        "torch",
        "torchaudio",
        "transformers",
        "sentence_transformers",
        "pyannote.audio",
        "whisperx",
        "yaml",  # PyYAML
    ]
    
    results = []
    
    for pkg_name in packages:
        result = {
            "package": pkg_name,
            "imported": False,
            "version": None,
            "error": None
        }
        
        try:
            # Import the package
            if "." in pkg_name:
                # Handle packages like pyannote.audio
                parts = pkg_name.split(".")
                module = __import__(pkg_name)
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                module = __import__(pkg_name)
            
            result["imported"] = True
            
            # Try to get version
            if hasattr(module, "__version__"):
                result["version"] = module.__version__
            elif hasattr(module, "VERSION"):
                result["version"] = str(module.VERSION)
            
            log(f"  ✓ {pkg_name}: {result['version'] or 'imported'}")
            
        except ImportError as e:
            result["error"] = f"ImportError: {str(e)}"
            log(f"  ✗ {pkg_name}: {result['error']}")
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}"
            log(f"  ✗ {pkg_name}: {result['error']}")
        
        results.append(result)
    
    return results


def check_torch_cuda():
    """Check PyTorch and CUDA availability"""
    log("Checking PyTorch and CUDA...")
    
    result = {
        "torch_available": False,
        "torch_version": None,
        "cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": None
    }
    
    try:
        import torch
        
        result["torch_available"] = True
        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["device_count"] = torch.cuda.device_count()
        
        log(f"  ✓ PyTorch: {result['torch_version']}")
        log(f"  CUDA compiled version: {result['cuda_version']}")
        log(f"  CUDA available: {result['cuda_available']}")
        log(f"  Device count: {result['device_count']}")
        
        if result["cuda_available"]:
            for i in range(result["device_count"]):
                device_name = torch.cuda.get_device_name(i)
                device_capability = torch.cuda.get_device_capability(i)
                result["devices"].append({
                    "id": i,
                    "name": device_name,
                    "capability": f"{device_capability[0]}.{device_capability[1]}"
                })
                log(f"    GPU {i}: {device_name} (compute {device_capability[0]}.{device_capability[1]})")
        else:
            log("  ⚠️  CUDA not available (OK for Phase-1 preprocessing)")
        
    except ImportError as e:
        result["error"] = f"ImportError: {str(e)}"
        log(f"  ✗ PyTorch import failed: {result['error']}")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        log(f"  ✗ PyTorch check failed: {result['error']}")
    
    return result


def check_ffmpeg():
    """Check ffmpeg availability"""
    log("Checking ffmpeg...")
    
    result = {
        "available": False,
        "path": None,
        "version": None,
        "error": None
    }
    
    try:
        # Check if ffmpeg is in PATH
        which_result = subprocess.run(
            ["which", "ffmpeg"],
            capture_output=True, text=True, timeout=5
        )
        
        if which_result.returncode == 0:
            result["path"] = which_result.stdout.strip()
            result["available"] = True
            
            # Get version
            version_result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=5
            )
            
            if version_result.returncode == 0:
                # First line contains version info
                result["version"] = version_result.stdout.split('\n')[0]
            
            log(f"  ✓ ffmpeg: {result['path']}")
            log(f"    Version: {result['version']}")
        else:
            result["error"] = "ffmpeg not found in PATH"
            log(f"  ✗ {result['error']}")
    
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        log(f"  ✗ ffmpeg check failed: {result['error']}")
    
    return result


def get_pip_list():
    """Get list of installed packages"""
    log("Getting pip list...")
    
    try:
        pip_output = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, timeout=30
        )
        
        if pip_output.returncode == 0:
            packages = json.loads(pip_output.stdout)
            log(f"  ✓ Found {len(packages)} installed packages")
            return packages
        else:
            log(f"  ✗ pip list failed: {pip_output.stderr}")
            return []
    
    except Exception as e:
        log(f"  ✗ pip list error: {str(e)}")
        return []


def validate_results(import_results, torch_result, ffmpeg_result):
    """Validate results against Phase-1 requirements"""
    log("Validating environment against Phase-1 requirements...")
    
    errors = []
    warnings = []
    status = "PASS"
    
    # Critical packages for Phase-1
    critical_packages = ["pandas", "numpy", "torch"]
    
    for pkg in critical_packages:
        pkg_result = next((r for r in import_results if r["package"] == pkg), None)
        if not pkg_result or not pkg_result["imported"]:
            errors.append(f"Critical package '{pkg}' failed to import")
            status = "FAIL"
    
    # Check audio tools
    audio_available = False
    
    if ffmpeg_result["available"]:
        audio_available = True
    
    torchaudio_result = next((r for r in import_results if r["package"] == "torchaudio"), None)
    if torchaudio_result and torchaudio_result["imported"]:
        audio_available = True
    
    if not audio_available:
        errors.append("No audio processing tool available (need ffmpeg OR torchaudio)")
        status = "FAIL"
    
    # Check optional but important packages
    optional_packages = ["librosa", "whisperx", "pyannote.audio"]
    
    for pkg in optional_packages:
        pkg_result = next((r for r in import_results if r["package"] == pkg), None)
        if not pkg_result or not pkg_result["imported"]:
            warnings.append(f"Optional package '{pkg}' not available: {pkg_result.get('error', 'unknown') if pkg_result else 'not checked'}")
    
    # CUDA warning (not critical for Phase-1)
    if not torch_result.get("cuda_available", False):
        warnings.append("CUDA not available (OK for Phase-1, required for Phase-2/3)")
    
    # Determine final status
    if status != "FAIL" and warnings:
        status = "WARN"
    
    log(f"  Status: {status}")
    if errors:
        log(f"  Errors: {len(errors)}")
        for err in errors:
            log(f"    - {err}")
    if warnings:
        log(f"  Warnings: {len(warnings)}")
        for warn in warnings:
            log(f"    - {warn}")
    
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings
    }


def main():
    """Main validation orchestration"""
    log("========================================")
    log("ENVIRONMENT VALIDATION ON COMPUTE NODE")
    log("========================================")
    
    # Get output directory from environment
    out_root = os.getenv("OUT_ROOT")
    if not out_root:
        log("❌ ERROR: OUT_ROOT environment variable not set")
        sys.exit(1)
    
    log_dir = Path(out_root) / "env_install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log(f"Output directory: {log_dir}")
    log("")
    
    # Run all checks
    python_basics = check_python_basics()
    log("")
    
    import_results = check_package_imports()
    log("")
    
    torch_result = check_torch_cuda()
    log("")
    
    ffmpeg_result = check_ffmpeg()
    log("")
    
    pip_list = get_pip_list()
    log("")
    
    # Validate
    validation = validate_results(import_results, torch_result, ffmpeg_result)
    log("")
    
    # Write outputs
    outputs = {
        "compute_imports.json": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "python": python_basics,
            "packages": import_results
        },
        "torch_cuda.json": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **torch_result
        },
        "ffmpeg_check.json": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **ffmpeg_result
        },
        "pip_list_compute.json": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "packages": pip_list
        },
        "compute_result.json": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hostname": os.getenv("HOSTNAME", "unknown"),
            "slurm_job_id": os.getenv("SLURM_JOB_ID", "unknown"),
            **validation,
            "referenced_files": {
                "imports": "compute_imports.json",
                "torch_cuda": "torch_cuda.json",
                "ffmpeg": "ffmpeg_check.json",
                "pip_list": "pip_list_compute.json"
            }
        }
    }
    
    log("Writing output files...")
    for filename, data in outputs.items():
        output_path = log_dir / filename
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        log(f"  ✓ {filename}")
    
    log("")
    log("========================================")
    log(f"VALIDATION COMPLETE: {validation['status']}")
    log("========================================")
    log("")
    
    # Exit with appropriate code
    if validation["status"] == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
