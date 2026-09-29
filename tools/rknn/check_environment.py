"""Check the pinned Ubuntu RKNN Toolkit 1.7.1 conversion environment."""

import argparse
import ctypes.util
import importlib
import json
import locale
import platform
import sys


EXPECTED_PACKAGES = {
    "rknn": "1.7.1",
    "numpy": "1.16.3",
    "cv2": "4.5.4",
    "onnxruntime": "1.10.0",
}

DISTRIBUTION_NAMES = {
    "rknn": "rknn-toolkit",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "onnxruntime": "onnxruntime",
}


def _check(name, expected, actual, passed):
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "passed": bool(passed),
    }


def _version_matches(actual, expected):
    if actual is None:
        return False
    actual = str(actual)
    return actual == expected or actual.startswith(expected + ".") or actual.startswith(
        expected + "+"
    )


def evaluate_environment(
    python_version,
    system,
    machine,
    encoding,
    package_versions,
    libraries,
):
    """Evaluate an injected environment probe and return JSON-safe checks."""

    checks = [
        _check("python", "3.6.9", python_version, python_version == "3.6.9"),
        _check("system", "Linux", system, str(system).lower() == "linux"),
        _check(
            "machine",
            "x86_64",
            machine,
            str(machine).lower() in ("x86_64", "amd64"),
        ),
        _check(
            "encoding",
            "UTF-8",
            encoding,
            str(encoding).lower().replace("-", "") == "utf8",
        ),
    ]
    for package_name in ("rknn", "numpy", "cv2", "onnxruntime"):
        expected = EXPECTED_PACKAGES[package_name]
        actual = package_versions.get(package_name)
        checks.append(
            _check(
                "package:{}".format(package_name),
                expected,
                actual,
                _version_matches(actual, expected),
            )
        )
    for library_name in ("stdc++", "protobuf"):
        present = bool(libraries.get(library_name))
        checks.append(
            _check(
                "library:{}".format(library_name),
                "available",
                "available" if present else "missing",
                present,
            )
        )
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _module_version(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        import pkg_resources

        return pkg_resources.get_distribution(
            DISTRIBUTION_NAMES[module_name]
        ).version
    except Exception:
        return None


def probe_environment():
    """Collect the current process and native-library environment."""

    return {
        "python_version": "{}.{}.{}".format(
            sys.version_info[0], sys.version_info[1], sys.version_info[2]
        ),
        "system": platform.system(),
        "machine": platform.machine(),
        "encoding": locale.getpreferredencoding(False),
        "package_versions": {
            name: _module_version(name) for name in EXPECTED_PACKAGES
        },
        "libraries": {
            "stdc++": bool(ctypes.util.find_library("stdc++")),
            "protobuf": bool(ctypes.util.find_library("protobuf")),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="检查 Ubuntu RKNN Toolkit 1.7.1 模型转换环境"
    )
    parser.parse_args(argv)
    report = evaluate_environment(**probe_environment())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
