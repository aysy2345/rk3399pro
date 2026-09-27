import json

from tools.rknn import check_environment


def _matching_probe():
    return {
        "python_version": "3.6.9",
        "system": "Linux",
        "machine": "x86_64",
        "encoding": "UTF-8",
        "package_versions": {
            "rknn": "1.7.1",
            "numpy": "1.16.3",
            "cv2": "4.5.4",
            "onnxruntime": "1.11.1",
        },
        "libraries": {"stdc++": True, "protobuf": True},
    }


def test_matching_conversion_environment_passes():
    report = check_environment.evaluate_environment(**_matching_probe())

    assert report["passed"] is True
    assert all(item["passed"] for item in report["checks"])


def test_environment_report_names_each_mismatch():
    probe = _matching_probe()
    probe.update(
        {
            "python_version": "3.11.9",
            "system": "Windows",
            "machine": "arm64",
            "encoding": "cp936",
            "package_versions": {
                "rknn": None,
                "numpy": "1.26.4",
                "cv2": "4.10.0",
                "onnxruntime": None,
            },
            "libraries": {"stdc++": False, "protobuf": False},
        }
    )

    report = check_environment.evaluate_environment(**probe)

    assert report["passed"] is False
    failures = {item["name"] for item in report["checks"] if not item["passed"]}
    assert failures == {
        "python",
        "system",
        "machine",
        "encoding",
        "package:rknn",
        "package:numpy",
        "package:cv2",
        "package:onnxruntime",
        "library:stdc++",
        "library:protobuf",
    }


def test_compatible_package_build_suffixes_are_accepted():
    probe = _matching_probe()
    probe["package_versions"] = {
        "rknn": "1.7.1+build1",
        "numpy": "1.16.3",
        "cv2": "4.5.4.60",
        "onnxruntime": "1.11.1",
    }

    report = check_environment.evaluate_environment(**probe)

    assert report["passed"] is True


def test_main_prints_json_and_returns_nonzero_on_failure(monkeypatch, capsys):
    probe = _matching_probe()
    probe["package_versions"]["rknn"] = None
    monkeypatch.setattr(check_environment, "probe_environment", lambda: probe)

    result = check_environment.main([])

    payload = json.loads(capsys.readouterr().out)
    assert result == 1
    assert payload["passed"] is False
    assert any(item["name"] == "package:rknn" for item in payload["checks"])
