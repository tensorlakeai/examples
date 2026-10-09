"""Install a scanner checkout with registry guards and no lifecycle hooks."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


def install(project: Path) -> None:
    manifest = json.loads((project / "package.json").read_text())
    policy_path = project / "pnpm-workspace.yaml"
    policy = yaml.safe_load(policy_path.read_text()) if policy_path.exists() else {}
    if policy is None:
        policy = {}
    if not isinstance(policy, dict):
        raise ValueError("Expected a mapping in pnpm-workspace.yaml")
    policy["minimumReleaseAge"] = max(int(policy.get("minimumReleaseAge", 0)), 1440)
    # Broad first-party exemptions would defeat the scanner's registry policy.
    policy.pop("minimumReleaseAgeExclude", None)
    policy.update(
        minimumReleaseAgeStrict=True,
        minimumReleaseAgeIgnoreMissingTime=False,
        trustLockfile=False,
        blockExoticSubdeps=True,
        strictDepBuilds=True,
        ignoreScripts=True,
        optimisticRepeatInstall=False,
        managePackageManagerVersions=False,
        packageManagerStrict=False,
        packageManagerStrictVersion=False,
    )
    workspaces = manifest.get("workspaces", [])
    if isinstance(workspaces, dict):
        workspaces = workspaces.get("packages", [])
    if workspaces and "packages" not in policy:
        policy["packages"] = workspaces
    # pnpm reads overrides from workspace YAML, rather than npm's root field.
    for name, version in manifest.get("overrides", {}).items():
        overrides = policy.setdefault("overrides", {})
        if isinstance(version, dict):
            for child, child_version in version.items():
                if child == ".":
                    overrides[name] = child_version
                else:
                    overrides[f"{name}>{child}"] = child_version
        else:
            overrides[name] = version
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False))
    command = ["pnpm", "--config.managePackageManagerVersions=false", "--config.packageManagerStrict=false"]
    if not (project / "pnpm-lock.yaml").exists():
        if (project / "package-lock.json").exists() or (project / "npm-shrinkwrap.json").exists():
            subprocess.run([*command, "import"], cwd=project, check=True)
        else:
            subprocess.run([*command, "install", "--lockfile-only", "--ignore-scripts"], cwd=project, check=True)
    subprocess.run([*command, "install", "--frozen-lockfile", "--ignore-scripts"], cwd=project, check=True)


if __name__ == "__main__":
    install(Path(sys.argv[1]))
