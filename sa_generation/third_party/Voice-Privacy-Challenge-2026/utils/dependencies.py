from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement, InvalidRequirement
from packaging.version import parse
import re
import importlib


def check_dependencies(requirements_file):
    missing_dependencies = []
    nonmatching_versions = []

    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if len(line) == 0 or line.startswith('#'):
                continue
            
            # Handle Git URL dependencies (e.g., git+https://github.com/...)
            if line.startswith('git+'):
                # Extract package name from Git URL (usually the repo name)
                # Format: git+https://github.com/user/repo.git or git+https://github.com/user/repo.git@branch
                match = re.search(r'github\.com/[^/]+/([^/]+?)(?:\.git|@|$)', line)
                if match:
                    repo_name = match.group(1).replace('.git', '')
                    # Try common package name variations
                    # First try the repo name as-is, then try with underscores instead of hyphens
                    package_names_to_try = [repo_name, repo_name.replace('-', '_')]
                    found = False
                    for package_name in package_names_to_try:
                        # Try to import the package to verify it's installed
                        try:
                            importlib.import_module(package_name)
                            found = True
                            break
                        except ImportError:
                            # Also try checking via importlib.metadata
                            try:
                                version(package_name)
                                found = True
                                break
                            except PackageNotFoundError:
                                continue
                    if not found:
                        missing_dependencies.append(line)
                else:
                    # If we can't extract package name, mark as potentially missing
                    # User will need to verify manually
                    missing_dependencies.append(line)
                continue
            
            # Handle standard package requirements
            try:
                requirement = Requirement(line)
                try:
                    installed_version = version(requirement.name)
                    if not parse(installed_version) in requirement.specifier:
                        nonmatching_versions.append((requirement, installed_version))
                except PackageNotFoundError:
                    missing_dependencies.append(line)
            except InvalidRequirement:
                # If it's not a valid requirement format, try to check if it's a package name
                # This handles edge cases
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                try:
                    version(package_name)
                except PackageNotFoundError:
                    missing_dependencies.append(line)

    error_msg = ''
    if missing_dependencies:
        error_msg += f'Missing dependencies: {", ".join(missing_dependencies)}.\n'
    if nonmatching_versions:
        error_msg += f'The following packages are installed with a version that does not match the requirement:\n'
        for req, installed_version in nonmatching_versions:
            error_msg += f'Package: {req.name}, installed: {installed_version}, required: {str(req.specifier)}\n'

    if len(error_msg) > 0:
        raise ModuleNotFoundError(f'{error_msg}--Make sure to install {requirements_file} to run this code!--')

