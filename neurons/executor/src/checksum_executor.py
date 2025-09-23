import argparse
import hashlib


def calculate_file_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of a file."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except FileNotFoundError:
        return ""


def main():
    parser = argparse.ArgumentParser(description="VerifyX Library Checksum Calculation")
    parser.add_argument("--lib_path", type=str, default="/usr/lib/libverifyx.so", help="Path to the shared library")

    args = parser.parse_args()
    print(calculate_file_checksum(args.lib_path))


if __name__ == "__main__":
    main()