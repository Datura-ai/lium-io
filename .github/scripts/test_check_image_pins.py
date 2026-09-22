import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import check_image_pins as c  # noqa: E402

DIGEST = "sha256:" + "a" * 64


def refs_and_skips(files):
    refs, skipped = c.collect(files)
    return set(refs), {pin.ref: reason for pin, reason in skipped}


class ComposeTest(unittest.TestCase):
    def test_image_pins_and_their_lines(self):
        refs, _ = c.collect(
            {
                "neurons/executor/docker-compose.yml": (
                    "services:\n"
                    "  executor-runner:\n"
                    "    image: daturaai/compute-subnet-executor-runner:latest\n"
                    "  watchtower:\n"
                    "    image: 'daturaai/lium-watchtower:1.2.0'  # the updater\n"
                )
            }
        )
        self.assertEqual(
            set(refs),
            {"daturaai/compute-subnet-executor-runner:latest", "daturaai/lium-watchtower:1.2.0"},
        )
        self.assertEqual(
            refs["daturaai/lium-watchtower:1.2.0"][0].where(),
            "neurons/executor/docker-compose.yml:5",
        )

    def test_service_with_build_is_skipped_everywhere_its_tag_is_used(self):
        refs, skips = refs_and_skips(
            {
                "x/docker-compose.local.yml": (
                    "services:\n"
                    "  executor:\n"
                    "    build:\n"
                    "      context: .\n"
                    "    image: compute-subnet-executor:local\n"
                    "  monitor:\n"
                    "    image: compute-subnet-executor:local\n"
                    "  autoheal:\n"
                    "    image: willfarrell/autoheal\n"
                )
            }
        )
        self.assertEqual(refs, {"willfarrell/autoheal"})
        self.assertIn("built", skips["compute-subnet-executor:local"])

    def test_build_override_file_marks_the_base_files_service(self):
        refs, skips = refs_and_skips(
            {
                "kp/docker-compose.yaml": "services:\n  aesmd:\n    image: lium-aesmd:local\n",
                "kp/docker-compose.build.yaml": "services:\n  aesmd:\n    build:\n      context: .\n",
            }
        )
        self.assertEqual(refs, set())
        self.assertIn("built", skips["lium-aesmd:local"])

    def test_alternative_compose_file_keeps_its_own_image(self):
        # app.local.yml builds `executor`; app.dev.yml pulls its own `executor` image
        refs, _ = refs_and_skips(
            {
                "x/docker-compose.app.local.yml": "services:\n  executor:\n    build: .\n    image: e:local\n",
                "x/docker-compose.app.dev.yml": "services:\n  executor:\n    image: daturaai/e:dev\n",
            }
        )
        self.assertEqual(refs, {"daturaai/e:dev"})

    def test_variables(self):
        refs, skips = refs_and_skips(
            {
                "docker-compose.yml": (
                    "services:\n"
                    "  a:\n"
                    "    image: daturaai/e@${EXECUTOR_IMAGE_SHA256}\n"
                    "  b:\n"
                    "    image: redis:${REDIS_TAG:-7.4.2}\n"
                ),
                ".github/workflows/cd.yml": "jobs:\n  x:\n    container:\n      image: ${{ env.IMAGE }}:staging\n",
            }
        )
        self.assertEqual(refs, {"redis:7.4.2"})
        self.assertIn("variable", skips["daturaai/e@${EXECUTOR_IMAGE_SHA256}"])
        self.assertIn("variable", skips["${{ env.IMAGE }}:staging"])

    def test_pulumi_style_image_keys(self):
        refs, _ = refs_and_skips(
            {
                "infrastructure/Pulumi.prod.yaml": (
                    "config:\n"
                    "  lium:env:\n"
                    '    EXECUTOR_IMAGE_REF: "daturaai/compute-subnet-executor:latest"\n'
                    "    DSTACK_VERIFIER_IMAGE_DOWNLOAD_URL: https://example.com/x.tar.gz\n"
                    "    image_tag_mutability: MUTABLE\n"
                )
            }
        )
        self.assertEqual(refs, {"daturaai/compute-subnet-executor:latest"})

    def test_comment_and_block_scalar_are_not_pins(self):
        refs, _ = refs_and_skips({"a.yml": "# image: nope:1\nimage: |\n  multi\n"})
        self.assertEqual(refs, set())


class DockerfileTest(unittest.TestCase):
    def test_from_forms(self):
        refs, skips = refs_and_skips(
            {
                "Dockerfile": (
                    "ARG BASE_IMAGE=python:3.11-slim\n"
                    "ARG UNSET\n"
                    "FROM $BASE_IMAGE AS build\n"
                    "FROM --platform=linux/amd64 python:3.11-slim@" + DIGEST + "\n"
                    "FROM build\n"
                    "FROM scratch\n"
                    "FROM ${UNSET}\n"
                    "ARG AFTER=ignored:1\n"
                    "FROM ubuntu:${AFTER}\n"
                ),
                "neurons/executor/Dockerfile.runner": "FROM docker:26-cli\n",
            }
        )
        self.assertEqual(refs, {"python:3.11-slim", "python:3.11-slim@" + DIGEST, "docker:26-cli"})
        self.assertEqual(skips["build"], "earlier build stage")
        self.assertEqual(skips["scratch"], "scratch")
        self.assertIn("variable", skips["${UNSET}"])
        self.assertIn("variable", skips["ubuntu:${AFTER}"])

    def test_additional_contexts_name(self):
        refs, skips = refs_and_skips(
            {
                "e2e/docker-compose.e2e.yml": (
                    "services:\n"
                    "  tester:\n"
                    "    build:\n"
                    "      context: ./tester\n"
                    "      additional_contexts:\n"
                    "        validator: service:validator\n"
                    "    image: lium-e2e/tester\n"
                ),
                "e2e/tester/Dockerfile": "FROM validator\n",
            }
        )
        self.assertEqual(refs, set())
        self.assertIn("additional_contexts", skips["validator"])


def completed(rc, stderr=""):
    return subprocess.CompletedProcess([], rc, "", stderr)


class DockerInspectTest(unittest.TestCase):
    def test_classification(self):
        cases = {
            "ok": completed(0),
            "missing": completed(1, "no such manifest: docker.io/daturaai/lium-watchtower:1.2.0"),
        }
        for want, proc in cases.items():
            with (
                mock.patch.object(c.subprocess, "run", return_value=proc),
                mock.patch.object(c, "registry_head") as head,
            ):
                self.assertEqual(c.docker_inspect("daturaai/lium-watchtower:1.2.0"), want)
                head.assert_not_called()

    def test_rate_limit_and_unknown_errors_fall_back_to_head(self):
        for stderr in (
            "toomanyrequests: You have reached your pull rate limit",
            "dial tcp: i/o timeout",
        ):
            with (
                mock.patch.object(c.subprocess, "run", return_value=completed(1, stderr)),
                mock.patch.object(c, "registry_head", return_value="ok") as head,
            ):
                self.assertEqual(c.docker_inspect("postgres:14.0-alpine"), "ok")
                head.assert_called_once_with("postgres:14.0-alpine")

    def test_digest_goes_to_head(self):
        with (
            mock.patch.object(c.subprocess, "run") as run,
            mock.patch.object(c, "registry_head", return_value="missing"),
        ):
            self.assertEqual(c.docker_inspect("python:3.11-slim@" + DIGEST), "missing")
            run.assert_not_called()


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.git("init", "-q")
        self.write(
            "docker-compose.yml", "services:\n  w:\n    image: daturaai/lium-watchtower:1.1.1\n"
        )
        self.commit()

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def write(self, name, text):
        (self.repo / name).write_text(text)

    def commit(self):
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")

    def run_main(self, *args, results=None):
        checked = []

        def fake(ref):
            checked.append(ref)
            return (results or {}).get(ref, "ok")

        with mock.patch.object(c, "docker_inspect", side_effect=fake), mock.patch("builtins.print"):
            rc = c.main(["--repo", str(self.repo), *args])
        return rc, checked

    def test_all_pins_checked_and_missing_fails(self):
        self.assertEqual(self.run_main(), (0, ["daturaai/lium-watchtower:1.1.1"]))
        rc, _ = self.run_main(results={"daturaai/lium-watchtower:1.1.1": "missing"})
        self.assertEqual(rc, 1)

    def test_base_checks_only_new_references(self):
        self.write(
            "docker-compose.yml", "services:\n  w:\n    image: daturaai/lium-watchtower:1.2.0\n"
        )
        self.write("Dockerfile", "FROM daturaai/lium-watchtower:1.1.1\n")
        self.commit()
        rc, checked = self.run_main(
            "--base", "HEAD^1", results={"daturaai/lium-watchtower:1.2.0": "missing"}
        )
        self.assertEqual((rc, checked), (1, ["daturaai/lium-watchtower:1.2.0"]))

    def test_unchecked_is_a_failure(self):
        rc, _ = self.run_main(
            results={"daturaai/lium-watchtower:1.1.1": "error: HTTP 429 (registry rate limit)"}
        )
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
