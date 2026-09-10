# The validator image itself plus pytest. `validator` is the image compose builds from neurons/validators/Dockerfile
# (the `validator` service, passed in as a named build context), so nothing from the validator's build is repeated
# here and the tester cannot drift from what the validator runs.
FROM validator

RUN pdm run python -m ensurepip --upgrade >/dev/null && pdm run python -m pip install --no-cache-dir -q pytest==8.4.2 pytest-timeout==2.4.0 requests

COPY entrypoint.sh /e2e-entrypoint.sh
RUN chmod +x /e2e-entrypoint.sh
ENTRYPOINT ["/e2e-entrypoint.sh"]
CMD ["pytest", "--help"]
