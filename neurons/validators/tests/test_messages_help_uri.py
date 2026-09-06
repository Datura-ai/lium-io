"""Every verdict that costs a provider score carries a help link.

The miner portal renders `help_uri` as "Learn more" on the node's error panel; only the
VerifyX templates set one, so the other ~50 error/warning codes shipped without a link.
"""

from services.task.messages import (
    REASON_CODE_DOCS_URL,
    VERIFYX_DEBUG_DOC_URL,
    GpuUsageMessages,
    PortCountMessages,
    SysboxRequiredMessages,
    VerifyXMessages,
    render_message,
)
from tests.helpers import make_context


def test_error_and_warning_templates_default_to_the_reason_code_docs():
    ctx = make_context()

    error = render_message(GpuUsageMessages.ORPHANED_CONTAINER, ctx=ctx, check_id="c")
    warning = render_message(PortCountMessages.INSUFFICIENT_PORTS, ctx=ctx, check_id="c")

    assert error.severity == "error"
    assert error.help_uri == REASON_CODE_DOCS_URL
    assert warning.severity == "warning"
    assert warning.help_uri == REASON_CODE_DOCS_URL


def test_template_and_call_site_links_win_over_the_default():
    ctx = make_context()

    own = render_message(VerifyXMessages.VERIFY_FAILED_NETWORK_SPEED_TOO_SLOW, ctx=ctx, check_id="c")
    override = render_message(
        GpuUsageMessages.ORPHANED_CONTAINER, ctx=ctx, check_id="c", help_uri="https://example.test/x"
    )

    assert own.help_uri == VERIFYX_DEBUG_DOC_URL
    assert override.help_uri == "https://example.test/x"


def test_info_verdicts_carry_no_link():
    ctx = make_context()

    ok = render_message(SysboxRequiredMessages.SYSBOX_OK, ctx=ctx, check_id="c")

    assert ok.severity == "info"
    assert ok.help_uri is None


def test_severity_override_decides_the_default():
    # A template rendered as "info" at the call site (the observe-mode pattern) gets no link.
    ctx = make_context()

    downgraded = render_message(
        PortCountMessages.INSUFFICIENT_PORTS, ctx=ctx, check_id="c", severity="info"
    )

    assert downgraded.help_uri is None
