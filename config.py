"""Centralized configuration for AutoCAD MCP Pro."""

from __future__ import annotations

import os
from pathlib import Path

#: Default ``MAX_DXF_BYTES``: 512 MB.
DEFAULT_MAX_DXF_BYTES = 512 * 1024 * 1024

#: Default ``MAX_HTTP_BODY_BYTES``: 8 MiB (1.8-17).
DEFAULT_MAX_HTTP_BODY_BYTES = 8 * 1024 * 1024
DEFAULT_UPLOAD_MAX_BYTES = 256 * 1024 * 1024


class Settings:
    """Server configuration loaded from environment variables."""

    @property
    def backend(self) -> str:
        """The engine the environment asks for: ``ezdxf`` or ``auto`` (both run ezdxf).

        Any other value, ``com`` included, is refused at startup by
        ``server._check_backend_setting``: the live COM engine was removed.

        Read live rather than captured in ``__init__``, and deliberately
        read-only. ``_make_backend`` used to read ``os.environ`` itself while
        this attribute had no production reader at all, so
        ``monkeypatch.setattr(settings, "backend", "ezdxf")`` looked like a
        guard and was a no-op — on a machine with AutoCAD open that meant a test
        attaching to the operator's live drawing. Late binding is what keeps
        ``monkeypatch.setenv`` working for the call sites that pin the headless
        backend correctly; the missing setter is what makes the broken spelling
        fail loudly instead of silently.
        """
        return os.environ.get("AUTOCAD_MCP_BACKEND", "auto").lower().strip()

    def __init__(self):
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper().strip()

        raw_paths = os.environ.get("ALLOWED_PATHS", "").strip()
        self.allowed_paths: list[Path] = []
        if raw_paths:
            for p in raw_paths.split(","):
                p = p.strip()
                if p:
                    self.allowed_paths.append(Path(p).resolve())

        self.max_undo_stack: int = int(os.environ.get("MAX_UNDO_STACK", "5"))

        # Reject opening DXF files larger than this many bytes (default 512 MB).
        # Set to 0 to disable the check. It was 50 MB until v1.6's track H: a
        # real plant layout read on 2026-09-24 was a 188 MB DXF, and the
        # understanding tools exist for exactly such drawings. The refusal
        # names the file's size and this variable.
        self.max_dxf_bytes: int = int(os.environ.get("MAX_DXF_BYTES", str(DEFAULT_MAX_DXF_BYTES)))

        # Ceiling on one QA read (drawing_critique, the validator, drawing_finalize,
        # drawing_deliver): the entities of every space are counted first and a
        # drawing holding more is refused with `scan_incomplete` rather than
        # read in part. 0 disables the check; a negative value is refused here
        # (it used to disable the check silently, because the ceiling is only
        # applied when positive).
        self.max_qa_entities: int = int(os.environ.get("MAX_QA_ENTITIES", "2000000"))
        if self.max_qa_entities < 0:
            raise ValueError(
                f"MAX_QA_ENTITIES={self.max_qa_entities} is negative. Set the most entities "
                "one QA read may scan (default 2000000), or 0 to disable the check."
            )

        # Hard cap on list/select limit params to keep tool responses bounded.
        self.max_list_limit: int = int(os.environ.get("MAX_LIST_LIMIT", "5000"))

        # Per-call timeout for the headless ezdxf backend. Every ezdxf method
        # runs under one document lock, so a single hung call (pathological DXF,
        # a render that never returns) would otherwise hold that lock forever
        # and deadlock every later tool call. Generous
        # because matplotlib PDF/PNG rendering of a large drawing is legitimately
        # slow. Set to 0 to disable the check.
        self.ezdxf_call_timeout: float = float(os.environ.get("EZDXF_CALL_TIMEOUT", "120"))

        # How long disconnect() (server shutdown) waits for a worker thread that
        # is still inside a call, in seconds. Nothing can stop a worker thread:
        # with EZDXF_CALL_TIMEOUT=0 a cancelled call keeps the document lock
        # until its worker ends and nothing is ever quarantined, and at shutdown
        # a worker still running after this grace is logged and left behind on a
        # daemon thread, so the process still exits. 0 = do not wait.
        self.shutdown_grace: float = max(0.0, float(os.environ.get("SHUTDOWN_GRACE", "5")))

        # How many undo steps the headless backend keeps. 0 = off, which is the
        # default because it is not free: ezdxf has no journal, so a step is a
        # full DXF snapshot. Measured — 10 entities 3.1 ms / 19 KB, 1000
        # entities 28 ms / 146 KB, 5000 entities 130 ms / 658 KB — and the cost
        # is paid on every mutating call, so switching it on for a large drawing
        # is a real trade the user should make deliberately rather than inherit.
        self.ezdxf_undo_depth: int = max(0, int(os.environ.get("EZDXF_UNDO_DEPTH", "0")))

        # Spec budgets (B07): a count above one of these is refused by path,
        # naming the variable, before anything is drawn; 0 disables a budget.
        # MAX_SPEC_ITEMS bounds each list of a pid_from_spec / arch_plan_from_spec
        # document (measured 2026-10-24: 200 P&ID items place in 451 ms headless,
        # so 2,000 is about 4.5 s).
        # MAX_VESSEL_TRAYS bounds a column's trays (10,000 trays build in 3.2 ms;
        # the budget bounds the block definition, not the time).
        self.max_spec_items: int = int(os.environ.get("MAX_SPEC_ITEMS", "2000"))
        self.max_vessel_trays: int = int(os.environ.get("MAX_VESSEL_TRAYS", "500"))

        # HTTP transport: refuse non-loopback bind unless explicitly opted in.
        self.allow_remote_http: bool = (
            os.environ.get("ALLOW_REMOTE_HTTP", "false").lower().strip() == "true"
        )
        self.mcp_auth_token: str = os.environ.get("MCP_AUTH_TOKEN", "").strip()
        # Idle keep-alive of an HTTP connection. uvicorn's own 5 s is shorter
        # than a reverse proxy's upstream pool (nginx / ALB 60 s, cloudflared
        # 90 s), so the proxy sends a request on a socket uvicorn is closing and
        # answers 502 Bad Gateway. Keep it above the proxy's idle timeout.
        self.http_keepalive_s: float = max(
            1.0, float(os.environ.get("HTTP_KEEPALIVE_S", "").strip() or "120")
        )

        # 1.8-17 (policy.py): which tools a client may reach. The defaults are
        # 1.7's behaviour - every tool, every cost. READ_ONLY runs only `read`
        # tools and writes no file; TOOL_ALLOWLIST / TOOL_DENYLIST take comma-
        # separated names or globs (`entity_*`), an entry naming no tool is
        # warned about at startup; MAX_TOOL_COST is the highest @cad_tool cost
        # a call may carry (read|safe|mutate|destructive|escape).
        from policy import parse_cost, parse_patterns

        self.read_only: bool = os.environ.get("READ_ONLY", "false").lower().strip() == "true"
        self.tool_allowlist: tuple[str, ...] = parse_patterns(os.environ.get("TOOL_ALLOWLIST"))
        self.tool_denylist: tuple[str, ...] = parse_patterns(os.environ.get("TOOL_DENYLIST"))
        self.max_tool_cost: str = parse_cost(os.environ.get("MAX_TOOL_COST"))

        # backends/session_backends.py: each MCP session of a headless server
        # gets its own engine (documents, active document, undo) unless
        # SESSION_ISOLATION=false; an engine idle SESSION_IDLE_TIMEOUT seconds
        # is released with its unsaved documents (0 = kept until shutdown).
        self.session_isolation: bool = (
            os.environ.get("SESSION_ISOLATION", "true").lower().strip() != "false"
        )
        self.session_idle_timeout: float = max(
            0.0, float(os.environ.get("SESSION_IDLE_TIMEOUT", "3600"))
        )

        # preview.py: HTTP Basic credentials of the /preview page; without a
        # password the page is off (404).
        self.preview_user: str = os.environ.get("PREVIEW_USER", "preview").strip() or "preview"
        self.preview_password: str = os.environ.get("PREVIEW_PASSWORD", "")

        # artifact_links.py: a file a tool writes during an HTTP call is also
        # served at GET /artifacts/<token>/<name> for this many seconds
        # (clamped to 24 h; 0 turns the links off). ARTIFACT_PUBLIC_URL is the
        # address clients reach the server at (e.g. behind a TLS proxy);
        # unset, the link is built from the request's own scheme and Host.
        self.artifact_link_ttl_s: float = min(
            86400.0, max(0.0, float(os.environ.get("ARTIFACT_LINK_TTL_S", "3600")))
        )
        self.artifact_public_url: str = (
            os.environ.get("ARTIFACT_PUBLIC_URL", "").strip().rstrip("/")
        )

        # 1.8-17 (http_guard.py): the HTTP transport's request guard. A request
        # carrying an Origin header must name one of HTTP_ALLOWED_ORIGINS
        # (default none: every browser origin is refused 403); its Host must be
        # the bound host:port or one of HTTP_ALLOWED_HOSTS (421); a body over
        # MAX_HTTP_BODY_BYTES (8 MiB; 0 disables) is refused 413. A request
        # authenticated with MCP_AUTH_TOKEN_READONLY gets only the mcp:read
        # scope and runs under the `read` ceiling. stdio is unaffected.
        self.http_allowed_origins: tuple[str, ...] = tuple(
            origin.strip().rstrip("/").lower()
            for origin in os.environ.get("HTTP_ALLOWED_ORIGINS", "").split(",")
            if origin.strip()
        )
        self.http_allowed_hosts: tuple[str, ...] = tuple(
            host.strip().lower()
            for host in os.environ.get("HTTP_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        )
        self.max_http_body_bytes: int = max(
            0, int(os.environ.get("MAX_HTTP_BODY_BYTES", str(DEFAULT_MAX_HTTP_BODY_BYTES)))
        )
        # uploads.py: POST /files stores a client's file (a scan, a DXF, a DWG)
        # and answers its server path. UPLOAD_MAX_BYTES bounds one upload
        # (clamped to 2 GiB; 0 turns uploads off); UPLOAD_RETAIN_S is how long
        # an upload is kept (5 min to 30 days); UPLOAD_DIR is the folder
        # (default <first ALLOWED_PATHS entry>/uploads, else a temp folder).
        self.upload_max_bytes: int = min(
            2 * 1024**3,
            max(0, int(os.environ.get("UPLOAD_MAX_BYTES", str(DEFAULT_UPLOAD_MAX_BYTES)))),
        )
        self.upload_retain_s: float = min(
            30 * 86400.0, max(300.0, float(os.environ.get("UPLOAD_RETAIN_S", "86400")))
        )
        self.upload_dir: str = os.environ.get("UPLOAD_DIR", "").strip()
        self.mcp_auth_token_readonly: str = os.environ.get("MCP_AUTH_TOKEN_READONLY", "").strip()

        # Exposed tool profile: "full" (default, everything) or "lean" (a
        # curated 64-tool drafting/inspection core for clients with tight tool
        # caps). Invalid values fall back to "full" with a warning at startup;
        # so does the "core" profile removed in v1.5.0 (see
        # server.REMOVED_TOOL_PROFILES).
        self.tool_profile: str = os.environ.get("TOOL_PROFILE", "full").lower().strip()

        # Tool packs: "all" (default) advertises every vertical pack; a comma
        # list ("core,pid,settings") advertises only those. core cannot be disabled.
        self.tool_packs: str = os.environ.get("TOOL_PACKS", "all").lower().strip()

        # Tool discovery: "search" (default since 1.8.0) advertises search_tools
        # + call_tool + describe_tool and the DISCOVERY_PINNED tools; "off"
        # advertises the full catalog (see discovery/transform.py and
        # server._apply_discovery_mode).
        #
        # Flipped in 1.8-11 on the plan's condition (section 7-2): spike S1
        # measured that a hidden tool stays directly callable (no per-write
        # call_tool wrapper is forced), and the v8 matrix passed 31/31 with its
        # MCP-client tasks going through call_tool. A breaking wire-level
        # change: DISCOVERY_MODE=off restores the 1.7.0 catalog. Invalid values
        # fall back to "off" with a warning at startup.
        self.discovery_mode: str = os.environ.get("DISCOVERY_MODE", "search").lower().strip()

        # 1.8-11: tools search mode keeps listed beside search_tools / call_tool
        # / describe_tool (comma list; empty pins nothing). A pin the active
        # profile or packs hide is simply not listed; an unregistered name is
        # warned about at startup. Static on purpose: a pin set that moved
        # between sessions would break every client's prompt-cache prefix.
        self.discovery_pinned: tuple[str, ...] = tuple(
            dict.fromkeys(
                name.strip()
                for name in os.environ.get(
                    "DISCOVERY_PINNED", "cad_batch,drawing_info,entity_list"
                ).split(",")
                if name.strip()
            )
        )


settings = Settings()
