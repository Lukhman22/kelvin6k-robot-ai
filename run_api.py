"""
Start the Kelvin6K API server (backend for the React dashboard).

    python run_api.py                 # http://127.0.0.1:8000
    python run_api.py --port 9000
    python run_api.py --reload        # auto-reload during development
    python run_api.py --restart       # stop whatever holds the port, then start

Docs: http://127.0.0.1:8000/docs
"""

import argparse
import asyncio
import logging
import socket
import subprocess
import sys
import time

log = logging.getLogger("kelvin6k.shutdown")

# How long to wait for open connections (the browser's MJPEG camera stream)
# before closing them. Without a bound, Ctrl+C waits for a browser that has no
# reason to hang up, because an MJPEG stream is designed to stay open forever.
GRACEFUL_SHUTDOWN_SECONDS = 5


def _quiet_socket_teardown(loop: asyncio.AbstractEventLoop) -> None:
    """
    Silences ONE specific, harmless asyncio artifact on Windows.

    CPython's _ProactorBasePipeTransport._call_connection_lost ends with:

        finally:
            if hasattr(self._sock, 'shutdown') and self._sock.fileno() != -1:
                self._sock.shutdown(socket.SHUT_RDWR)

    If the peer has already sent a TCP reset, that shutdown() raises
    ConnectionResetError [WinError 10054]. The line above it has ALREADY called
    protocol.connection_lost(), so the request is finished and the application
    has been notified; only the socket teardown failed, and _sock.close() on the
    next line still runs. Nothing is lost -- but asyncio's default handler logs
    it as "Exception in callback ...", which reads like a real fault.

    This filter is deliberately narrow: it matches ONLY ConnectionResetError
    with winerror 10054 raised from that exact callback. Every other exception,
    including any other ConnectionResetError, goes to the default handler
    untouched, so genuine application errors are never hidden.
    """
    previous = loop.get_exception_handler()

    def handler(active_loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        origin = f"{context.get('message', '')} {context.get('handle', '')!r}"
        if (
            isinstance(exc, ConnectionResetError)
            and getattr(exc, "winerror", None) == 10054
            and "_call_connection_lost" in origin
        ):
            log.debug("Ignored socket teardown reset after connection_lost: %s", exc)
            return
        if previous is not None:
            previous(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def port_owner(host: str, port: int):
    """
    Returns the PID holding host:port, or None if the port is free.

    Worth the trouble because uvicorn's own message for this ("[Errno 10048]
    only one usage of each socket address...") arrives AFTER "Application
    startup complete", which reads as though the server started fine. The old
    process keeps serving, so stale code appears to be a code bug -- a
    confusion that cost real debugging time on this project.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, port))
        return None  # bound successfully => nothing else has it
    except OSError:
        pass
    finally:
        probe.close()

    # Something holds it. Ask Windows which process.
    try:
        output = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return -1  # in use, owner unknown

    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING":
            if parts[1].endswith(f":{port}"):
                try:
                    return int(parts[4])
                except ValueError:
                    return -1
    return -1


def stop_pid(pid: int) -> bool:
    result = subprocess.run(
        ["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Kelvin6K API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="stop whatever is already using the port, then start",
    )
    args = parser.parse_args()

    owner = port_owner(args.host, args.port)
    if owner is not None:
        if args.restart:
            if owner > 0:
                print(f"Stopping the process already on port {args.port} (PID {owner})...")
                if not stop_pid(owner):
                    print(f"Could not stop PID {owner}. Close that window and retry.")
                    sys.exit(1)
                time.sleep(1.5)
            else:
                print(f"Port {args.port} is in use but the owner could not be identified.")
                sys.exit(1)
        else:
            print()
            print("=" * 68)
            print(f" A Kelvin6K backend is ALREADY RUNNING on {args.host}:{args.port}")
            if owner > 0:
                print(f" It is process ID {owner}.")
            print("=" * 68)
            print()
            print(" Nothing was started, so the running server is untouched.")
            print(" It may be running OLDER code than the files on disk -- if you")
            print(" just changed Python code, restart it so the change takes effect:")
            print()
            print("     python run_api.py --restart")
            print()
            print(" To keep the running one instead, just leave it and reload the")
            print(" browser page. To start a second server on another port:")
            print()
            print(f"     python run_api.py --port {args.port + 1}")
            print()
            sys.exit(1)

    # Imported only now: --help and the port check stay instant, and a
    # "port busy" message does not wait for the whole framework to load.
    import uvicorn

    print(f"Kelvin6K API -> http://{args.host}:{args.port}  (docs at /docs)")
    print("Press Ctrl+C in THIS window to stop it (closing the window with X")
    print("leaves the process running and holding the port).")

    if args.reload:
        # The reloader runs the app in a child process we don't own, so the
        # exception handler cannot be installed here. Development only.
        uvicorn.run(
            "api.server:app",
            host=args.host,
            port=args.port,
            reload=True,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
        )
        return

    # Run the server ourselves purely so the event loop is reachable before
    # serving starts. uvicorn.Server.serve() still installs its own SIGINT
    # handler, so Ctrl+C behaves exactly as before.
    config = uvicorn.Config(
        "api.server:app",
        host=args.host,
        port=args.port,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    server = uvicorn.Server(config)

    async def serve() -> None:
        _quiet_socket_teardown(asyncio.get_running_loop())
        await server.serve()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        # Reaching here means Ctrl+C arrived outside uvicorn's own handler.
        # Exiting quietly is correct; the shutdown hooks have already run.
        pass


if __name__ == "__main__":
    main()
