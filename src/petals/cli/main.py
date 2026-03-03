import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: petals <command> [args]")
        print("Commands:")
        print("  run_server       Start a Petals compute node")
        print("  run_dht          Start a Petals DHT node")
        print("  run_http_server  Start an OpenAI-compatible HTTP server")
        print("  configure        Run the interactive configuration wizard")
        print("  check-network    Check if your node is reachable from the Internet")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the command from argv so the submodules parse their own args
    del sys.argv[1]

    if command == "run_server":
        from petals.cli.run_server import main as run_server_main
        run_server_main()
    elif command == "run_dht":
        from petals.cli.run_dht import main as run_dht_main
        run_dht_main()
    elif command == "run_http_server":
        from petals.cli.run_http_server import main as run_http_server_main
        run_http_server_main()
    elif command == "configure":
        from petals.cli.configure import run_wizard
        run_wizard()
    elif command == "check-network":
        from petals.cli.check_network import main as check_network_main
        check_network_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: run_server, run_dht, run_http_server, configure, check-network")
        sys.exit(1)

if __name__ == "__main__":
    main()
