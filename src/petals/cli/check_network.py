import argparse
import sys

from petals.server.reachability import check_direct_reachability

def main():
    parser = argparse.ArgumentParser(description="Check if your Petals node is reachable from the Internet.")
    parser.add_argument(
        "--port",
        type=int,
        default=31330,
        help="The port your Petals server is listening on (default: 31330).",
    )
    parser.add_argument(
        "--public_ip",
        type=str,
        default=None,
        help="Your public IP address. Useful if you are behind a NAT.",
    )

    args = parser.parse_args()

    print(f"Checking reachability on port {args.port}...")

    kwargs = {"port": args.port}
    if args.public_ip:
        kwargs["host_maddrs"] = [f"/ip4/{args.public_ip}/tcp/{args.port}"]
    else:
        kwargs["host_maddrs"] = [f"/ip4/0.0.0.0/tcp/{args.port}"]

    try:
        is_reachable = check_direct_reachability(**kwargs)
    except Exception as e:
        print(f"\nError while checking reachability: {e}")
        sys.exit(1)

    if is_reachable:
        print("\nSuccess! Your node is reachable from the Internet.")
        print(f"Other peers can connect to you via port {args.port}.")
    else:
        print("\nFailure! Your node is not reachable from the Internet.")
        print("\nThis usually means you are behind a NAT or a firewall.")
        print("To fix this, you need to set up port forwarding on your router:")
        print(f"  1. Access your router's administration interface.")
        print(f"  2. Find the 'Port Forwarding' or 'Virtual Server' settings.")
        print(f"  3. Forward TCP port {args.port} to the local IP address of this machine.")
        print(f"  4. Ensure any local firewalls (e.g., Windows Firewall, ufw) allow inbound traffic on port {args.port}.")
        print("\nAfter configuring port forwarding, you can run this tool again to verify.")
        sys.exit(1)

if __name__ == "__main__":
    main()
