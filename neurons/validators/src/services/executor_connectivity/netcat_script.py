from services.executor_connectivity.models import PortPair


class NetcatScript:
    """Builds netcat test scripts."""

    @staticmethod
    def batch(ports: list[PortPair], token: str, batch_idx: int) -> str:
        """Build script for batch port testing."""
        port_list = ' '.join([str(p.internal) for p in ports])
        return f'''
echo "Batch {batch_idx}: starting ports {port_list}" >&2
for port in {port_list}; do
    (
        echo "Binding port $port" >&2
        body="{token}:$port"
        printf "HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: ${{#body}}\\r\\nConnection: close\\r\\n\\r\\n$body" | nc -l -p $port
        nc_status=$?
        [ $nc_status -eq 0 ] && echo "Port $port served" >&2 || echo "Port $port failed (nc exit $nc_status)" >&2
    ) &
done
wait
echo "Batch {batch_idx}: completed" >&2
sleep 60
'''

    @staticmethod
    def single(port: PortPair, token: str) -> str:
        """Build script for single port testing."""
        return f'''
echo "Testing port {port.internal}" >&2
body="{token}:{port.internal}"
printf "HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: ${{#body}}\\r\\nConnection: close\\r\\n\\r\\n$body" | nc -l -p {port.internal}
'''
