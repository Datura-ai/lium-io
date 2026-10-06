import json


from services.port_utils import DEFAULT_PORT_RANGE, get_all_ports


def test_get_all_ports_from_mappings():
    """Test port extraction from JSON port_mappings."""
    # Arrange
    port_mappings = json.dumps([[9000, 9100], [9001, 9101], [9002, 9102]])
    ssh_port = 22

    # Act
    result = get_all_ports(port_range=None, port_mappings=port_mappings, ssh_port=ssh_port)

    # Assert
    assert result == [(9000, 9100), (9001, 9101), (9002, 9102)]


def test_get_all_ports_from_range_dash():
    """Test port generation from dash-separated range."""
    # Arrange
    port_range = "9000-9005"
    ssh_port = 22

    # Act
    result = get_all_ports(port_range=port_range, port_mappings=None, ssh_port=ssh_port)

    # Assert
    assert result == [(9000, 9000), (9001, 9001), (9002, 9002), (9003, 9003), (9004, 9004), (9005, 9005)]


def test_get_all_ports_from_range_comma():
    """Test port generation from comma-separated list."""
    # Arrange
    port_range = "9000, 9005, 9010"
    ssh_port = 22

    # Act
    result = get_all_ports(port_range=port_range, port_mappings=None, ssh_port=ssh_port)

    # Assert
    assert result == [(9000, 9000), (9005, 9005), (9010, 9010)]


def test_get_all_ports_default_range():
    """Test fallback to default range when no port_range or port_mappings."""
    # Arrange
    ssh_port = 22

    # Act
    result = get_all_ports(port_range=None, port_mappings=None, ssh_port=ssh_port)

    # Assert
    assert len(result) == DEFAULT_PORT_RANGE[1] - DEFAULT_PORT_RANGE[0]
    assert result[0] == (20000, 20000)
    assert result[-1] == (65535, 65535)


def test_get_all_ports_excludes_ssh_port():
    """Test that SSH port is excluded from results."""
    # Arrange
    port_range = "9000-9005"
    ssh_port = 9002  # SSH port within range

    # Act
    result = get_all_ports(port_range=port_range, port_mappings=None, ssh_port=ssh_port)

    # Assert
    internal_ports = [p[0] for p in result]
    assert 9002 not in internal_ports
    assert len(result) == 5  # 6 ports minus 1 excluded


def test_get_all_ports_sorted_ascending():
    """Test that ports are sorted by internal port ascending."""
    # Arrange
    port_mappings = json.dumps([[9005, 9105], [9001, 9101], [9003, 9103]])
    ssh_port = 22

    # Act
    result = get_all_ports(port_range=None, port_mappings=port_mappings, ssh_port=ssh_port)

    # Assert
    internal_ports = [p[0] for p in result]
    assert internal_ports == sorted(internal_ports)
    assert result == [(9001, 9101), (9003, 9103), (9005, 9105)]


def test_get_all_ports_empty_range():
    """Test fallback to default range when port_range is empty string."""
    # Arrange
    ssh_port = 22

    # Act
    result = get_all_ports(port_range="", port_mappings=None, ssh_port=ssh_port)

    # Assert - empty string is falsy, so default range is used
    assert len(result) == DEFAULT_PORT_RANGE[1] - DEFAULT_PORT_RANGE[0]


def test_get_all_ports_mappings_priority_over_range():
    """Test that port_mappings takes priority over port_range."""
    # Arrange
    port_range = "8000-8005"
    port_mappings = json.dumps([[9000, 9100]])
    ssh_port = 22

    # Act
    result = get_all_ports(port_range=port_range, port_mappings=port_mappings, ssh_port=ssh_port)

    # Assert - port_mappings wins
    assert result == [(9000, 9100)]
