"""Shared pytest configuration — options and markers for the test suite."""


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live-server integration tests against MIMA_TEST_BASE_URL",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires a running Mima server")
