"""Synthetic downstream MCP servers.

Each package under here is a *real* MCP server from a protocol perspective --
it advertises tools, validates arguments, and answers ``tools/call`` -- but
every tool is fake from an execution perspective. Nothing touches the
filesystem, the network, a mailbox, a calendar, a database, or a shell.
"""
