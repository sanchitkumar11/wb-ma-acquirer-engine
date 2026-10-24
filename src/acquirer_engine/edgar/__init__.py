"""SEC EDGAR client: rate-limited, disk-cached, with as-of guards.

Reused from my earlier project buyer-universe-agent (src/bua/edgar/, commit 0786d02): only the
client, cache, rate limiter and parsers. The frames API helpers and the DERA and SIC modules were
left out because this engine doesn't screen the whole market.
"""
