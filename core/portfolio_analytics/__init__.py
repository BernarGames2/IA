"""The investor's own portfolio on real prices: dashboard (value, evolution, returns against benchmarks,
allocation, alerts, movements) and an advanced risk report.

There is no user database: the page keeps the positions and the movements in the browser and sends them with
each request, so every function here is stateless. Prices are daily adjusted closes from the market-data provider
(Yahoo Finance chart endpoint in production), converted to reais date by date with USDBRL=X; cash in reais is
modelled by the CDI (BCB SGS 12). Nothing is filled in when a series is missing: the asset is reported as
without data and left out of the numbers that need it.
"""
