# Screenshots

Screenshots of the deployed service at https://faulttrack.jennasrunbooks.com, for the course's "Cloud deployment" criterion. They are taken on the maintainer's host after deploying (the checklist is in issue #29), since the devcontainer has no Fly.io access.

| File | Shows |
|---|---|
| `01-docs-predict-case.png` | `/docs`: **Try it out** on `POST /predict/case` with the fixture case, response visible |
| `02-results-page.png` | the results page at `/` |
| `03-curl-predict-case.png` | the terminal: the fixture `curl` against the custom domain |
| `04-fly-status.png` | `fly status`: one machine, region `iad` |

The fixture is [`tests/fixtures/simple_metrics.csv`](../../tests/fixtures/simple_metrics.csv) with `detection_time` from [`tests/fixtures/inject_time.txt`](../../tests/fixtures/inject_time.txt). Its true fault is `cpu` on `productcatalogservice` ([`tests/fixtures/README.md`](../../tests/fixtures/README.md)).

If a file listed here is missing, it hasn't been added yet.
