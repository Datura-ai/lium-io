# compose healthcheck for the miner: the 422 the signed route gives an empty body = uvicorn up, migrations done (the image has no curl)
import os, urllib.error, urllib.request
url = f"http://127.0.0.1:{os.environ.get('INTERNAL_PORT', '8000')}/executors"
try:
    urllib.request.urlopen(urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"}), timeout=3)
except urllib.error.HTTPError as e:
    raise SystemExit(0 if e.code == 422 else 1)
except Exception:
    raise SystemExit(1)
raise SystemExit(1)
