This is the script to be loaded in the executor to check a lot of ports in one command.

## Build and Push

```bash
docker build -t arhangel662/batch-port-verifier:latest .

docker push arhangel662/batch-port-verifier:latest
```


## Start on server
```bash
docker run  -e API_PORT={OPEN_PORT} --network=host arhangel662/batch-port-verifier:latest
```


## Check ports
in
```bash
curl -X POST http://{EXTERNAL_IP}:{OPEN_PORT}/check-ports \
  -H "Content-Type: application/json" \
  -d '{"external_ip":"{EXTERNAL_IP}","ports":[[9000,9000], [9001,9002]]}'
```
out:
```json
{
  "duration": 0.004006862640380859,
  "success_count": 1,
  "results": {
    "9000": true,
    "9001": false
  }
}
```