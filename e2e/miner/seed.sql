-- The miner's executor table as `lium mine`/the portal sync would leave it: one live executor (the stack's) and one
-- the provider took offline. Both assigned to the e2e validator. Idempotent. Values come from stack.env via psql -v.
INSERT INTO executor (uuid, address, port, validator, price_per_hour, price_per_gpu)
VALUES (:'live_uuid', :'live_ip', :'port', :'validator', 0.5, 0.5),
       (:'dead_uuid', :'dead_ip', :'port', :'validator', 0.5, 0.5)
ON CONFLICT DO NOTHING;
