minor
`RentedPod.gpu_uuids` (optional) and `RentedExecutorsResponse.filler_gpu_uuids_by_executor` (optional, default empty): the GPU UUIDs the backend assigned to each pod, and per executor the GPU UUIDs its active fillers hold (a whole-node filler lists every card). An older backend sends neither.
