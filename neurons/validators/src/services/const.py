MIN_JOB_TAKEN_TIME = 20

GPU_MODEL_RATES = {
    "NVIDIA B300 SXM6 AC": .05,
    "NVIDIA B200": .05,
    "NVIDIA H200": .56,
    "NVIDIA H200 NVL": .49,
    "NVIDIA H100 80GB HBM3": .10,
    "NVIDIA H100 NVL": .01,
    "NVIDIA H100 PCIe": .01,
    "NVIDIA H800 80GB HBM3": 0.02,
    "NVIDIA H800 NVL": 0.01,
    "NVIDIA H800 PCIe": 0.01,
    # Supported operationally, but excluded from the unrented pool by default.
    "NVIDIA GeForce RTX 5080": 0.0,
    "NVIDIA GeForce RTX 5070 Ti": 0.0,
    "NVIDIA GeForce RTX 5070": 0.0,
    "NVIDIA GeForce RTX 5060 Ti": 0.0,
    "NVIDIA GeForce RTX 5060": 0.0,
    "NVIDIA GeForce RTX 5090": 0.025,
    # Supported operationally, but excluded from the unrented pool by default.
    "NVIDIA GeForce RTX 4080 SUPER": 0.0,
    "NVIDIA GeForce RTX 4080": 0.0,
    "NVIDIA GeForce RTX 4070 Ti": 0.0,
    "NVIDIA GeForce RTX 4070 Ti SUPER": 0.0,
    "NVIDIA GeForce RTX 4070 SUPER": 0.0,
    "NVIDIA GeForce RTX 4070": 0.0,
    "NVIDIA GeForce RTX 4060 Ti": 0.0,
    "NVIDIA GeForce RTX 4060": 0.0,
    "NVIDIA GeForce RTX 4090": 0.05,
    "NVIDIA GeForce RTX 4090 D": 0.02,
    # "NVIDIA RTX 4000 Ada Generation": 0.005,
    "NVIDIA RTX PRO 4000 Blackwell": 0.0,
    "NVIDIA RTX PRO 5000 Blackwell": 0.0,
    "NVIDIA RTX 5000 Ada Generation": 0.0,
    "NVIDIA RTX 5880 Ada Generation": 0.0,
    "NVIDIA RTX 6000 Ada Generation": 0.01,
    "NVIDIA L4": 0.01,
    "NVIDIA L40S": 0.01,
    "NVIDIA L40": 0.01,
    # "NVIDIA RTX 2000 Ada Generation": 0.001,
    "NVIDIA A100 80GB PCIe": 0.01,
    "NVIDIA A100-SXM4-80GB": 0.05,
    "NVIDIA A10 Tensor Core GPU": 0.0,
    "NVIDIA RTX A6000": 0.009,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 0.025, # 2.5x 6000 ada
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": 0.027, # 2.7x 6000 ada
    "NVIDIA RTX A5000": 0.002,
    "NVIDIA RTX A4500": 0.002,
    "NVIDIA RTX A4000": 0.002,
    "NVIDIA RTX A2000": 0.0,
    # "NVIDIA A40": 0.002,
    # "NVIDIA A30": 0.002,
    "NVIDIA T4 Tensor Core GPU": 0.0,
    "NVIDIA Tesla V100 Tensor Core GPU": 0.0,
    "NVIDIA TITAN V": 0.0,
    "NVIDIA GeForce RTX 3090 Ti": 0.0,
    "NVIDIA GeForce RTX 3090": 0.02,
    "NVIDIA GeForce RTX 3080 Ti": 0.0,
    "NVIDIA GeForce RTX 3080": 0.0,
    "NVIDIA GeForce RTX 3070 Ti": 0.0,
    "NVIDIA GeForce RTX 3070": 0.0,
    "NVIDIA GeForce RTX 3060 Ti": 0.0,
    "NVIDIA GeForce RTX 3060 Laptop GPU": 0.0,
    "NVIDIA GeForce RTX 3060": 0.0,
    "NVIDIA GeForce RTX 3050": 0.0,
    "NVIDIA Quadro RTX 8000": 0.0,
    "NVIDIA Quadro RTX 6000": 0.0,
    "NVIDIA Quadro RTX 5000": 0.0,
    "NVIDIA TITAN RTX": 0.0,
    "NVIDIA GeForce RTX 2080 Ti": 0.0,
    "NVIDIA GeForce RTX 2080 SUPER": 0.0,
    "NVIDIA GeForce RTX 2070 SUPER": 0.0,
    "NVIDIA GeForce RTX 2060 SUPER": 0.0,
    "NVIDIA GeForce RTX 2060": 0.0,
    "NVIDIA GeForce GTX 1660 Ti": 0.0,
    "NVIDIA GeForce GTX 1660 SUPER": 0.0,
    "NVIDIA GeForce GTX 1660": 0.0,
    "NVIDIA Tesla P100": 0.0,
    "NVIDIA Tesla P40": 0.0,
    "NVIDIA Quadro P4000": 0.0,
    "NVIDIA TITAN Xp": 0.0,
    "NVIDIA GeForce GTX 1080 Ti": 0.0,
    "NVIDIA GeForce GTX 1080": 0.0,
    "NVIDIA GeForce GTX 1070 Ti": 0.0,
    "NVIDIA GeForce GTX 1070": 0.0,
    "NVIDIA GeForce GTX 1060": 0.0,
    "NVIDIA Tesla M40": 0.0,
}

MAX_UPLOAD_SPEED = 1000
MAX_DOWNLOAD_SPEED = 1000

JOB_TAKEN_TIME_WEIGHT = 0.9
UPLOAD_SPEED_WEIGHT = 0.05
DOWNLOAD_SPEED_WEIGHT = 0.05

MAX_GPU_COUNT = 14

UNRENTED_MULTIPLIER = 1

GPU_UTILIZATION_LIMIT = 5  # percent
GPU_MEMORY_UTILIZATION_LIMIT = 5  # percent

# DAH-2427 ghost GPU: an orphaned CUDA kernel left by a hard-killed GPU process pins the
# card at full utilization with no memory and no process attached (observed for days on
# RTX PRO 6000 nodes). The "no memory" half is expressed in whichever unit each data source
# provides: the periodic check reads NVML memory-utilization percent, the teardown sweep
# reads nvidia-smi used MiB.
GPU_WEDGE_UTILIZATION_MIN = 95  # percent
GPU_WEDGE_MEMORY_MAX = 1  # percent (periodic check, NVML memory-utilization)
GPU_WEDGE_SWEEP_MEMORY_MAX_MIB = 16  # teardown sweep, nvidia-smi memory.used
GPU_WEDGE_SWEEP_SETTLE_SECONDS = 5  # let GPU state settle after a container is removed

MIN_PORT_COUNT = 3
BATCH_PORT_VERIFICATION_SIZE = 300
BATCH_PORT_TIMEOUT = 40
BATCH_PORT_CONCURRENCY = 200
BATCH_HEALTH_CHECK_TIMEOUT = 10  # seconds to wait for batch verifier to become healthy
VERIFY_JOB_REQUIRED_COUNT = 6 * 24 * 1

# Emission split between the rented "mining" pool and the unrented + burn pool is
# sourced from shared config at runtime via core.config.get_total_burn_emission()
# (DAH-2274) — the backend is the source of truth. The expected production value
# used by the test suite lives in tests/constants.py.
BURNER_EMISSION = 0.01

# Rental Price Incentive Constants
TEMPO = 360  # blocks per epoch (from subtensor)
SECONDS_PER_BLOCK = 12  # seconds per block
FIXED_RATIO = 0.41  # fixed constant for rental emission calculation

IS_NOT_DEPOSITED_SCORE_MULTIPLIER = 0.5
DOCKER_DIND_IMAGE = "daturaai/dind:0.0.1"

LIB_NVIDIA_ML_DIGESTS = {
    "535.54.03": "49e63c42aa95bba6b9aa562ee57e496c:15a37892671187547b6dd21a07e8149315e529211dc30ca6ee8d8d089a338d53",
    "535.86.10": "7351f43a025ecdde1208a7a4e2f1cf26:ec9b270c4fdf2b51515c9eb5f185cbc0478322c43a3876caab80ebfb5cd1dab1",
    "535.104.05": "01265268cdc362e02952c51fb4f49d11:f17ec20319ddfe5beeb0e80ee0a89fc312d23ec764bc8ac640c3df860a0b0566",
    "535.129.03": "15da23a575907f6d455896dffb4cc8ab:82e40f00b57a91840b62386094c266bf962a1c6304f5e898300fdb186df12246",
    "535.154.05": "79858da5e3a0283a76212fa92d361203:3e25c6fc550943319b475998d3154c48279ba3a92a32b1441adc1e2070a63378",
    "535.161.07": "e1bd6e1ec1211ee5fea246f1635d8364:b3fdee85f8d01defb963e286d4faaad939b860a3dc5313f48d31e32e2048c8a0",
    "535.161.08": "0507ce7072af817d9bd6efa0609f2738:fb0862f58f33e93e6709482dcafb44c4665ccd24e3d34ee10c91a9e122d04bce",
    "535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f",
    "535.183.06": "03ed7fa2134095b32f9d0d24a774c6ba:5899d928c18f39656d4c5a573a509acaf621b896644abe31e78ad715171c2ce6",
    "535.261.03": "eece3d8387df42b9d1906710cebf784b:bff3d13ebff1ac22b8a1f5d2c0c05910564ec868c2e70bc5f971dcaf82a36765",
    "535.274.02": "939800fdf0d88c143e416203d68a7d39:25e82746a4eb51597e9e901bc59d5a4e05c5971f8e0069df49c6d4f6cfeb4b51",
    "535.288.01": "b67f475a5ac428ef45106c8c3718d24a:844e81737f3a5d0db3230f307e5e97fd1e3eae2e3588d9aaa1498c55083a0a5d",
    "535.216.01": "96479a06139fc5261d06f432970d6a7b:63ec13c213f50fc193f1b56f4d56ea1dddb5974d505d3dfc670a10824d800753",
    "535.216.03": "189634bf960b9a2efe1af8011d27ccf7:f9aec03b89ff0fba865b7059d8dce3258f2fda8f5ae748b3da98a65b5e06b46e",
    "535.230.02": "cc34ae85c2238b9a49067e683c1998cf:23767b1f9a39ace459e4d9950e1225ba6d89f4ac0299de9e7815b5c413f53acf",
    "535.247.01": "b44440a3031606c8034b7f4b73f88e9c:8355dd1d32973e66cb83083be2937d59ce30d794ee59c7def5df3c33dc790f5a",
    "545.23.06": "5ad33588e91af67139efb54fe9fefc68:7864c8334d3d8b9c092e3f6f89b288c50fa3c517cffeec7397a3dbc23de62098",
    "545.29.06": "85ad949d7553ab96cce5c811e229c7c7:f2ad84109af6facf93cb33f9ca79354c2c4672ed7d90a9e09a00954c6a82b438",
    "550.120": "48be49d0e792b5ee76f73857c0bef35a:3bc49d033d45882a0e71a6e99a64105ad1a4f00cdad06b33bfd37175b8584373",
    "550.127.05": "bfa2733eee442016792bcbf130156e3d:f196afa282f435e48b19561849d451e3de26932af8a76f75136f1dfc5533f247",
    "550.54.14": "6137cba707dab1ab1b8e88e6a6fd89a0:ac63ee701ffc3611b4acfd9c612f772f8c1d038df3f135cea7d5a1377d279f84",
    "550.54.15": "9625642dcf8765f52e332c8e38fbef73:a12e4d671706b1e96abe91047611c6b413b9ec9ae0fb68eae8c4790a5df8cfaa",
    "550.78": "1f335d1f068931fe7f2ce13117d1602b:54b57e46525c3b1c9776442a65a5a010b92773a165b9f5f9e5b2a24be063b969",
    "550.90.07": "c95828f8a8ab7f17743b40561b812c96:03d713a02c4b20b08b00f4c67ea03956dd2a48d1b4ca930dffb6024387157f16",
    "550.90.12": "d7702d394ab213a725abeb345185a072:b2675b61ec855fb37cec0e11e2738a98f909d737f0b31963e6687826267b7864",
    "550.144.03": "41c3f2bcda579d3203408977c2a39d90:51d4ac8866536c6d21b1e91af63b658a4d24042cd3a686cc9283fdbcd946be65",
    "550.163.01": "63bdcc86bbc24cb9d4340c767d497d9a:c2f8e560da76086bbd83941ec5f2e14626a32ad53d9ffc8619e01bdf56d56b22",
    "555.42.02": "0262f396e80847dccefc8ccf52cff1ae:a61d5a06765bd40420b0e105e2cb68a7895de882dbca7571b345db1146d50d15",
    "555.42.06": "69774adffa76471490e6d8fac9067725:f72c427a42a17f1c9d5689c1cb2216263b144f298cf3143cc6bb9f7ab13f745f",
    "560.28.03": "6d6e0122cff1ac777a9e37ba09b886cb:05256acca07d038283bc263f684368f3f4031e314fbad95d1da8ff1946eff237",
    "560.35.03": "93a3f8ef77af86b79314c00b0788aeed:660081fa0dfaa772c8dd11ca90e6d326273bd84813e2d809024352e308e35334",
    "560.35.05": "1eec299b50e33a6cfa5155ded53495ab:9ec08a9bc379bdcbd0bae105556b75d6ee89be1566032c4f0b82d25f89aa6431",
    "565.57.01": "c801dd3fc4660f3a8ddf977cfdffe113:05c8df9cec2c01c935594c57d37287ee358ed4e759270dbfec350061d342ad80",
    "550.127.08": "ac925f2cd192ad971c5466d55945a243:0107189f70491091fe47f0cbeb90d2ff21494a115bb710dcca3c693112fb8252",
    "550.142": "e68b535a61be6434fc7f12450561a3d0:e967e2bb6d0f636aa452e2790c92db6ff82f124e370c1417f934e150e81b3071",
    "570.86.15": "b9fdb5be515d09610f26b60e7e061ac8:edf88a4e27853377f5e3a6daede90926ef76e850ce45594444c6ff457cd3b086",
    "570.124.06": "acdd9e2ca82caa7e7daf07d395e67aab:fc48c5f5e44996bd41b1892a0ece6714bc715ad7c41361f691d64ebeb52fc4d4",
    "570.86.16": "5624cc31de5a9dc7caeedefd7803a509:656d876d2a453a61b5f65e1c382811b49a494af0f0b69b01ae916f5de3234b7e",
    "570.133.20": "94717b6033b598c8f963dedb43e187a3:39eb997c5e0b7fa181cb908f8d481bb684c5dc7d68e063852b76c7bc9f133ae4",
    "570.144": "0da473423dd6958c42a9347f4d336cef:a040b22849ab22e2e29337e62d1dd86ef1838b4bef8bf3efa44e3525e01f04a0",
    "570.148.08": "ce76cfd47b0262450053d4ac50cbcab1:82ed7464081aa881e13e9b673d294fd8bc3f067338ff97fa4c4fd241c92e0a07",
    "570.153.02": "829381960585f40b058d55d394d438ae:a90c1c885346f2c270b6c33267c55f177021677dca3727c8d4a990ac94797e32",
    "570.133.07": "069476e75bb8e576c0c78e9d2dd6abdd:b6b6cad3a3ab99ce7faae3896a9a497d3e2f91df045fcbc559ba397c5e2650d0",
    "570.158.01": "6387bfd5309e3741d1e0ef56274babc0:50cfaa3b5462b264ae4439e5850186408b698cb72cde3d0894d77dd2c374808a",
    "570.172.08": "021fa367c3fb8fa54422aa8c79a6a784:4ffe4baabaf73f76ce941269ce2e80da1b09ec5ab4af30bffebe6d1979a3e62e",
    "570.169": "9d1f189e783be38b52e86aada4ea721b:988e1c7f9dcfd30b15f2b55926bc38da46411c9b68ab7245a6d42760450bd30a",
    "570.195.03": "05481133d8b1ae692cf28a1cdb47e728:e50f81689bfc29408e1255207befbf5f9550fad74e22a2ad45cc9f2869637aa9",
    "570.211.01": "d94663bcbcdcadcfc2ca3a2406bc8518:36b25540f20f5cbcf47faed0f386abcd4896e7c768145f5b7d87ff7e8dd2b021",
    "575.51.03": "d01d04bf5e770102cbf9fe3c2302d903:b2910dea374c6c038f860b3ebf9e1408ee5946dd5da59cab6023220f968e8eff",
    "575.57.08": "b49c325ff0d74199597d9b19b9b407a6:ba4159cd8797a30e275e46a17b174b3d9a4a7768f82da9ca7263832da3f6d816",
    "575.64.03": "ffad1bd4cfbf8abe4f2edd24e5687c64:6e2e60d6f2ad3957f33465626e7793286c741cf8a9e3a6da8fbaacf3fc453370",
    "575.64.05": "e06f67418707f3ff4792cb270c6b36ef:3d9a6015bc20a7a15dda9aafa07174fd67723a59915964c2b7ba3f004fc14c8c",
    "580.65.06": "c98bc1ae60a208db4db9f19e50aaf4a3:54658f1b05c647760509be6729e41e98a73bae855b5d09984d735abce4818bd7",
    "580.76.05": "03fda3e528d0eabbf5ffa73670d2deaa:da6bcacfb548646e0318f54a74df766a975e459e08509a451b4140941ab896ea",
    "580.82.07": "f7e2623164324de58f1c3a5daf6ed475:008cc04afc61b8317ed4ddccfa2d1bd367ceb6e71334eb352dc846e0ac450340",
    "580.82.09": "175e6a907fbc54f344223f7da3164d19:68d2ee4a36fad63bf0507718376b67f9247de8402c041e7e68f63f29f536299c",
    "580.95.05": "abb3628879c4801b3e5f5e3351d01c96:9933cfe943ca96d9b3c352221154dd0afd5ed5406fcb9a3d15935ff4f2c380a9",
    "580.105.08": "2ffbe6a28257332c99ba5bd4d9a115de:bee24a7507366126cdf7441c7fec4705e85c3a61c9669e08d70d7d86a8ed6f99",
    "580.126.09": "09c22d9afa2e321bfeb63eb93eb3d1a8:7cf5b9e876f8a082616b5b632f426fdf2ac6fb2e04bfa90c5a3f4eebafe90910",
    "580.119.02": "2a549d896267cd4537302a7c7f572a70:0b8424b9fb42caf4a718fb1cb820232a0cd37fb94ecf64cbaf93c57dea454acd",
    "580.126.16": "602c2f2fdfdbbfeadbeda157a12c89b4:1195dafd6b9e41645240ea97c4bb8e2e5c53a87a8b4bd18e4bf1befcb2f3da7b",
    "580.126.18": "c743bd753c2a72562ba81cd53655b482:1a0f5afd550344d357b071d70c84ea1cf2553675b4c68268ccc14da64d562ab5",
    "580.126.20": "aeb23ff04f08663f8d29699b7166965a:85f43e9c005904dbd3cb3dea42042236d4ade1b635739581af204b8383bcc1ac",
    "580.142": "ba092110c430e70382ee1274bf561796:f2e0fe7110fcc12cbeaf685f60a0c1bfd37d748010b1f8fba9c4f3c8bdd2aa93",
    "580.159.03": "e20a51c15ea69c8689c870cfff8a274c:a610e3e6a9ebb901372dcd2b912a7a040c35406ced930290dd902b3fbf1c797c",
    "580.159.04": "112ed00d4a4880bb33ccc001df34c2f0:941c7f7999351d8b9b69f275a20ebd2818e3314678fc12dba6da8a6f12b74d27",
    "580.167.08": "fa0c084327835d0369e5307a1ba3a882:c7eac74626efce631035360d6a5d1f9d72b02d81739ab0adbf0521f6c6a0f10a",
    "580.173.02": "bcc62db30ccee22da8c64ecde69ead52:82d7b58d3f870b0635e8999c56eb8fd2d795c920daa2661e628ca523eee5f557",
    "590.44.01": "b5f88f19314d6e0b951e350129d018dc:560b1aff40089484c67d63889db11e7be9bfdae6f476006d5ef3bd3c37b547bd",
    "590.48.01": "7c4674fdfdeb75c20af25bd41e3a0de6:12f3bcd4ba447599a2077297e3a4ff4288205b082c2e76346718ae85c058c4b2",
    "595.45.04": "1dbe78234657c5522cb8fe9c9cfde141:f7d1d0eb39f16e8e88a77e9c5127e9a2a93e4b5b1344be1f18faedcfb67bc58f",
    "595.58.03": "af7923894f6ad89eafb78c03daf422a9:9a0ef13c817030b07f931cbe6115a70f7674ecbd3bee6047417ffc7ae699ed1b",
    "595.71.05": "020cd1156cbce5ebbf12963d0c70496e:9eb4358b7fea76556657670a6ae6b0017eaa4256b56c421a36626bf8c2b5f3f5",
    "610.43.02": "5ad6c02411f730682597558ae8f3a9f8:2dc828b3f5027f98e05c7607c1d8129d11bd28de4c2091c5cd7e32dbc21ec172",
}

# VerifyX Validation Constants
MEMORY_ALLOCATION_PERCENTAGE = 75
MEMORY_MIN_TEST_GB = 8
MEMORY_MAX_TEST_GB = 128
STORAGE_MIN_AVAILABLE_GB = 100
STORAGE_THROUGHPUT_TEST_GB = 5
NETWORK_TIMEOUT_SECONDS = 120
NETWORK_MIN_DOWNLOAD_SPEED_MBPS = 50.0

PREFERRED_POD_PORTS = [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009]

POD_CONTAINER_PREFIX = "pod_"
FILLER_CONTAINER_PREFIX = "filler_"
# DAH-2475: prefix of the persistent DPHN model/runtime cache volumes. The backend builds the full
# name with the model + runtime version baked in; the validator only needs the prefix, to recognise
# which volumes belong to the cache when sweeping or reclaiming them.
DPHN_CACHE_VOLUME_PREFIX = "dphn_cache_"
# DAH-2475: what one node's DPHN cache costs on disk, and how much room the node must keep free after
# downloading it. The floor mirrors the backend's EXECUTORS_FILTER_MIN_GB — below it the node drops out
# of the rental listing, where neither renters nor fillers can reach it — and the margin is headroom
# above that floor. Duplicated here on purpose: only the host knows whether the cache already exists,
# so the affordability decision has to live next to that fact. Keep in step with the backend.
DPHN_CACHE_SIZE_GB = 40
DPHN_CACHE_LISTING_FLOOR_GB = 100
DPHN_CACHE_FREE_MARGIN_GB = 50
FILLER_CONTAINER_GRACE_MINUTES = 15
# ISSUE-050: a filler run younger than this is not penalized for a missing container —
# it may still be finishing its create/stop race with the backend snapshot.
FILLER_LIVENESS_GRACE_MINUTES = 10

# Owner of an executor's active default job, as reported by compute-app.
# An executor running the miner's OWN default job earns no unrented incentive.
DEFAULT_JOB_OWNER_MINER = "miner"
# A Lium-owned default job (e.g. the PEARL idle filler) — Lium keeps the unrented incentive and may
# lower the node's GPU power limit, so GpuPowerLimitCheck exempts these nodes (DAH-2356).
DEFAULT_JOB_OWNER_LIUM = "lium"

# Container name prefixes that count as "rental-related" on an executor.
# All producers of short-lived containers competing for the 9100-9130 port range
# MUST be listed here so that container_cleanup and wait_for_port_check_containers
# both see them. Adding a new prefix is a one-line edit that both guards inherit.
#   pod_*          — long-lived user rentals (validator-owned)
#   filler_*       — long-lived filler runtime (validator-owned, not customer-rented)
#   container_*    — validator DinD/port-check probes (hotkey-scoped)
#   health_check_* — backend executor_health_check probes (hotkey-agnostic, epoch-suffixed)
RENTAL_CONTAINER_PREFIXES = ("pod_", "filler_", "container_", "health_check_")

# For simplicity, store whitelist in code. Can be updated to use DB if needed. 
TDX_WHITELIST = {
    "OS_IMAGE_HASH": set(
        [
            # DAH-2338 — official upstream dstack-nvidia 0.5.11 (meta-dstack v0.5.11
            # release, git_revision ce04e924, NVIDIA driver 595.58.03, prod/non-dev
            # image). Single-image policy: the legacy 0.5.5 hashes (upstream 9b69bb16…
            # and DAH-2311 rebuilt dbfde543…) are removed — CVMs still booting them
            # must be redeployed on 0.5.11.
            "a6eafc5f007f642d8ea90c7fa8881f1e6715720ccb531941a28218f4f26d7b02",
        ]
    ),
    # Compose-file hashes vary per environment. G2: each approved hash carries a
    # monotonically increasing release version — append-only, next release gets the
    # next integer. Acceptance requires version >= settings.TDX_MINIMUM_COMPOSE_VERSION
    # ("at/above the floor", newest-wins) so a known-bad release is retired by raising
    # the floor instead of aging out of a rolling window. Membership checks
    # (`hash in TDX_WHITELIST["COMPOSE_HASH"][env]`) keep working on the dict keys.
    "COMPOSE_HASH": {
        "PROD": {
            # DAH-2338 — executor-v1.108 measured compose: executor-runner pinned to
            # daturaai/compute-subnet-executor-runner:latest @
            # sha256:f85b948b6cb280423b17e34ea28c0f98139243ecea32cd42106f90d19dc619f1
            # (executor @ sha256:94b5e734…, pushed 2026-07-07), init/pre-launch as
            # of 077f42c1 (G1 GPU attestation guest env whitelist). Replaces v2
            # 0995e41b… (executor-v1.107 runner b58211e7…) — CVMs on the old digest
            # must redeploy with EXECUTOR_RUNNER_IMAGE_DIGEST=sha256:f85b948b… .
            # Assumes default `lium-cvm.sh new` flags (no --enable-logs/--enable-sysinfo).
            "ab4d14336f0762c0d8ec7631a69148246661de84ceead7a215f8a33b74fd43e6": 3,
        },
        "STAGE": {
            "72c9c91a1b72cb016e1ed2ac85cdb1414502165dc3eb3723642f30a5ef0fcb11": 1,
            # DAH-2338 — 0.5.11 staging compose: executor-runner pinned to
            # daturaai/compute-subnet-executor-runner:dev digest sha256:58db7cd5…
            # (pushed 2026-07-06), pre-launch sysbox force-install included.
            # Assumes default `lium-cvm.sh new` flags (no --enable-logs/--enable-sysinfo).
            "feedf7cb08a9905f3ef66eb9bab4af310310965589e5aed070594c326bf44e16": 2,
        },
        "LOCAL": {
            "2d655bf8eca15eaec6cc5800acae99eaeb21fc3dafcfcf594139c827596a7828": 1,
        },
    }
}
