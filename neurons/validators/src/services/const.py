MIN_JOB_TAKEN_TIME = 20

GPU_MODEL_RATES = {
    "NVIDIA B200": .05,
    "NVIDIA H200": .56,
    "NVIDIA H100 80GB HBM3": .10,
    "NVIDIA H100 NVL": .01,
    "NVIDIA H100 PCIe": .01,
    "NVIDIA H800 80GB HBM3": 0.02,
    "NVIDIA H800 NVL": 0.01,
    "NVIDIA H800 PCIe": 0.01,
    "NVIDIA GeForce RTX 5090": 0.025,
    "NVIDIA GeForce RTX 4090": 0.05,
    "NVIDIA GeForce RTX 4090 D": 0.02,
    # "NVIDIA RTX 4000 Ada Generation": 0.005,
    "NVIDIA RTX 6000 Ada Generation": 0.01,
    "NVIDIA L4": 0.01,
    "NVIDIA L40S": 0.01,
    "NVIDIA L40": 0.01,
    # "NVIDIA RTX 2000 Ada Generation": 0.001,
    "NVIDIA A100 80GB PCIe": 0.01,
    "NVIDIA A100-SXM4-80GB": 0.05,
    "NVIDIA RTX A6000": 0.009,
    "NVIDIA RTX A5000": 0.002,
    "NVIDIA RTX A4500": 0.002,
    "NVIDIA RTX A4000": 0.002,
    # "NVIDIA A40": 0.002,
    # "NVIDIA A30": 0.002,
    "NVIDIA GeForce RTX 3090": 0.02,
}

# GPU Price * 24 / TAO Price
REQUIRED_DEPOSIT_AMOUNT = {
    'NVIDIA B200': 0.223,
    'NVIDIA H200': 0.158,
    'NVIDIA H100 80GB HBM3': 0.103,
    'NVIDIA H100 NVL': 0.086,
    'NVIDIA H100 PCIe': 0.086,
    'NVIDIA H800 80GB HBM3': 0.051,
    'NVIDIA H800 NVL': 0.045,
    'NVIDIA H800 PCIe': 0.045,
    'NVIDIA GeForce RTX 5090': 0.014,
    'NVIDIA GeForce RTX 4090': 0.010,
    'NVIDIA GeForce RTX 4090 D': 0.008,
    # 'NVIDIA RTX 4000 Ada Generation': 0.009,
    'NVIDIA RTX 6000 Ada Generation': 0.017,
    'NVIDIA L4': 0.008,
    'NVIDIA L40S': 0.027,
    'NVIDIA L40': 0.024,
    # 'NVIDIA RTX 2000 Ada Generation': 0.005,
    'NVIDIA A100 80GB PCIe': 0.027,
    'NVIDIA A100-SXM4-80GB': 0.031,
    'NVIDIA RTX A6000': 0.018,
    'NVIDIA RTX A5000': 0.009,
    'NVIDIA RTX A4500': 0.008,
    'NVIDIA RTX A4000': 0.008,
    # 'NVIDIA A40': 0.008,
    # 'NVIDIA A30': 0.005,
    'NVIDIA GeForce RTX 3090': 0.008
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

MIN_PORT_COUNT = 3
BATCH_PORT_VERIFICATION_SIZE = 1000
VERIFY_JOB_REQUIRED_COUNT = 6 * 24 * 1

TOTAL_BURN_EMISSION = 0.8
BURNER_EMISSION = 0.01

IS_NOT_DEPOSITED_SCORE_MULTIPLIER = 0.5
DOCKER_DIND_IMAGE = "daturaai/dind:0.0.0"

HASHCAT_CONFIGS = {
    "NVIDIA RTX A5000": {
        "digits": 11,
        "average_time": [
            24.251156330108643,
            24.459399509429932,
            25.07683423360189,
            26.078879714012146,
            27.233995351791386,
            27.801182564099634,
            29.58513449941363,
            30.492227721214295,
        ],
    },
    "NVIDIA RTX A6000": {
        "digits": 11,
        "average_time": [
            24.251156330108643,
            24.459399509429932,
            25.07683423360189,
            26.078879714012146,
            27.233995351791386,
            27.801182564099634,
            29.58513449941363,
            30.492227721214295,
        ],
    },
    "NVIDIA RTX A4500": {
        "digits": 11,
        "average_time": [
            24.251156330108643,
            24.459399509429932,
            25.07683423360189,
            26.078879714012146,
            27.233995351791386,
            27.801182564099634,
            29.58513449941363,
            30.492227721214295,
        ],
    },
    "NVIDIA RTX A4000": {
        "digits": 11,
        "average_time": [
            32.62807669639587,
            33.436131143569945,
            33.88327717781067,
            34.187138891220094,
            35.52240489006042,
            37.14521159331004,
            39.016253103528705,
            40.42734135985374,
        ],
    },
    "NVIDIA GeForce RTX 3090": {
        "digits": 11,
        "average_time": [
            22.13358383178711,
            24.477075362205504,
            26.968040720621747,
            29.163842380046844,
            31.934904451370237,
            34.341850678126015,
            37.18430421011789,
            39.15856931209564,
        ],
    },
    "NVIDIA RTX 6000 Ada Generation": {
        "digits": 11,
        "average_time": [
            12.016858005523682,
            13.232668924331666,
            14.015261713663739,
            14.904895508289338,
            15.89838502883911,
            16.701006396611533,
            18.079130056926182,
            19.553341883420945,
        ],
    },
    "NVIDIA L40S": {
        "digits": 11,
        "average_time": [
            10.906689882278442,
            9.32911479473114,
            12.892356348037719,
            13.338897478580474,
            14.28122389793396,
            15.280945293108621,
            15.630833080836705,
            17.76026642918587,
        ],
    },
    "NVIDIA L40": {
        "digits": 11,
        "average_time": [
            10.906689882278442,
            9.32911479473114,
            12.892356348037719,
            13.338897478580474,
            14.28122389793396,
            15.280945293108621,
            15.630833080836705,
            17.76026642918587,
        ],
    },
    "NVIDIA L4": {
        "digits": 11,
        "average_time": [
            27.768908500671387,
            27.90283513069153,
            27.773880004882812,
            27.653605222702026,
            27.88539433479309,
            27.88539433479309,
            27.88539433479309,
            27.88539433479309,
        ],
    },
    "NVIDIA RTX 4000 Ada Generation": {
        "digits": 11,
        "average_time": [
            23.84185085296631,
            25.37116765975952,
            25.933285299936934,
            27.255381512641907,
            28.95430653572082,
            30.480634721120204,
            32.16756559780665,
            33.507733607292174,
        ],
    },
    "NVIDIA H100 PCIe": {
        "digits": 11,
        "average_time": [
            18.3540611743927,
            17.581688284873962,
            19.558610963821412,
            23.779386079311372,
            25.929840545654294,
            28.815886704126996,
            29.60572577885219,
            33.850944715738294,
        ],
    },
    "NVIDIA H100 NVL": {
        "digits": 11,
        "average_time": [
            18.3540611743927,
            17.581688284873962,
            19.558610963821412,
            23.779386079311372,
            25.929840545654294,
            28.815886704126996,
            29.60572577885219,
            33.850944715738294,
        ],
    },
    "NVIDIA H100 80GB HBM3": {
        "digits": 11,
        "average_time": [
            18.3540611743927,
            17.581688284873962,
            19.558610963821412,
            23.779386079311372,
            25.929840545654294,
            28.815886704126996,
            29.60572577885219,
            33.850944715738294,
        ],
    },
    "NVIDIA A100 80GB PCIe": {
        "digits": 11,
        "average_time": [
            18.69497232437134,
            20.42860324382782,
            22.53571968078613,
            25.373827075958253,
            26.749426555633544,
            31.196198654174804,
            32.80575948442732,
            37.11309432387352,
        ],
    },
    "NVIDIA A100-SXM4-80GB": {
        "digits": 11,
        "average_time": [
            18.69497232437134,
            20.42860324382782,
            22.53571968078613,
            25.373827075958253,
            26.749426555633544,
            31.196198654174804,
            32.80575948442732,
            37.11309432387352,
        ],
    },
    "NVIDIA A40": {
        "digits": 11,
        "average_time": [
            22.828101253509523,
            23.189609861373903,
            21.3694882551829,
            23.657343721389772,
            28.178246479034424,
            27.75535701115926,
            30.86851720128741,
            34.388632106781,
        ],
    },
    "NVIDIA A30": {
        "digits": 11,
        "average_time": [
            22.828101253509523,
            23.189609861373903,
            21.3694882551829,
            23.657343721389772,
            28.178246479034424,
            27.75535701115926,
            30.86851720128741,
            34.388632106781,
        ],
    },
    "NVIDIA RTX 2000 Ada Generation": {
        "digits": 11,
        "average_time": [
            22.828101253509523,
            23.189609861373903,
            21.3694882551829,
            23.657343721389772,
            28.178246479034424,
            27.75535701115926,
            30.86851720128741,
            34.388632106781,
        ],
    },
    "NVIDIA GeForce RTX 4090 D": {
        "digits": 11,
        "average_time": [
            12.535813426971435,
            13.367040371894836,
            14.397390270233155,
            15.773727321624756,
            16.52033654212952,
            18.87070236206055,
            20.572682762145995,
            22.169760519266127,
        ],
    },
    "NVIDIA GeForce RTX 4090": {
        "digits": 11,
        "average_time": [
            11.02204384803772,
            11.871551060676575,
            12.621799103418986,
            13.46524715423584,
            14.425264406204224,
            12.915648317337036,
            16.706109033312117,
            17.858580154180526,
        ],
    },
    "NVIDIA GeForce RTX 5090": {
        "digits": 11,
        "average_time": [
            12.805432319641113,
            12.805432319641113,
            12.805432319641113,
            13.46524715423584,
            14.425264406204224,
            12.915648317337036,
            16.706109033312117,
            17.858580154180526,
        ],
    },
    "NVIDIA H200": {
        "digits": 11,
        "average_time": [
            13.78846188,
            13.20821786,
            14.69337816,
            17.86422935,
            19.47975516,
            21.64789315,
            22.24125861,
            25.43047319,
        ],
    },
}

LIB_NVIDIA_ML_DIGESTS = {
    "535.183.01": "58fc46eefa8ebb265293556951a75a39:67185f510159acdc8f38b768b059bfb0f3ec5869baaffd1dc1c949e52012b18f",
    "535.183.06": "03ed7fa2134095b32f9d0d24a774c6ba:5899d928c18f39656d4c5a573a509acaf621b896644abe31e78ad715171c2ce6",
    "535.216.01": "96479a06139fc5261d06f432970d6a7b:63ec13c213f50fc193f1b56f4d56ea1dddb5974d505d3dfc670a10824d800753",
    "535.216.03": "189634bf960b9a2efe1af8011d27ccf7:f9aec03b89ff0fba865b7059d8dce3258f2fda8f5ae748b3da98a65b5e06b46e",
    "535.230.02": "cc34ae85c2238b9a49067e683c1998cf:23767b1f9a39ace459e4d9950e1225ba6d89f4ac0299de9e7815b5c413f53acf",
    "535.247.01": "b44440a3031606c8034b7f4b73f88e9c:8355dd1d32973e66cb83083be2937d59ce30d794ee59c7def5df3c33dc790f5a",
    "545.23.06": "5ad33588e91af67139efb54fe9fefc68:7864c8334d3d8b9c092e3f6f89b288c50fa3c517cffeec7397a3dbc23de62098",
    "545.29.06": "85ad949d7553ab96cce5c811e229c7c7:f2ad84109af6facf93cb33f9ca79354c2c4672ed7d90a9e09a00954c6a82b438",
    "550.120": "48be49d0e792b5ee76f73857c0bef35a:3bc49d033d45882a0e71a6e99a64105ad1a4f00cdad06b33bfd37175b8584373",
    "550.127.05": "bfa2733eee442016792bcbf130156e3d:f196afa282f435e48b19561849d451e3de26932af8a76f75136f1dfc5533f247",
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
    "575.51.03": "d01d04bf5e770102cbf9fe3c2302d903:b2910dea374c6c038f860b3ebf9e1408ee5946dd5da59cab6023220f968e8eff",
    "575.57.08": "b49c325ff0d74199597d9b19b9b407a6:ba4159cd8797a30e275e46a17b174b3d9a4a7768f82da9ca7263832da3f6d816",
    "575.64.03": "ffad1bd4cfbf8abe4f2edd24e5687c64:6e2e60d6f2ad3957f33465626e7793286c741cf8a9e3a6da8fbaacf3fc453370",
    "580.82.07": "f7e2623164324de58f1c3a5daf6ed475:008cc04afc61b8317ed4ddccfa2d1bd367ceb6e71334eb352dc846e0ac450340",
    "580.82.09": "175e6a907fbc54f344223f7da3164d19:68d2ee4a36fad63bf0507718376b67f9247de8402c041e7e68f63f29f536299c",
    "580.95.05": "abb3628879c4801b3e5f5e3351d01c96:9933cfe943ca96d9b3c352221154dd0afd5ed5406fcb9a3d15935ff4f2c380a9",
}

DOCKER_DIGESTS = {
    "26.1.3": "52d8fcc2c4370bf324cdf17cbc586784:90ca53cb65a70096c5767941c4518f26b1e350f903d134c0a0667c1477532ee9",
}

DOCKER_DIGEST = "52d8fcc2c4370bf324cdf17cbc586784:90ca53cb65a70096c5767941c4518f26b1e350f903d134c0a0667c1477532ee9"
PYTHON_DIGEST = '805bdc61166863665f1ecfc80aca749c:a9db1b429b10a90962ac3600f3952c0ec2c6cb7e61a2ac6c8c9cc8f14d6884f4'

# VerifyX Validation Constants
MEMORY_ALLOCATION_PERCENTAGE = 75
MEMORY_MIN_TEST_GB = 8
MEMORY_MAX_TEST_GB = 128
STORAGE_MIN_AVAILABLE_GB = 100
STORAGE_THROUGHPUT_TEST_GB = 5
NETWORK_TIMEOUT_SECONDS = 120
NETWORK_MIN_DOWNLOAD_SPEED_MBPS = 50.0

PREFERRED_POD_PORTS = [22, 20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009]
