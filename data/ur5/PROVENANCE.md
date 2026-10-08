# Where this data came from

    dataset   DORLR/ur5_ur5_force_sensor_test
    host      https://huggingface.co/datasets/DORLR/ur5_ur5_force_sensor_test
    fetched   2026-10-11
    format    LeRobot v2 (meta/info.json, meta/tasks.parquet, data/chunk-000/episode_*.parquet)
    selected  episodes 0-3 of 4, columns timestamp / observation.state / observation.wrench
    left out  observation.images.*  -- the camera streams, ~100x the size of everything
                                    used here and not needed for a force study

    robot     UR5 arm, force/torque sensor at the wrist
    rate      30 Hz
    per ep    20 s, 600-601 frames
    state     7 channels: 6 arm joints + robotiq_85_left_knuckle_joint (gripper)
    wrench    6 channels: fx, fy, fz, tx, ty, tz

The six force channels are declared in the dataset's own `meta/info.json` and
`csid/data.py` asserts against that declaration at load time, so a future change to the
upstream column order fails loudly instead of silently studying the wrong quantity.

The staged copy is byte-identical to the upstream parquet files, restricted to the three
columns used. SHA-256 of each staged file:

    5fc2d9812c5709e8262639ce55fdc4f1eab08ee06bf2d78723b64d1bd594bdbc  ep0.parquet
    b6722bc090e348d30f95d0de371be1691cf56b1eeb31acba05bc86dd85fb6299  ep1.parquet
    5e1d8a9196c3e0af461c77d4ab2976803d7635041c7607a10cb64b178b794968  ep2.parquet
    26022ed821c565a0ed53b57be2a502e0e8952c17d99c5d1bb2189e34c8c67d98  ep3.parquet
    e52a6b3a699181383df8a48f2e0cb0f485634e82f361a7c64a5348860287a1ab  info.json
    071b506f644a9692a6bbc761c17da6e3ec596b2aea552e4d4e997835afcc2e39  tasks.parquet
