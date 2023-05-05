from gymnasium.envs.registration import register


def register_env():
    register(
        id="Parking-v0",
        entry_point="parking_env.parking:Parking",
    )
