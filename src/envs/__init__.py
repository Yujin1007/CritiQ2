import gymnasium
from envs.create_env import make_env
from envs.push.single import SinglePush
from envs.stretch import BaseStretchEnv
from envs.clutter.stretch_multi_obj import StretchMultiObjectEnv
from envs.drawer.stretch_drawer import StretchDrawer

gymnasium.register(
    "singlepush",
    "envs:SinglePush",
)

# gymnasium.register(
#     "StretchMultiObjectEnv",
#     "envs:StretchMultiObjectEnv",
# )

gymnasium.register(
    "StretchDrawer",
    "envs:StretchDrawer",
)
