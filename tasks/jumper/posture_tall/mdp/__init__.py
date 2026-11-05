"""MDP terms specific to jumper.posture_tall.

Only what differs from `jumper.posture`: a posture command that leans its height
draws towards the top of the range, and one reward term. Everything else is
imported from `tasks/jumper/posture/mdp/`.
"""

from . import commands, rewards

__all__ = ["commands", "rewards"]
