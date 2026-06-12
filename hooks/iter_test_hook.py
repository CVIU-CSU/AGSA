from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class IterTestHook(Hook):
    def after_val(self, runner: Runner):
        runner.test()
