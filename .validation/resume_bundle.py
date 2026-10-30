"""Offline recovery experiment using the public Grain/Orbax checkpoint bridge."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile

import grain
import orbax.checkpoint as ocp
from orbax.checkpoint import test_utils


class Unbatch(grain.experimental.FlatMapTransform):
    def flat_map(self, element):
        return element


dataset = grain.experimental.FlatMapIterDataset(
    grain.MapDataset.source([[0, 1], [2, 3]]).to_iter_dataset(), Unbatch()
)
iterator = iter(dataset)
assert list(iterator) == [0, 1, 2, 3]
options = ocp.CheckpointManagerOptions(max_to_keep=1)
with tempfile.TemporaryDirectory() as root:
    with ocp.CheckpointManager(root, options=options) as manager:
        manager.save(
            0,
            args=ocp.args.Composite(
                model=ocp.args.PyTreeSave({'value': 7}),
                data_iter=grain.checkpoint.CheckpointSave(iterator),
            ),
        )
        manager.wait_until_finished()

    with ThreadPoolExecutor(max_workers=1) as executor:
        handlers = {
            'model': test_utils.ErrorCheckpointHandler(
                ocp.PyTreeCheckpointHandler(), executor
            ),
            'data_iter': grain.checkpoint.CheckpointHandler(),
        }
        with ocp.CheckpointManager(root, item_handlers=handlers, options=options) as manager:
            manager.save(
                1,
                args=ocp.args.Composite(
                    model=test_utils.ErrorSaveArgs({'value': 8}),
                    data_iter=grain.checkpoint.CheckpointSave(iterator),
                ),
            )
            try:
                manager.wait_until_finished()
            except SystemError:
                pass
            else:
                raise AssertionError('Injected commit failure was not propagated')

            result = {
                'grain_source': grain.__file__,
                'orbax_source': ocp.__file__,
                'history': manager.all_steps(),
                'latest': manager.latest_step(),
                'checkpoint_0_exists': (Path(root) / '0').is_dir(),
            }
            restore_args = ocp.args.Composite(
                model=test_utils.ErrorRestoreArgs(),
                data_iter=grain.checkpoint.CheckpointRestore(iterator),
            )
            try:
                restored = manager.restore(None, args=restore_args)
            except FileNotFoundError as error:
                result['latest_restore_error'] = str(error)
                restored = manager.restore(0, args=restore_args)
                result['restored_via'] = 'explicit_step_0_control'
            else:
                result['restored_via'] = 'latest'
            result['model_value'] = restored['model']['value']
            result['iterator_identity_preserved'] = restored['data_iter'] is iterator
            result['records_after_restore'] = list(iterator)
            assert result['model_value'] == 7
            assert result['checkpoint_0_exists']
            assert result['iterator_identity_preserved']
            print(json.dumps(result, sort_keys=True), flush=True)
