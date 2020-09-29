#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import parlai.utils.testing as testing_utils
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args


class TestImageChatTeacher(unittest.TestCase):
    """
    Basic test for an example of the Image-Chat dataset.
    """

    def test_check_examples(self):

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir

            # Check the first entry of the first episode for each datatype
            opts_and_examples = [
                ({'datatype': 'train'}, {}),
                (
                    {'datatype': 'valid'},
                    {
                        'label_candidates': {
                            'num_cands': 100,
                            'first': 'i work as a vet so no days off over here!',
                            'last': 'And what else? ',
                        }
                    },
                ),
                (
                    {'datatype': 'test'},
                    {
                        'label_candidates': {
                            'num_cands': 100,
                            'first': "Wow, engineering, sounds impressive.  I'm sure the income will be awesome.",
                            'last': 'but the worst part is you have to clean every day and keep the flat tidy all the time.  ',
                        }
                    },
                ),
            ]
            for kwargs, example in opts_and_examples:
                all_kwargs = {
                    **kwargs,
                    'task': 'image_chat',
                    'datapath': data_path,
                    'category_frac': 0.75,
                }
                parser = setup_args()
                parser.set_defaults(**all_kwargs)
                opt = parser.parse_args([])
                agent = RepeatLabelAgent(opt)
                teacher = create_task(opt, agent).get_task_agent()
                actual_message = teacher.get(episode_idx=1, entry_idx=1)

                # Check for field equality
                self.assertEqual(set(actual_message.keys()), set(example.keys()))

                # Check label candidates
                if 'label_candidates' in example:
                    params = example['label_candidates']
                    self.assertEqual(
                        len(actual_message['label_candidates']), params['num_cands']
                    )
                    self.assertEqual(
                        actual_message['label_candidates'][0], params['first']
                    )
                    self.assertEqual(
                        actual_message['label_candidates'][-1], params['last']
                    )

                # Check other fields
                for key in [k for k in example.keys() if k != 'label_candidates']:
                    self.assertEqual(example[key], actual_message[key])


if __name__ == '__main__':
    unittest.main()
