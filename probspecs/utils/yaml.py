#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import ruamel.yaml
from frozendict import frozendict
from collections import OrderedDict


# dump ordered maps as simple YAML mappings from:
# https://stackoverflow.com/questions/53874345/how-do-i-dump-an-ordereddict-out-as-a-yaml-file
class Representer(ruamel.yaml.RoundTripRepresenter):
    pass


ruamel.yaml.add_representer(
    frozendict, Representer.represent_dict, representer=Representer
)
ruamel.yaml.add_representer(
    OrderedDict, Representer.represent_dict, representer=Representer
)


yaml = ruamel.yaml.YAML()
yaml.Representer = Representer
yaml.Constructor = ruamel.yaml.SafeConstructor
yaml.compact_seq_map = True
yaml.sequence_dash_offset = 0
