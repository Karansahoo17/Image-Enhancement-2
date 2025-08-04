"""Registry for architectures"""

class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, ' f'items={list(self._module_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def register(self, module=None, module_name=None):
        def _register(module):
            if module_name is None:
                name = module.__name__
            else:
                name = module_name
            if name in self._module_dict:
                raise KeyError(f'{name} is already registered in {self.name}')
            self._module_dict[name] = module
            return module
        
        if module is not None:
            return _register(module)
        return _register

# Create architecture registry
ARCH_REGISTRY = Registry('arch')
