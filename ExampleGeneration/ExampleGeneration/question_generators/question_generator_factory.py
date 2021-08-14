import os, json
import sys
from setuptools import find_packages
from pkgutil import iter_modules
import logging
logger = logging.getLogger(__name__) # pylint: disable=invalid-name


class QGenFactory:
    def __init__(self):
        pass

    def upper_to_lower_notation_name(self, qgen_name):
        return ''.join(['_' + c.lower()  if c.isupper() else c for c in qgen_name ])[1:]

    def find_qgen(self, path, callange_to_find):
        modules = list()
        for pkg in [''] + find_packages(path):
            pkgpath = path + '/' + pkg.replace('.', '/')
            if sys.version_info.major == 2 or (sys.version_info.major == 3 and sys.version_info.minor < 6):
                for _, name, ispkg in iter_modules([pkgpath]):
                    if not ispkg:
                        modules.append(pkg + '.' + name)
            else:
                for info in iter_modules([pkgpath]):
                    if not info.ispkg:
                        modules.append(pkg + '.' + info.name)

        found_qgen = [module for module in modules if module.find(callange_to_find) > -1]
        if len(found_qgen) > 0:
            found_qgen = found_qgen[0]
            if found_qgen.startswith('.'):
                found_qgen =  found_qgen[1:]
        else:
            found_qgen = None

        return found_qgen

    def get_qgen(self, qgen_name, template, args):
        qgen_name_lower = self.upper_to_lower_notation_name(qgen_name)
        module_name = self.find_qgen(os.path.dirname(os.path.abspath(__file__)), qgen_name_lower)
        try:
            mod = __import__('ExampleGeneration.question_generators.' + module_name, fromlist=[qgen_name])
        except:
            logger.error(module_name + ' module not found!!')
            assert (ValueError('qgen_name not found!'))

        return getattr(mod, qgen_name)(template, args)

    def create_new_qgen(self, qgen_name, qgen_module, copy_from, args):
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/qgens')
        copy_from_lower = self.upper_to_lower_notation_name(copy_from)
        qgen_name_lower = self.upper_to_lower_notation_name(qgen_name)
        copy_from_module = self.find_qgen(os.getcwd(), copy_from_lower)
        if copy_from_module is None:
            assert (ValueError('copy_from qgen not found!'))
        copy_from_path = copy_from_module.replace('.',os.sep) + '.py'

        if not os.path.isdir(qgen_module):
            os.mkdir(qgen_module)
            open(os.path.join(qgen_module,'__init__.py'), 'a').close()

        with open(copy_from_path,'r') as f:
            copied_qgen_txt =  f.read()
            copied_qgen_txt = copied_qgen_txt.replace(copy_from, qgen_name)

        if len(qgen_module) > 0:
            new_qgen_path = os.path.join(qgen_module, qgen_name_lower) + '.py'
        else:
            new_qgen_path = qgen_name_lower + '.py'
        with open(new_qgen_path, 'w') as f:
            f.write(copied_qgen_txt)

        # duplicating the test
        os.chdir('../../tests/qgens')
        if not os.path.isdir(qgen_module):
            os.mkdir(qgen_module)
        with open(copy_from_path.replace('.py','_test.py'),'r') as f:
            copied_qgen_txt =  f.read()
            copied_qgen_txt = copied_qgen_txt.replace('qgens.' + copy_from_module, \
                                                                'qgens.' + qgen_module + '.' + qgen_name_lower)
            copied_qgen_txt = copied_qgen_txt.replace(copy_from, qgen_name)
            copied_qgen_txt = copied_qgen_txt.replace(copy_from_lower, qgen_name_lower)

        if len(qgen_module) > 0:
            new_qgen_path = os.path.join(qgen_module, qgen_name_lower) + '_test.py'
        else:
            new_qgen_path = qgen_name_lower + '_test.py'
        with open(new_qgen_path, 'w') as f:
            f.write(copied_qgen_txt)

        # adding to config file:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations",
                               args.config_file_name) , 'r') as f:
            config = json.load(f)
        config[qgen_name] = config[copy_from]
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations",
                               args.config_file_name) , 'w') as f:
            json.dump(config, f ,indent=4)
