import os, json
import sys
from setuptools import find_packages
from pkgutil import iter_modules

class DataJobFactory:
    def __init__(self):
        pass

    def upper_to_lower_notation_name(self, datajob_name):
        return ''.join(['_' + c.lower()  if c.isupper() else c for c in datajob_name ])[1:]

    def find_datajob(self, path, callange_to_find):
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

        found_datajob = [module for module in modules if module.find(callange_to_find) > -1]
        if len(found_datajob) > 0:
            found_datajob = found_datajob[0]
            if found_datajob.startswith('.'):
                found_datajob =  found_datajob[1:]
        else:
            found_datajob = None

        return found_datajob

    def get_datajob(self, datajob_name, datajob_type, args):
        datajob_type_lower = self.upper_to_lower_notation_name(datajob_type)
        module_name = self.find_datajob(os.path.dirname(os.path.abspath(__file__)) + '/datajobs', datajob_type_lower)
        try:
            mod = __import__('datajobs.' + module_name, fromlist=[datajob_type])
        except:
            assert (ValueError('datajob_name not found!'))

        return getattr(mod, datajob_type + 'DataJob')(datajob_name, args)

    def create_new_datajob(self, datajob_name, copy_from, args):
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/datajobs')
        copy_from_lower = self.upper_to_lower_notation_name(copy_from)
        datajob_name_lower = self.upper_to_lower_notation_name(datajob_name)
        copy_from_module = self.find_datajob(os.getcwd(), copy_from_lower)
        if copy_from_module is None:
            assert (ValueError('copy_from datajob not found!'))
        copy_from_path = copy_from_module.replace('.',os.sep) + '.py'

        with open(copy_from_path,'r') as f:
            copied_datajob_txt =  f.read()
            copied_datajob_txt = copied_datajob_txt.replace(copy_from, datajob_name)

        new_datajob_path = datajob_name_lower + '.py'
        with open(new_datajob_path, 'w') as f:
            f.write(copied_datajob_txt)

        # duplicating the test
        os.chdir('../../tests/datajobs')
        with open(copy_from_path.replace('.py','_test.py'),'r') as f:
            copied_datajob_txt =  f.read()
            copied_datajob_txt = copied_datajob_txt.replace('datajobs.' + copy_from_module, \
                                                                'datajobs.' + datajob_name_lower)
            copied_datajob_txt = copied_datajob_txt.replace(copy_from, datajob_name)
            copied_datajob_txt = copied_datajob_txt.replace(copy_from_lower, datajob_name_lower)

        new_datajob_path = datajob_name_lower + '_test.py'
        with open(new_datajob_path, 'w') as f:
            f.write(copied_datajob_txt)

        # adding to config file:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations",
                               args.config_file_name) , 'r') as f:
            config = json.load(f)
        config[datajob_name] = config[copy_from]
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configurations",
                               args.config_file_name) , 'w') as f:
            json.dump(config, f ,indent=4)
