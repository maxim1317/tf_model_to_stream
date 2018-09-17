import os
# import shutil

from winshell import CreateShortcut

from utils.misc import ensure_dir
# from consts import reg_path

drone_bat_path = 'etc/Drone_Demo.bat'

ensure_dir(drone_bat_path)

drone_bat = open(drone_bat_path, 'w')

drone_bat_path = os.path.abspath(drone_bat_path)

drone_bat.write('ECHO ON\n')
drone_bat.write('rem A batch script to execute a Python script\n')
drone_bat.write('cd ' + os.path.abspath('.') + '\n')
drone_bat.write('python run.py\n')
drone_bat.write('PAUSE\n')

drone_bat.close()

# shutil.copy(drone_bat_path, os.path.join(os.environ["HOMEPATH"], "Desktop"))
CreateShortcut(Path=os.path.join(os.path.join(os.environ["HOMEPATH"], "Desktop"), "Drone Demo.lnk"), Target=drone_bat_path, Icon=(os.path.abspath('ico/sibintek.ico'), 0), Description="Run Drone Demo")

reg_bat_path = 'etc/Registration.bat'

ensure_dir(reg_bat_path)

reg_bat = open(reg_bat_path, 'w')

reg_bat_path = os.path.abspath(reg_bat_path)

reg_bat.write('ECHO ON\n')
reg_bat.write('rem A batch script to execute a Python script\n')
reg_bat.write('cd ' + os.path.abspath('.') + '\n')
reg_bat.write('python run.py -r\n')
reg_bat.write('PAUSE\n')

reg_bat.close()

# shutil.copy(reg_bat_path, os.path.join(os.environ["HOMEPATH"], "Desktop"))
CreateShortcut(Path=os.path.join(os.path.join(os.environ["HOMEPATH"], "Desktop"), "Registration.lnk"), Target=reg_bat_path, Icon=(os.path.abspath('ico/sibintek.ico'), 0), Description="Run Registration")