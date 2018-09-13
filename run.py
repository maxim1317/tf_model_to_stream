import os, sys
import time
import subprocess
import signal
import platform
import webbrowser
from termcolor import colored
import colorama

colorama.init()

# set system/version dependent "start_new_session" analogs
kwargs = {}
if platform.system() == 'Windows':
    # from msdn [1]
    CREATE_NEW_PROCESS_GROUP = 0x00000200  # note: could get it from subprocess
    DETACHED_PROCESS = 0x00000008          # 0x8 | 0x200 == 0x208
    kwargs.update(creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)  
elif sys.version_info < (3, 2):  # assume posix
    kwargs.update(preexec_fn=os.setsid)
else:  # Python 3.2+ and Unix
    kwargs.update(start_new_session=True)

front_path = os.path.abspath('../dron-demo-stream/')
back_path  = os.path.abspath('../dron-demo-back/')
back_war   = os.path.abspath(os.path.join(back_path, 'target\\dron-demo-0.0.1-SNAPSHOT.war'))
cv_path    = os.path.abspath('./')

front_call = 'npm run serve'
back_call  = 'java -jar'
back_comp  = 'mvn clean install'
cv_call    = 'python tf_to_stream.py'


print(front_path)
print(back_path)
print(cv_path)
print(back_war)

def spc(amount=0):
    '''
        spc(amount=0)
        ---
        Returns:
        <amount>_of_spaces for formating purposes     
    '''
    return ' ' * amount

def check_dir(file_path):
    '''
        ensure_dir(file_path)
        ---
        Makes path to <file_path> if it doesn't exist

    '''
    directory = os.path.dirname(file_path)
    if os.path.exists(directory):
        return 0
    return 1

def signal_handler(procs):
        print('You pressed Ctrl+C!')
        for proc in procs:
            proc.kill()
        sys.exit()

def run_proc(call, path):
	try:
	    p = subprocess.Popen(call, cwd=path, shell=True, **kwargs)
	    assert not p.poll()
	    return p
	except OSError as e:
	    print("Execution failed:", e)

def compile_back():
	run_proc(back_comp, back_path)

def preparation():

	err = 0

	print(' Checking Directories...\n')
	if check_dir(front_path) == 0:
		print(colored(' Looking for Front...' + spc(23) + '[ ' , color='white') + colored('  OK  ', color='green') + colored(' ]' , color='white'))
	else:
		print(colored(' Looking for Front...' + spc(23) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
		print(colored(' Path ', color='yellow'), front_path, colored(' not found', color='yellow'))
		err += 1

	if check_dir(back_path) == 0:
		print(colored(' Looking for Back...' + spc(24) + '[ ' , color='white') + colored('  OK  ', color='green') + colored(' ]' , color='white'))
		if os.path.isfile(back_war):
			print(colored(' -> Looking for Back Executable...' + spc(10) + '[ ' , color='white') + colored('  OK  ', color='green') + colored(' ]' , color='white'))
		else:
			print(colored(' -> Looking for Back Executable...' + spc(10) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
			print(colored(' -> Compiling...', color='yellow'))
			print()
			compile_back()
			print()
	else:
		print(colored(' Looking for Back...' + spc(24) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
		print(colored(' Path ', color='yellow'), back_path, colored(' not found', color='yellow'))
		err += 1

	if check_dir(cv_path) == 0:
		print(colored(' Looking for CV...' + spc(26) + '[ ' , color='white') + colored('  OK  ', color='green') + colored(' ]' , color='white'))
	else:
		print(colored(' Looking for CV...' + spc(26) + '[ ' , color='white') + colored('FAILED', color='red') + colored(' ]' , color='white'))
		print(colored(' Path ', color='yellow'), cv_path, colored(' not found', color='yellow'))
		err += 1

	if err > 0:
		print(colored('\n Errors found. Check paths manually.'))
		exit(1)

def running():
	procs = []

	print('Starting Demonstration...\n')

	print('\n#######################################################\n')
	print(colored(' Starting Front...', color='yellow'))
	print('\n#######################################################\n')
	procs.append(run_proc(front_call, front_path))
	time.sleep(1)

	print('\n#######################################################\n')
	print(colored(' Starting Back...', color='yellow'))
	# print(back_call + ' ' + os.path.basename(back_war))
	procs.append(run_proc(back_call + ' ' + os.path.basename(back_war), os.path.dirname(back_war)))
	time.sleep(1)

	print('\n#######################################################\n')
	print(colored(' Starting CV...', color='yellow'))
	procs.append(run_proc(cv_call, cv_path))

	return procs

def open_url(url):
    if sys.platform=='win32':
        webbrowser.get("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s").open(url)
    elif sys.platform=='darwin':
        subprocess.Popen(['open', url])
    else:
        try:
            subprocess.Popen(['xdg-open', url])
        except OSError:
            print('Please open a browser on: ' + url)

procs = []

url = 'http://localhost:8080'

try:
    print('\n#######################################################\n')
    preparation()
    print('\n#######################################################\n')
    procs = running()
    print('\n#######################################################\n')

    time.sleep(7)
    open_url(url)

    exit_codes = [p.wait() for p in procs]
	
except KeyboardInterrupt:
    
    for proc in procs:
        proc.kill()
    sys.exit()

signal.signal(signal.SIGINT, signal_handler(procs))