import os

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

MIN_LEVEL = 30


"""
Set logging threshold on current logger. Lower levels produce more verbose outputs
and should only be used during development.
"""
def set_level(level):
    global MIN_LEVEL
    MIN_LEVEL = level

def debug(msg, *args):
    if MIN_LEVEL <= DEBUG:
        print('%s: %s'%('DEBUG', msg % args))

def info(msg, *args):
    if MIN_LEVEL <= INFO:
        print('%s: %s'%('INFO', msg % args))

def warn(msg, *args):
    if MIN_LEVEL <= WARN:
        warnings.warn(colorize('%s: %s'%('WARN', msg % args), 'yellow'))

def error(msg, *args):
    if MIN_LEVEL <= ERROR:
        print(colorize('%s: %s'%('ERROR', msg % args), 'red'))
